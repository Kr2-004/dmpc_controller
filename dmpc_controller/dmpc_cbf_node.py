#!/usr/bin/env python3
import time
import numpy as np
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Float32MultiArray

import casadi as ca

def quat_to_yaw(qz: float, qw: float)  -> float:
    return 2.0 * np.arctan2(qz, qw)

class DMPCNode(Node):
    def __init__(self) -> None:
        super().__init__('dmpc_node')

        # --- Robot parameters ---
        self.r_wheel: float = 0.05
        self.b_track: float = 0.085
        self.v_max: float   = 0.20 
        self.w_max: float   = 6.0
        self.omega_max: float = 7.0

        # --- MPC parameters ---
        self.Ts: float = 0.01
        self.N: int    = 50

        # Weights
        self.Qp  = np.diag([10.0 , 25.0])
        self.Qth = 2.0
        self.Qv  = 0.2
        self.R   = np.diag([1.0, 0.5])
        self.Sdu = 1.5

        # Stop condition
        self.x_stop: float = 3.0

        # State & reference
        self.x_current: Optional[np.ndarray] = None
        self.ref: Optional[np.ndarray]       = None
        self.u_last = np.array([0.05, 0.0])
        self.start_wall_time = time.time()
        self.v_ref: float=0.25

        # --- Warm start ---
        self.U_prev: Optional[np.ndarray] = None

        # Build solver (pure MPC)
        self._build_solver()

        # ROS params
        self.declare_parameter('robot_name', 'puzzlebot1')
        robot_name = self.get_parameter('robot_name').get_parameter_value().string_value

        # --- Subscriptions ---
        self.sub_pose = self.create_subscription(
            PoseStamped, f'/vicon/{robot_name}/{robot_name}/pose', self.pose_cb, 20)
        self.sub_vs = self.create_subscription(
            PoseStamped, f'/vs/reference/{robot_name}', self.vs_cb, 20)
        self.sub_vref = self.create_subscription(
            Float32, '/vs/reference_speed', self.vref_cb, 10)

        # --- Publishers ---
        self.pub_L = self.create_publisher(Float32, f'/{robot_name}/VelocitySetL', 10)
        self.pub_R = self.create_publisher(Float32, f'/{robot_name}/VelocitySetR', 10)

        self.timer = self.create_timer(self.Ts, self.control_step)
        self.get_logger().info(f"✅ PURE MPC node ready for {robot_name}.")

    # ---------------- Callbacks ----------------
    def pose_cb(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        th = quat_to_yaw(msg.pose.orientation.z, msg.pose.orientation.w)
        self.x_current = np.array([x, y, th], dtype=float)

    def vs_cb(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        th = quat_to_yaw(msg.pose.orientation.z, msg.pose.orientation.w)
        self.ref = np.array([x, y, th], dtype=float)

    def vref_cb(self, msg: Float32):
        self.v_ref = float(msg.data)

    # ---------------- CasADi solver (PURE MPC) ----------------
    def _build_solver(self):
        nx, nu, N, Ts = 3, 2, self.N, self.Ts

        U = ca.SX.sym('U', nu * N)

        x0 = ca.SX.sym('x0', nx)
        ref = ca.SX.sym('ref', nx)
        u_last = ca.SX.sym('u_last', nu)
        v_ref = ca.SX.sym('v_ref')

        # Unicycle model
        def f(x, u):
            th = x[2]
            v, w = u[0], u[1]
            return ca.vertcat(
                x[0] + Ts * v * ca.cos(th),
                x[1] + Ts * v * ca.sin(th),
                x[2] + Ts * w
            )

        J = 0
        Xpred = []
        xk = x0
        for j in range(N):
            uj = U[j*nu:(j+1)*nu]
            xk = f(xk, uj)
            Xpred.append(xk)

            # --- Reference prediction ---
            ref_jx = ref[0] + v_ref * Ts * j * ca.cos(ref[2])
            ref_jy = ref[1] + v_ref * Ts * j * ca.sin(ref[2])
            ref_jth = ref[2]

            dx = xk[0] - ref_jx
            dy = xk[1] - ref_jy
            e_th = ca.fmod(xk[2] - ref_jth + ca.pi, 2*ca.pi) - ca.pi

            # Longitudinal / lateral errors
            e_long =  ca.cos(ref[2]) * dx + ca.sin(ref[2]) * dy
            e_lat  = -ca.sin(ref[2]) * dx + ca.cos(ref[2]) * dy

            # --- Cost terms ---
            J += self.Qp[0,0]*(e_long**2) + self.Qp[1,1]*(e_lat**2) + self.Qth*(e_th**2)
            J += ca.mtimes([uj.T, self.R, uj])

            # Adaptive Qv
            align_err = e_lat**2 + 0.2*(e_th**2)
            alpha_gate = ca.exp(-10.0 * align_err)
            v_des = alpha_gate * v_ref
            J += self.Qv * ((uj[0] - v_des)**2)

            # Δu penalty
            if j == 0:
                du = uj - u_last
            else:
                uj_prev = U[(j-1)*nu:j*nu]
                du = uj - uj_prev
            J += self.Sdu * ca.mtimes([du.T, du])

        # Constraints
        g = []
        for j in range(N):
            v, w = U[j*nu], U[j*nu+1]
            wl = (2*v - w*self.b_track) / (2*self.r_wheel)
            wr = (2*v + w*self.b_track) / (2*self.r_wheel)
            g += [v, self.v_max - v,
                  w + self.w_max, self.w_max - w,
                  self.omega_max - wl, self.omega_max + wl,
                  self.omega_max - wr, self.omega_max + wr]

        Z = U
        nlp = {
            'x': Z,
            'f': J,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(x0, ref, u_last, v_ref)
        }

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 40,
            'ipopt.tol': 1e-3
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.U_len = (nu * N)
        self.g_dim = len(g)

    # ---------------- Control loop ----------------
    def control_step(self):
        if self.x_current is None or self.ref is None:
            return

        if self.x_current[0] >= self.x_stop:
            self.pub_L.publish(Float32(data=0.0))
            self.pub_R.publish(Float32(data=0.0))
            self.get_logger().info("Reached stop condition (x >= 3.0 m).")
            return

        # Warm start
        if self.U_prev is not None and self.U_prev.shape[0] == self.U_len:
            U0 = np.hstack([self.U_prev[2:], self.U_prev[-2:]])
        else:
            U0 = np.tile(self.u_last, self.N)

        Z0 = U0

        params = np.concatenate([
            self.x_current, self.ref, self.u_last, [self.v_ref]
        ])

        try:
            sol = self.solver(
                x0=Z0,
                p=params,
                lbg=np.zeros(self.g_dim),
                ubg=np.full(self.g_dim, np.inf)
            )
            Ustar = np.array(sol['x']).flatten()
            u0 = Ustar[:2]
            self.U_prev = Ustar.copy()

        except Exception as e:
            self.get_logger().warn(f"Solver failed: {e}")
            u0 = np.array([0.05, 0.0])
            self.U_prev = None

        # Apply controls
        v_cmd = float(np.clip(u0[0], 0.0, self.v_max))
        w_cmd = float(np.clip(u0[1], -self.w_max, self.w_max))

        wl = (2*v_cmd - w_cmd*self.b_track) / (2*self.r_wheel)
        wr = (2*v_cmd + w_cmd*self.b_track) / (2*self.r_wheel)
        wl = np.clip(wl, -self.omega_max, self.omega_max)
        wr = np.clip(wr, -self.omega_max, self.omega_max)

        self.pub_L.publish(Float32(data=float(wl)))
        self.pub_R.publish(Float32(data=float(wr)))
        self.u_last = np.array([v_cmd, w_cmd])

        t = time.time() - self.start_wall_time
        self.get_logger().info(
            f"[t={t:4.1f}s] v={v_cmd:.3f}, w={w_cmd:.3f} | WL={wl:.2f}, WR={wr:.2f} | "
            f"x={self.x_current[0]:.2f}, y={self.x_current[1]:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_L.publish(Float32(data=0.0))
        node.pub_R.publish(Float32(data=0.0))
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
