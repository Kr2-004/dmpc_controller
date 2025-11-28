#!/usr/bin/env python3
import time
import numpy as np
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Float32MultiArray
import casadi as ca


def quat_to_yaw(qz: float, qw: float) -> float:
    return 2.0 * np.arctan2(qz, qw)


class DMPCNode(Node):
    def __init__(self) -> None:
        super().__init__("dmpc_node")

        # --- Robot parameters ---
        self.r_wheel = 0.05
        self.b_track = 0.085
        self.v_max = 0.25#0.8
        self.w_max = 2.0 #6.0
        self.omega_max = 5.0 #10.0

        # --- MPC parameters ---
        self.Ts = 0.01
        self.N = 50

        # Weights
        self.Qp = np.diag([1.5, 12.0])   # [longitudinal, lateral]
        self.Qth = 2.0
        self.Qv = 2.0
        self.R = np.diag([0.5, 0.5])
        self.Sdu = 5.0

        # --- CBF parameters ---
        self.R_robot = 0.20
        self.R_obs   = 0.20
        self.r_buf   = 0.10
        self.alpha_cbf = 200
        self.rho_slack = 1e5# keep affordable to prevent paralysis

        # --- Detour shaping weights (tuned) ---
        self.w_approach = 10.0#20.0   # penalize approaching (facing) component
        self.w_tangent  = 200.0   # encourage consistent detour side
        self.w_clear    = 200.0   # reward clearance (larger h)
        self.safe_margin = 0.12  # react close to obstacle
        self.near_k  = 20.0       #sharper near/on gating

        # Stop condition
        self.x_stop = 3.0

        # State & reference
        self.x_current: Optional[np.ndarray] = None
        self.ref: Optional[np.ndarray] = None
        self.u_last = np.array([0.05, 0.0])
        self.start_wall_time = time.time()
        self.v_ref = 0.25

        # Obstacle
        self.obstacle_xy: Optional[np.ndarray] = None
        self.U_prev: Optional[np.ndarray] = None

        # Build solver
        self._build_solver()

        # --- ROS I/O ---
        self.sub_pose = self.create_subscription(
            PoseStamped, "/vicon/puzzlebot2/puzzlebot2/pose", self.pose_cb, 20)
        self.sub_vs = self.create_subscription(
            PoseStamped, "/vs/reference/puzzlebot2", self.vs_cb, 20)
        self.sub_vref = self.create_subscription(
            Float32, "/vs/reference_speed", self.vref_cb, 10)
        self.sub_obs_pose = self.create_subscription(
            PoseStamped, "/vicon/Obstacle/Obstacle/pose", self.obstacle_cb, 10)

        self.pub_L = self.create_publisher(Float32, "/puzzlebot2/VelocitySetL", 10)
        self.pub_R = self.create_publisher(Float32, "/puzzlebot2/VelocitySetR", 10)
        self.pub_cbf = self.create_publisher(Float32MultiArray, "/cbf_monitor", 10)

        self.timer = self.create_timer(self.Ts, self.control_step)
        self.get_logger().info("✅ DMPC node with tangential detour + CBF relief ready (100 Hz)")

    # ---------------- Callbacks ----------------
    def pose_cb(self, msg: PoseStamped):
        self.x_current = np.array(
            [msg.pose.position.x,
             msg.pose.position.y,
             quat_to_yaw(msg.pose.orientation.z, msg.pose.orientation.w)],
            dtype=float)

    def vs_cb(self, msg: PoseStamped):
        self.ref = np.array(
            [msg.pose.position.x,
             msg.pose.position.y,
             quat_to_yaw(msg.pose.orientation.z, msg.pose.orientation.w)],
            dtype=float)

    def vref_cb(self, msg: Float32):
        self.v_ref = float(msg.data)

    def obstacle_cb(self, msg: PoseStamped):
        self.obstacle_xy = np.array([msg.pose.position.x, msg.pose.position.y], dtype=float)

    # ---------------- CasADi solver ----------------
    def _build_solver(self):
        nx, nu, N, Ts = 3, 2, self.N, self.Ts

        U = ca.SX.sym("U", nu * N)
        delta = ca.SX.sym("delta")

        x0 = ca.SX.sym("x0", nx)
        ref = ca.SX.sym("ref", nx)
        u_last = ca.SX.sym("u_last", nu)
        v_ref = ca.SX.sym("v_ref")
        obs = ca.SX.sym("obs", 2)
        cbf_on = ca.SX.sym("cbf_on")
        side_pref = ca.SX.sym("side_pref")  # +1 left / -1 right

        def f(x, u):
            th = x[2]
            v, w = u[0], u[1]
            return ca.vertcat(
                x[0] + Ts * v * ca.cos(th),
                x[1] + Ts * v * ca.sin(th),
                x[2] + Ts * w,
            )

        def relu(z):
            return 0.5 * (z + ca.fabs(z))

        J = 0
        Xpred = []
        xk = x0
        dmin = self.R_robot + self.R_obs + self.r_buf

        for j in range(N):
            uj = U[j * nu:(j + 1) * nu]
            xk = f(xk, uj)
            Xpred.append(xk)

            # --- Reference rollout ---
            ref_jx = ref[0] + v_ref * Ts * j * ca.cos(ref[2])
            ref_jy = ref[1] + v_ref * Ts * j * ca.sin(ref[2])
            ref_jth = ref[2]

            dx = xk[0] - ref_jx
            dy = xk[1] - ref_jy
            e_th = ca.fmod(xk[2] - ref_jth + ca.pi, 2 * ca.pi) - ca.pi
            e_long = ca.cos(ref[2]) * dx + ca.sin(ref[2]) * dy
            e_lat  = -ca.sin(ref[2]) * dx + ca.cos(ref[2]) * dy

            # Tracking & smoothness
            J += self.Qp[0, 0] * (e_long**2) + self.Qp[1, 1] * (e_lat**2) + self.Qth * (e_th**2)
            J += ca.mtimes([uj.T, self.R, uj])

            # --- Detour shaping + forward progression ---
            vj, thj = uj[0], xk[2]
            e_th_vec = ca.vertcat(ca.cos(thj), ca.sin(thj))

            r   = xk[0:2] - obs
            r2  = ca.mtimes([r.T, r])
            dist = ca.sqrt(r2 + 1e-9)
            r_hat = r / (dist + 1e-9)
            h = r2 - dmin**2

            near = ca.fmax(dist - (dmin + self.safe_margin), 0.0)
            near_gate = ca.exp(-self.near_k * near)   # (0,1], ~1 when close

            # geometry
            cos_toward  = -ca.mtimes([e_th_vec.T, r_hat])  # + when facing obstacle
            sin_tangent = r_hat[0]*e_th_vec[1] - r_hat[1]*e_th_vec[0]  # + CCW, - CW

            # (1) penalize approaching (only when facing)
            approach_pen = self.w_approach * near_gate * relu(cos_toward)**2
            # (2) encourage consistent detour side (prefer sign(side_pref * sin_tangent) > 0)
            detour_pen   = self.w_tangent * near_gate * relu(-side_pref * sin_tangent)**2
            # (3) reward clearance
            clear_reward = - self.w_clear * near_gate * h
            # (4) once lateral, reward forward speed to pass the obstacle
            side_clear   = ca.fabs(sin_tangent)
            forward_push = - 50.0 * near_gate * side_clear * vj
            # (5) tiny turning nudge in chosen direction
            turn_bias    = 10.0 * near_gate * (uj[1] - 0.4 * side_pref)**2

            J += approach_pen + detour_pen + clear_reward + forward_push + turn_bias
            J -= 0.1 * vj  # small general push forward

            # --- Adaptive velocity desire ---
            align_err = e_lat**2 + 0.5 * (e_th**2)
            alpha_align = ca.exp(-4.0 * align_err)
            safe_gate = 1.0 - ca.exp(-1.0 * relu(h))
            speed_push = 0.3 + 0.7 * alpha_align * safe_gate
            v_des = speed_push * v_ref
            J += self.Qv * ((uj[0] - v_des)**2)

            # Δu cost
            if j == 0:
                du = uj - u_last
            else:
                uj_prev = U[(j - 1) * nu:(j) * nu]
                du = uj - uj_prev
            J += self.Sdu * ca.mtimes([du.T, du])

        # Slack penalty
        J += self.rho_slack * (delta**2)

        # --- Constraints: input & wheels ---
        g = []
        for j in range(N):
            v, w = U[j * nu], U[j * nu + 1]
            wl = (2 * v - w * self.b_track) / (2 * self.r_wheel)
            wr = (2 * v + w * self.b_track) / (2 * self.r_wheel)
            g += [v, self.v_max - v,
                  w + self.w_max, self.w_max - w,
                  self.omega_max - wl, self.omega_max + wl,
                  self.omega_max - wr, self.omega_max + wr]

        # --- Soft CBF with tangential relief (prevents stall when sliding around) ---
        for j in range(N):
            xj = Xpred[j]
            vj = U[j*nu]
            pj = xj[0:2]
            thj = xj[2]
            e_th_vec = ca.vertcat(ca.cos(thj), ca.sin(thj))

            r   = pj - obs
            r2  = ca.mtimes([r.T, r])
            h   = r2 - dmin**2
            dh  = 2.0 * ca.mtimes([r.T, e_th_vec]) * vj

            dist = ca.sqrt(r2 + 1e-9)
            r_hat = r / (dist + 1e-9)
            sin_tangent = r_hat[0]*e_th_vec[1] - r_hat[1]*e_th_vec[0]

            tangential_relief = 0.15 * ca.fabs(sin_tangent) * ca.fabs(vj)  # loosen when sliding
            near_inside = ca.exp(-3.0 * relu(h))   # ~1 when h<=0
            alpha_eff = 0.7 * self.alpha_cbf * (1.0 + 2.0 * near_inside)

            # dh + alpha*h >= -delta  (with small offset and relief)
            g.append(cbf_on * (dh + alpha_eff*h + 0.03 + tangential_relief) + delta)

        g.append(delta)  # slack >= 0

        Z = ca.vertcat(U, delta)
        nlp = {
            "x": Z,
            "f": J,
            "g": ca.vertcat(*g),
            "p": ca.vertcat(x0, ref, u_last, v_ref, obs, cbf_on, side_pref),
        }

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 40,
            "ipopt.tol": 1e-3,
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        self.U_len = nu * N
        self.g_dim = len(g)

    # ---------------- Control loop ----------------
    def control_step(self):
        if self.x_current is None or self.ref is None:
            return

        if self.x_current[0] >= self.x_stop:
            self.pub_L.publish(Float32(data=0.0))
            self.pub_R.publish(Float32(data=0.0))
            self.get_logger().info("Reached stop condition (x >= 3.0 m)")
            return

        # Warm start
        if self.U_prev is not None and self.U_prev.shape[0] == self.U_len:
            U0 = np.hstack([self.U_prev[2:], self.U_prev[-2:]])
        else:
            U0 = np.tile(self.u_last, self.N)
        Z0 = np.concatenate([U0, [0.0]])

        # Parameters
        if self.obstacle_xy is None:
            obs_xy = np.array([0.0, 0.0])
            cbf_on_val = np.array([0.0])
            side_pref_val = np.array([1.0])
        else:
            obs_xy = self.obstacle_xy
            cbf_on_val = np.array([1.0])
            # choose side based on ref y relative to obstacle
            side_pref_val = np.array([1.0 if (self.ref[1] - obs_xy[1]) >= 0.0 else -1.0])

        params = np.concatenate([
            self.x_current, self.ref, self.u_last,
            [self.v_ref], obs_xy, cbf_on_val, side_pref_val,
        ])

        try:
            sol = self.solver(
                x0=Z0,
                p=params,
                lbg=np.zeros(self.g_dim),
                ubg=np.full(self.g_dim, np.inf),
            )
            Zstar = np.array(sol["x"]).flatten()
            Ustar = Zstar[:self.U_len]
            u0 = Ustar[:2]
            self.U_prev = Ustar.copy()
        except Exception as e:
            self.get_logger().warn(f"Solver failed: {e}")
            u0 = np.array([0.05, 0.0])
            self.U_prev = None

        v_cmd = float(np.clip(u0[0], 0.0, self.v_max))
        w_cmd = float(np.clip(u0[1], -self.w_max, self.w_max))

        wl = (2 * v_cmd - w_cmd * self.b_track) / (2 * self.r_wheel)
        wr = (2 * v_cmd + w_cmd * self.b_track) / (2 * self.r_wheel)
        wl = np.clip(wl, -self.omega_max, self.omega_max)
        wr = np.clip(wr, -self.omega_max, self.omega_max)

        self.get_logger().info(
            f"[DEBUG] v={v_cmd:.2f}, w={w_cmd:.2f}, WL={wl:.1f}, WR={wr:.1f}"
        )

        self.pub_L.publish(Float32(data=float(wl)))
        self.pub_R.publish(Float32(data=float(wr)))
        self.u_last = np.array([v_cmd, w_cmd])

        t = time.time() - self.start_wall_time
        self.get_logger().info(
            f"[t={t:4.1f}s] v={v_cmd:.3f}, w={w_cmd:.3f} | "
            f"WL={wl:.2f}, WR={wr:.2f} | x={self.x_current[0]:.2f}, y={self.x_current[1]:.2f}"
        )

        # Publish CBF monitor
        if self.obstacle_xy is not None:
            r = self.x_current[0:2] - self.obstacle_xy
            h = float(np.dot(r, r) - (self.R_robot + self.R_obs + self.r_buf) ** 2)
            msg = Float32MultiArray()
            msg.data = [h, v_cmd, w_cmd]
            self.pub_cbf.publish(msg)

        if self.obstacle_xy is not None and self.x_current is not None:
            r = self.x_current[0:2] - self.obstacle_xy
            dist = np.linalg.norm(r)
            h = dist**2 - (self.R_robot + self.R_obs + self.r_buf)**2

            rel_dir = r[0]*np.cos(self.x_current[2]) + r[1]*np.sin(self.x_current[2])
            rel_toward = max(-rel_dir, 0.0)  # only toward
            dh = -2.0 * rel_toward * self.u_last[0]  # use last v_cmd

            cbf_val = dh + self.alpha_cbf * h
            cbf_active = cbf_val < 0.0

            self.get_logger().info(
                f"[CBF monitor] dist={dist:.3f}, h={h:.3f}, dh={dh:.3f}, cbf_val={cbf_val:.3f}, active={cbf_active}"
            )

        if self.ref is not None and self.x_current is not None:
            e_long = np.cos(self.ref[2])*(self.x_current[0]-self.ref[0]) + np.sin(self.ref[2])*(self.x_current[1]-self.ref[1])
            e_lat  = -np.sin(self.ref[2])*(self.x_current[0]-self.ref[0]) + np.cos(self.ref[2])*(self.x_current[1]-self.ref[1])
            self.get_logger().info(f"[ERROR] e_long={e_long:.3f}, e_lat={e_lat:.3f}, yaw_diff={(self.x_current[2]-self.ref[2]):.3f}")

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


if __name__ == "__main__":
    main()
