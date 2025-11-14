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
        super().__init__('dmpc_cbf_node')

        # --- Robot parameters ---
        self.r_wheel: float = 0.05
        self.b_track: float = 0.085
        self.v_max: float   = 0.25
        self.w_max: float   = 6.0
        self.omega_max: float = 10.0

        # --- MPC parameters ---
        self.Ts: float = 0.01
        self.N: int    = 120

        # Weights (kept close to yours)
        self.Qp  = np.diag([35.0 , 40.0])  # [long, lat]
        self.Qth = 20.0
        self.Qv  = 2.0
        self.R   = np.diag([15.2, 18.2])
        self.Sdu = 20.0

        # --- Safety (CBF, discrete-time decay) ---
        self.R_robot: float = 0.20
        self.R_obs:   float = 0.20
        self.margin:  float = 0.10         # extra buffer to be conservative
        self.gamma_cbf: float = 0.2       # decay factor in [0,1]; smaller = more conservative
        self.rho_cbf: float = 2e2        

        # --- Lyapunov "backup" gate ---
        # V = q_long*e_long^2 + q_lat*e_lat^2 + q_th*e_th^2
        self.V_q_long: float = 1.0
        self.V_q_lat:  float = 3.0
        self.V_q_th:   float = 2.0
        # Backup controller gains (smooth, conservative)
        self.k_lat: float = 2.0
        self.k_th:  float = 2.0
        self.k_vdrop: float = 0.35   # reduces v when misaligned
        self.rho_V: float = 12   

        # Stop condition
        self.x_stop: float = 3.0

        # State & reference
        self.x_current: Optional[np.ndarray] = None
        self.ref: Optional[np.ndarray]       = None
        self.u_last = np.array([0.05, 0.0])
        self.start_wall_time = time.time()
        self.v_ref: float=0.25

        # Obstacle state
        self.obstacle_xy: Optional[np.ndarray] = None

        # Warm start
        self.U_prev: Optional[np.ndarray] = None

        # Build solver
        self._build_solver()

        # ROS I/O
        self.declare_parameter('robot_name', 'puzzlebot2')
        robot_name = self.get_parameter('robot_name').get_parameter_value().string_value

        self.sub_pose = self.create_subscription(
            PoseStamped, f'/vicon/{robot_name}/{robot_name}/pose', self.pose_cb, 20)
        self.sub_vs = self.create_subscription(
            PoseStamped, f'/vs/reference/{robot_name}', self.vs_cb, 20)
        self.sub_vref = self.create_subscription(
            Float32, '/vs/reference_speed', self.vref_cb, 10)
        self.sub_obs_pose = self.create_subscription(
            PoseStamped, '/vicon/Obstacle/Obstacle/pose', self.obstacle_cb, 10)

        self.pub_L = self.create_publisher(Float32, f'/{robot_name}/VelocitySetL', 10)
        self.pub_R = self.create_publisher(Float32, f'/{robot_name}/VelocitySetR', 10)
        self.pub_cbf = self.create_publisher(Float32MultiArray, '/cbf_monitor', 10)

        self.timer = self.create_timer(self.Ts, self.control_step)
        self.get_logger().info(f"✅ DMPC node ready for {robot_name} with discrete CBF + Lyapunov gate.")

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

    def obstacle_cb(self, msg: PoseStamped):
        xo = msg.pose.position.x
        yo = msg.pose.position.y
        self.obstacle_xy = np.array([xo, yo], dtype=float)

    # ---------------- CasADi solver ----------------
    def _build_solver(self):
        nx, nu, N, Ts = 3, 2, self.N, self.Ts

        # Decision vars: U (nu*N), per-step slacks for CBF (N) and Lyapunov gate (N)
        U = ca.SX.sym('U', nu * N)
        s_cbf = ca.SX.sym('s_cbf', N)  # >= 0
        s_V   = ca.SX.sym('s_V',   N)  # >= 0

        # Parameters
        x0   = ca.SX.sym('x0', nx)     # current state
        ref  = ca.SX.sym('ref', nx)    # current reference pose (position, heading)
        u_last = ca.SX.sym('u_last', nu)
        v_ref = ca.SX.sym('v_ref')
        obs = ca.SX.sym('obs', 2)      # obstacle position
        gamma = ca.SX.sym('gamma')     # CBF decay (0..1)
        d_act = ca.SX.sym('d_act')     # active safety radius

        def f(x, u):
            th = x[2]
            v, w = u[0], u[1]
            return ca.vertcat(
                x[0] + Ts * v * ca.cos(th),
                x[1] + Ts * v * ca.sin(th),
                x[2] + Ts * w
            )

        # Build prediction including x[0] = x0
        X = [x0]
        for j in range(N):
            uj = U[j*nu:(j+1)*nu]
            X.append(f(X[-1], uj))   # X[j+1]

        # Helper: errors (long/lat/heading) wrt a straight-ahead moving ref
        def err_components(x, ref_pose, j_idx):
            # advance the reference along its heading at speed v_ref
            ref_jx = ref_pose[0] + v_ref * Ts * j_idx * ca.cos(ref_pose[2])
            ref_jy = ref_pose[1] + v_ref * Ts * j_idx * ca.sin(ref_pose[2])
            ref_jth = ref_pose[2]
            dx = x[0] - ref_jx
            dy = x[1] - ref_jy
            e_th = ca.fmod(x[2] - ref_jth + ca.pi, 2*ca.pi) - ca.pi
            e_long =  ca.cos(ref_pose[2]) * dx + ca.sin(ref_pose[2]) * dy
            e_lat  = -ca.sin(ref_pose[2]) * dx + ca.cos(ref_pose[2]) * dy
            return e_long, e_lat, e_th

        # Backup controller (smooth, always stabilizing around the path)
        def backup_u(x, ref_pose, j_idx):
            e_long, e_lat, e_th = err_components(x, ref_pose, j_idx)
            # speed drops with misalignment and lateral error
            v_des = v_ref * ca.exp(-self.k_vdrop*(e_lat**2 + 0.5*e_th**2))
            v_b = ca.fmax(0.0, ca.fmin(self.v_max, v_des))
            # heading & lateral correction
            w_b = -self.k_th*e_th - self.k_lat*e_lat
            w_b = ca.fmax(-self.w_max, ca.fmin(self.w_max, w_b))
            return ca.vertcat(v_b, w_b)

        # Lyapunov function V
        def V_of(x, ref_pose, j_idx):
            e_long, e_lat, e_th = err_components(x, ref_pose, j_idx)
            return (self.V_q_long*(e_long**2) +
                    self.V_q_lat *(e_lat**2)  +
                    self.V_q_th  *(e_th**2))

        # Cost
        J = 0
        for j in range(1, N+1):  # states X[1]..X[N] correspond to j=1..N
            xj = X[j]
            uj = U[(j-1)*nu:j*nu]
            e_long, e_lat, e_th = err_components(xj, ref, j)

            # Tracking + effort + Δu
            J += self.Qp[0,0]*(e_long**2) + self.Qp[1,1]*(e_lat**2) + self.Qth*(e_th**2)
            J += ca.mtimes([uj.T, self.R, uj])

            # Alignment-based speed shaping (your idea, kept)
            align_err = e_lat**2 + 0.5*(e_th**2)
            alpha_gate = ca.exp(-6.0*align_err)
            v_des = alpha_gate * v_ref
            J += self.Qv * ((uj[0] - v_des)**2)

            # Δu penalty
            if j == 1:
                du = uj - u_last
            else:
                uj_prev = U[(j-2)*nu:(j-1)*nu]
                du = uj - uj_prev
            J += self.Sdu * ca.mtimes([du.T, du])

        # Slack penalties
        J += self.rho_cbf * ca.dot(s_cbf, s_cbf)
        J += self.rho_V   * ca.dot(s_V,   s_V)

        # Constraints
        g = []
        lbg = []
        ubg = []

        # Input and wheel limits each step
        for j in range(N):
            v, w = U[j*nu], U[j*nu+1]
            wl = (2*v - w*self.b_track) / (2*self.r_wheel)
            wr = (2*v + w*self.b_track) / (2*self.r_wheel)

            # v in [0, v_max]; w in [-w_max, w_max]
            g += [v, self.v_max - v, w + self.w_max, self.w_max - w]
            lbg += [0.0, 0.0, 0.0, 0.0]
            ubg += [ca.inf, ca.inf, ca.inf, ca.inf]

            # wheel rates in [-omega_max, omega_max]
            g += [ self.omega_max - wl, self.omega_max + wl,
                   self.omega_max - wr, self.omega_max + wr ]
            lbg += [0.0, 0.0, 0.0, 0.0]
            ubg += [ca.inf, ca.inf, ca.inf, ca.inf]

        # --- Discrete-time CBF with decay, per-step slack ---
        # h(x) = ||p - obs||^2 - d_act^2
        # Enforce: h_{j+1} - (1-gamma) h_j + s_cbf[j] >= 0  for j=0..N-1
        for j in range(N):
            pj   = X[j][0:2]
            pjp1 = X[j+1][0:2]
            rj   = pj   - obs
            rjp1 = pjp1 - obs
            hj   = ca.mtimes([rj.T, rj])     - d_act**2
            hjp1 = ca.mtimes([rjp1.T, rjp1]) - d_act**2
            g.append(hjp1 - (1.0 - gamma)*hj + s_cbf[j])
            lbg.append(0.0)
            ubg.append(ca.inf)

            # enforce s_cbf[j] >= 0
            g.append(s_cbf[j])
            lbg.append(0.0)
            ubg.append(ca.inf)

        # --- Lyapunov "do-no-worse-than-backup", per-step slack ---
        # V(X[j+1]) - V(X[j]) <= V(Xb[j+1]) - V(X[j]) + s_V[j]
        for j in range(N):
            xj = X[j]
            xjp1 = X[j+1]
            ubj = backup_u(xj, ref, j)              # backup input at step j
            xb_next = f(xj, ubj)                     # backup's next state
            V_now   = V_of(xj,     ref, j)
            V_cand  = V_of(xjp1,   ref, j+1)
            V_back  = V_of(xb_next, ref, j+1)
            g.append( (V_cand - V_now) - (V_back - V_now) - s_V[j] )
            lbg.append(-ca.inf)
            ubg.append(0.0)

            # s_V[j] >= 0
            g.append(s_V[j])
            lbg.append(0.0)
            ubg.append(ca.inf)

        # --- Build NLP ---
        Z = ca.vertcat(U, s_cbf, s_V)
        nlp = {
            'x': Z,
            'f': J,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(x0, ref, u_last, v_ref, obs, gamma, d_act)
        }

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 200,
            'ipopt.tol': 1e-4
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.Z_dim = Z.size1()
        self.U_len = (nu * N)
        self.g_dim = len(g)
        self.lbg = np.zeros(self.g_dim)  # will overwrite with proper arrays
        self.ubg = np.zeros(self.g_dim)
        # cache
        self._lbg = np.array(lbg, dtype=float)
        self._ubg = np.array(ubg, dtype=float)

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

        # initial slacks to small values
        s0_cbf = np.full(self.N, 1e-6)
        s0_V   = np.full(self.N, 1e-6)
        Z0 = np.concatenate([U0, s0_cbf, s0_V])

        # Obstacle parameters
        if self.obstacle_xy is None:
            # Place obstacle far away; constraints become inactive naturally
            obs_xy = np.array([1e6, 1e6], dtype=float)
        else:
            obs_xy = self.obstacle_xy.astype(float)

        d_act_val = (self.R_robot + self.R_obs + self.margin)
        gamma_val = float(np.clip(self.gamma_cbf, 0.0, 1.0))

        params = np.concatenate([
            self.x_current, self.ref, self.u_last, [self.v_ref],
            obs_xy, [gamma_val], [d_act_val]
        ])

        try:
            sol = self.solver(
                x0=Z0,
                p=params,
                lbg=self._lbg,
                ubg=self._ubg
            )
            Zstar = np.array(sol['x']).flatten()
            Ustar = Zstar[:self.U_len]
            u0 = Ustar[:2]
            self.U_prev = Ustar.copy()
        except Exception as e:
            self.get_logger().warn(f"Solver failed: {e}")
            u0 = np.array([0.05, 0.0])
            self.U_prev = None

        v_cmd = float(np.clip(u0[0], 0.0, self.v_max))
        w_cmd = float(np.clip(u0[1], -self.w_max, self.w_max))

        wl = (2*v_cmd - w_cmd*self.b_track) / (2*self.r_wheel)
        wr = (2*v_cmd + w_cmd*self.b_track) / (2*self.r_wheel)
        wl = float(np.clip(wl, -self.omega_max, self.omega_max))
        wr = float(np.clip(wr, -self.omega_max, self.omega_max))

        self.pub_L.publish(Float32(data=wl))
        self.pub_R.publish(Float32(data=wr))
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
