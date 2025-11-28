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

        self.avoid_side: Optional[int] = None   # +1 = CCW (left), -1 = CW (right)
        self.near_thresh: float = 0.25          # distance [m] to “decide side” once

        # --- Robot parameters ---
        self.r_wheel: float = 0.05
        self.b_track: float = 0.085
        self.v_max: float   = 0.25
        self.w_max: float   = 3.0
        self.omega_max: float = 4.0

        # --- MPC parameters ---
        self.Ts: float = 0.01
        self.N: int    = 50

        # Weights
        self.Qp  = np.diag([5.5 , 5.0])
        self.Qth = 2.0
        self.Qv  = 2.0
        self.R   = np.diag([0.5, 0.5])
        self.Sdu = 3.0

        # --- CBF parameters ---
        self.R_robot: float = 0.20
        self.R_obs:   float = 0.20
        self.r_buf:   float = 0.10
        self.alpha_cbf: float = 200.0
        self.rho_slack: float = 1e5

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

        # --- Warm start ---
        self.U_prev: Optional[np.ndarray] = None

        # Build solver
        self._build_solver()

        # ROS I/O
        self.sub_pose = self.create_subscription(
            PoseStamped, '/vicon/puzzlebot2/puzzlebot2/pose', self.pose_cb, 20)
        self.sub_vs = self.create_subscription(
            PoseStamped, '/vs/reference/puzzlebot2', self.vs_cb, 20)
        self.sub_vref = self.create_subscription(
            Float32, '/vs/reference_speed', self.vref_cb, 10)
        self.sub_obs_pose = self.create_subscription(
            PoseStamped, '/vicon/Obstacle/Obstacle/pose', self.obstacle_cb, 10)

        self.pub_L = self.create_publisher(Float32, '/puzzlebot2/VelocitySetL', 10)
        self.pub_R = self.create_publisher(Float32, '/puzzlebot2/VelocitySetR', 10)
        self.pub_cbf = self.create_publisher(Float32MultiArray, '/cbf_monitor', 10)

        self.timer = self.create_timer(self.Ts, self.control_step)
        self.get_logger().info("✅ DMPC node (CasADi + Ipopt, nonlinear) ready (100 Hz).")

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

        U = ca.SX.sym('U', nu * N)

        x0 = ca.SX.sym('x0', nx)
        ref = ca.SX.sym('ref', nx)
        u_last = ca.SX.sym('u_last', nu)
        v_ref = ca.SX.sym('v_ref')
        obs = ca.SX.sym('obs', 2)
        cbf_on = ca.SX.sym('cbf_on')
        side_sel = ca.SX.sym('side_sel')  # +1 or -1

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

            # --- Errors ---
            ref_jx = ref[0] + v_ref * Ts * j * ca.cos(ref[2])
            ref_jy = ref[1] + v_ref * Ts * j * ca.sin(ref[2])
            ref_jth = ref[2]

            dx = xk[0] - ref_jx
            dy = xk[1] - ref_jy
            e_th = ca.fmod(xk[2] - ref_jth + ca.pi, 2*ca.pi) - ca.pi
            e_lat = -ca.sin(ref[2]) * dx + ca.cos(ref[2]) * dy

            # --- Cost terms ---
            #e_pos = xk[0:2] - ref[0:2]
            #J += ca.mtimes([e_pos.T, self.Qp, e_pos]) + self.Qth * (e_th**2)
            e_long =  ca.cos(ref[2]) * dx + ca.sin(ref[2]) * dy
            e_lat  = -ca.sin(ref[2]) * dx + ca.cos(ref[2]) * dy
            J += self.Qp[0,0]*(e_long**2) + self.Qp[1,1]*(e_lat**2) + self.Qth*(e_th**2)
            J += ca.mtimes([uj.T, self.R, uj])

            # Adaptive Qv gating
            align_err = e_lat**2 + 0.5*(e_th**2)
            alpha_gate = ca.exp(-4.0 * align_err)   # high when aligned, low when misaligned
            v_des = alpha_gate * v_ref
            J += self.Qv * ((uj[0] - v_des)**2)

            # Δu penalty (slightly softer first step)
            if j == 0:
                du = uj - u_last
                #J += (0.5*self.Sdu) * ca.mtimes([du.T, du])
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

        # --- CBF + hard state constraint (no penetration) ---
        dmin = (self.R_robot + self.R_obs + self.r_buf)
        margin = 0.18                     # 8 cm guard band (tune 0.05–0.12)
        d_act = dmin + margin

        for j in range(N):
            xj = Xpred[j]
            vj = U[j*nu]
            pj = xj[0:2]
            thj = xj[2]

            e_th_vec = ca.vertcat(ca.cos(thj), ca.sin(thj))
            r = pj - obs

            # Use activation radius (guard band)
            h = ca.mtimes([r.T, r]) - d_act**2          # >= 0 outside guard
            dh = 2.0 * ca.mtimes([r.T, e_th_vec]) * vj  # directional time derivative

            # -------- Near-obstacle gates & tangent alignment --------
            side = side_sel
            # Near gate: 1 when close to guard band, 0 when far
            # smoothstep via sigmoid on h (h=0 at guard band)
            k_near = 15.0
            near_gate = 1.0 / (1.0 + ca.exp(k_near * h))   # ~1 near, ~0 far

            # Are we moving toward the obstacle? rel_dir > 0 means heading points at obs
            rel_dir = r[0]*ca.cos(thj) + r[1]*ca.sin(thj)
            rel_toward = 0.5*(rel_dir + ca.fabs(rel_dir))  # relu(rel_dir)

            # Slowdown when near & pointing toward obstacle (turn-first behavior)
            turn_first_gain = 0.85
            v_gate = 1.0 - turn_first_gain * near_gate * ca.fmin(1.0, rel_toward / (1e-3 + ca.norm_2(r)))

            # Desired v near obstacle = v_ref * existing alpha_gate (alignment) * v_gate
            # (alpha_gate already defined earlier in your cost; reuse it here)
            v_des_near = v_gate * v_ref * alpha_gate
            J += self.Qv * ((vj - v_des_near)**2)

            # Tangent heading (choose side via parameter 'side' passed as +1/-1; see below)
            #side = cbf_on * 2.0 - 1.0    # placeholder; will overwrite via 'p' later
            # Unit tangent along circle about obstacle: t = side * R90 * (r/||r||)
            r_norm = ca.fmax(1e-6, ca.norm_2(r))
            t_hat_x =  side * (-r[1] / r_norm)
            t_hat_y =  side * (  r[0] / r_norm)
            th_tan  = ca.atan2(t_hat_y, t_hat_x)
            e_tan   = ca.fmod(thj - th_tan + ca.pi, 2*ca.pi) - ca.pi

            # Add tangent alignment cost only when near
            Q_tan = 50.0
            J += Q_tan * near_gate * (e_tan**2)

            # (A) Hard state constraint: NEVER enter unsafe set (with guard band)
            #     h >= 0  -> add as inequality: g >= 0
            g.append(cbf_on * h)

            # (B) Hard ZCBF: forward invariance wrt the guard band
            #     dh + alpha*h >= 0
            g.append(cbf_on * (dh + self.alpha_cbf * h))
        # ✅ --- Discrete-time CBF tightening (prevents inter-sample tunneling) ---
        gamma = 0.8   # 0.4–0.8 works well

        h_seq = []
        p0 = x0[0:2]
        r0 = p0 - obs
        h0 = ca.mtimes([r0.T, r0]) - d_act**2
        h_seq.append(h0)

        for j in range(N):
            pj = Xpred[j][0:2]
            rj = pj - obs
            hj = ca.mtimes([rj.T, rj]) - d_act**2
            h_seq.append(hj)

        for j in range(N):
            g.append(cbf_on * (h_seq[j+1] - (1.0 - gamma)*h_seq[j]))
        # --- Stage-0 (current state) safety constraints ---
        p0   = x0[0:2]
        th0  = x0[2]
        r0   = p0 - obs
        h0   = ca.mtimes([r0.T, r0]) - d_act**2
        e_th0 = ca.vertcat(ca.cos(th0), ca.sin(th0))
        v0    = U[0]  # first control's linear speed

        dh0 = 2.0 * ca.mtimes([r0.T, e_th0]) * v0

        # Hard guard-band at current state
        g.append(cbf_on * h0)                       # h(x0) >= 0

        # ZCBF at the very first step (prevents inter-sample tunneling)
        g.append(cbf_on * (dh0 + self.alpha_cbf * h0))

        # --- Build NLP ---
        Z = ca.vertcat(U)
        nlp = {
            'x': Z,
            'f': J,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(x0, ref, u_last, v_ref, obs, cbf_on, side_sel)
        }

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 40,
            'ipopt.tol': 1e-3
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.Z_dim = Z.size1()
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
        Z0 = U0.copy()

        if self.obstacle_xy is None:
            obs_xy = np.array([0.0, 0.0])
            cbf_on_val = np.array([0.0])
        else:
            obs_xy = self.obstacle_xy
            cbf_on_val = np.array([1.0])

        # Decide avoid side once when near
        if self.obstacle_xy is not None and self.x_current is not None:
            r_vec = self.x_current[0:2] - self.obstacle_xy
            dist  = float(np.linalg.norm(r_vec))
            if self.avoid_side is None and dist < self.near_thresh:
                # Side selection by cross of (ref_dir) with (obs - robot):
                # positive => obstacle on left => go CW around it (side = -1)
                # negative => obstacle on right => go CCW around it (side = +1)
                th_ref = self.ref[2] if self.ref is not None else self.x_current[2]
                ref_dir = np.array([np.cos(th_ref), np.sin(th_ref)])
                cross_z = ref_dir[0]*r_vec[1] - ref_dir[1]*r_vec[0]
                self.avoid_side = -1 if cross_z > 0 else +1
        elif self.obstacle_xy is None:
            self.avoid_side = None

        side_val = float(self.avoid_side) if (self.avoid_side is not None) else +1.0
        params = np.concatenate([
            self.x_current, self.ref, self.u_last, [self.v_ref], obs_xy, cbf_on_val, [side_val]
        ])

        try:
            sol = self.solver(
                x0=Z0,
                p=params,
                lbg=np.zeros(self.g_dim),
                ubg=np.full(self.g_dim, np.inf)
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
        wl = np.clip(wl, -self.omega_max, self.omega_max)
        wr = np.clip(wr, -self.omega_max, self.omega_max)

        # Recompute consistent v,w from clipped wheels
        v_cmd = float(self.r_wheel * (wl + wr) / 2.0)
        w_cmd = float(self.r_wheel * (wr - wl) / self.b_track)

        self.pub_L.publish(Float32(data=float(wl)))
        self.pub_R.publish(Float32(data=float(wr)))
        self.u_last = np.array([v_cmd, w_cmd], dtype=float)

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

