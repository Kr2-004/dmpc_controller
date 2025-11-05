#!/usr/bin/env python3
import time
import math
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
        self.v_max = 0.25  # 0.8
        self.w_max = 2.0   # 6.0
        self.omega_max = 5.0  # 10.0

        # --- MPC parameters ---
        self.Ts = 0.01
        self.N = 50

        # Weights
        self.Qp = np.diag([1.0, 4.0])   # [longitudinal, lateral]
        self.Qth = 2.0
        self.Qv = 2.0
        self.R = np.diag([0.5, 0.5])
        self.Sdu = 5.0

        # --- CBF parameters ---
        self.R_robot = 0.20
        self.R_obs   = 0.20
        self.r_buf   = 0.10
        self.alpha_cbf = 1.0     # ↓ softened (was 1.5)
        self.rho_slack = 100.0

        # Directional + hysteresis thresholds
        # dmin = Rr + Ro + r_buf  → use a bit beyond that for hysteresis
        self.hyst_d_on  = (self.R_robot + self.R_obs + self.r_buf + 0.06)  # engage near boundary
        self.hyst_d_off = (self.R_robot + self.R_obs + self.r_buf + 0.16)  # release further away
        # Convert to h = d^2 - dmin^2 at runtime once we know dmin

        # Detour shaping weights
        self.w_approach =50.0
        self.w_tangent  = 40.0
        self.w_clear    = 70.0
        self.safe_margin = 0.12
        self.near_k  = 10.0

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

        # CBF hysteresis state
        self.cbf_active = False

        # Debug throttle
        self.step_count = 0
        self.debug_every = 50  # print full block every N steps

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
        self.get_logger().info("✅ DMPC node with directional CBF + hysteresis + tangential detour ready (100 Hz)")

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
            uj = U[j * 2:(j + 1) * 2]
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
            cos_toward = ca.fmax(cos_toward, 0.0)          # zero when obstacle behind
            sin_tangent = r_hat[0]*e_th_vec[1] - r_hat[1]*e_th_vec[0]

            # (1) penalize approaching (only when facing)
            approach_pen = self.w_approach * near_gate * relu(cos_toward)**2
            # (2) encourage consistent detour side (prefer sign(side_pref * sin_tangent) > 0)
            #detour_gate = ca.fmax(cos_toward, 0.0)
            detour_pen   = self.w_tangent * near_gate * relu(-side_pref * sin_tangent)**2
            # (3) reward clearance
            clear_reward = - self.w_clear * near_gate * h
            # (4) once lateral, reward forward speed to pass the obstacle
            side_clear   = ca.fabs(sin_tangent)
            forward_push = - 120.0 * near_gate * side_clear * vj
            # (5) tiny turning nudge in chosen direction
            turn_bias    = 1.0 * near_gate * (uj[1] - 0.4 * side_pref)**2

            J += approach_pen + detour_pen + clear_reward + forward_push + turn_bias

            # --- Adaptive velocity desire (direction-aware safety) ---
            align_err = e_lat**2 + 0.5 * (e_th**2)
            alpha_align = ca.exp(-4.0 * align_err)
            safe_geom = 1.0 - ca.exp(-1.0 * relu(h))
            # If NOT facing obstacle, allow speed even when h small:
            dir_gate = 0.5 * (1.0 + cos_toward)              # ∈ [0, +], >0 when facing
            safe_dir = 1.0 - ca.fmin(dir_gate, 1.0)  # ~1 when not facing, ~0 when facing
            safe_mix = ca.fmax(safe_geom, safe_dir)
            speed_push = 0.3 + 0.7 * alpha_align * safe_mix
            v_des = speed_push * v_ref
            J += self.Qv * ((uj[0] - v_des)**2)

            # Δu cost
            if j == 0:
                du = uj - u_last
            else:
                uj_prev = U[(j - 1) * 2:(j) * 2]
                du = uj - uj_prev
            J += self.Sdu * ca.mtimes([du.T, du])

        # Slack penalty
        J += self.rho_slack * (delta**2)

        # --- Constraints: input & wheels ---
        g = []
        for j in range(N):
            v, w = U[j * 2], U[j * 2 + 1]
            wl = (2 * v - w * self.b_track) / (2 * self.r_wheel)
            wr = (2 * v + w * self.b_track) / (2 * self.r_wheel)
            g += [v, self.v_max - v,
                  w + self.w_max, self.w_max - w,
                  self.omega_max - wl, self.omega_max + wl,
                  self.omega_max - wr, self.omega_max + wr]

        # --- Directional soft CBF with tangential relief + hysteresis gate ---
        for j in range(N):
            xj = Xpred[j]
            vj = U[j*2]
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
            cos_toward  = -ca.mtimes([e_th_vec.T, r_hat])  # + when facing obstacle
            dir_gate = ca.fmax(cos_toward, 0.0)                # only clamp when facing

            tangential_relief = 0.15 * ca.fabs(sin_tangent) * ca.fabs(vj)
            near_inside = ca.exp(-3.0 * relu(h))   # ~1 when h<=0
            alpha_eff = 0.7 * self.alpha_cbf * (1.0 + 2.0 * near_inside)

            # before: cbf_term = dir_gate * (dh + alpha_eff*h + 0.03) + tangential_relief
            cbf_term = 0.5*(dh + alpha_eff*h) + dir_gate*(dh + alpha_eff*h + 0.03) + tangential_relief

            # dh + alpha*h ≥ -delta  → append as inequality g_i = cbf_on*cbf_term + delta >= 0
            g.append(cbf_on * cbf_term + delta)

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
        self.U_len = 2 * N
        self.g_dim = len(g)

    # ---------- Debug helpers ----------
    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def _rollout_debug(self, x0, Ustar, ref, v_ref, obs_xy, side_pref):
        Ts = self.Ts
        N = self.N
        dmin = self.R_robot + self.R_obs + self.r_buf

        x = x0.copy()
        pred = {
            "h": [], "dh": [], "cbf_margin": [], "dist": [],
            "near_gate": [], "cos_toward": [], "sin_tangent": [],
            "v": [], "w": [], "v_des0": None,
            "alpha_align0": None, "safe_gate0": None
        }

        for j in range(N):
            v = float(Ustar[2*j + 0])
            w = float(Ustar[2*j + 1])

            # Step dynamics
            x = np.array([
                x[0] + Ts * v * math.cos(x[2]),
                x[1] + Ts * v * math.sin(x[2]),
                self._wrap_angle(x[2] + Ts * w)
            ])

            # Reference rollout for gates
            ref_jx = ref[0] + v_ref * Ts * j * math.cos(ref[2])
            ref_jy = ref[1] + v_ref * Ts * j * math.sin(ref[2])
            ref_jth = ref[2]
            dx = x[0] - ref_jx
            dy = x[1] - ref_jy
            e_th = self._wrap_angle(x[2] - ref_jth)
            e_long = math.cos(ref[2]) * dx + math.sin(ref[2]) * dy
            e_lat  = -math.sin(ref[2]) * dx + math.cos(ref[2]) * dy

            h_tmp = (x[0]-obs_xy[0])**2 + (x[1]-obs_xy[1])**2 - dmin**2
            align_err = e_lat**2 + 0.5*(e_th**2)
            alpha_align = math.exp(-4.0 * align_err)
            safe_gate = 1.0 - math.exp(-1.0 * max(h_tmp, 0.0))
            if j == 0:
                speed_push = 0.3 + 0.7 * alpha_align * max(safe_gate, 0.0)
                pred["v_des0"] = speed_push * v_ref
                pred["alpha_align0"] = alpha_align
                pred["safe_gate0"] = safe_gate

            # Geometry wrt obstacle
            r = np.array([x[0]-obs_xy[0], x[1]-obs_xy[1]])
            r2 = float(np.dot(r, r))
            dist = math.sqrt(r2 + 1e-9)
            r_hat = r / (dist + 1e-9)
            e_th_vec = np.array([math.cos(x[2]), math.sin(x[2])])

            h = r2 - dmin**2
            dh = 2.0 * float(np.dot(r, e_th_vec)) * v

            cos_toward  = - float(np.dot(e_th_vec, r_hat))
            sin_tangent = r_hat[0]*e_th_vec[1] - r_hat[1]*e_th_vec[0]

            near = max(dist - (dmin + self.safe_margin), 0.0)
            near_gate = math.exp(-self.near_k * near)

            # directional cbf margin approximation (for debug only)
            near_inside = math.exp(-3.0 * max(h, 0.0)) if h <= 0 else math.exp(-3.0 * 0.0)
            alpha_eff = 0.7 * self.alpha_cbf * (1.0 + 2.0 * near_inside)
            dir_gate = max(cos_toward, 0.0)
            tangential_relief = 0.15 * abs(sin_tangent) * abs(v)
            cbf_margin = dir_gate * (dh + alpha_eff*h + 0.03) + tangential_relief

            pred["h"].append(h)
            pred["dh"].append(dh)
            pred["cbf_margin"].append(cbf_margin)
            pred["dist"].append(dist)
            pred["near_gate"].append(near_gate)
            pred["cos_toward"].append(cos_toward)
            pred["sin_tangent"].append(sin_tangent)
            pred["v"].append(v)
            pred["w"].append(w)

        h_arr = np.array(pred["h"])
        min_h_idx = int(np.argmin(h_arr))
        min_h = float(h_arr[min_h_idx])
        cbf_margin_min = float(np.min(pred["cbf_margin"]))
        first_near_idx = next((i for i, g in enumerate(pred["near_gate"]) if g > 0.2), -1)

        return {
            "min_h": min_h,
            "min_h_idx": min_h_idx,
            "cbf_margin_min": cbf_margin_min,
            "first_near_idx": first_near_idx,
            "sin_at_min": float(pred["sin_tangent"][min_h_idx]),
            "cos_toward_at_min": float(pred["cos_toward"][min_h_idx]),
            "near_gate_at_min": float(pred["near_gate"][min_h_idx]),
            "dist_at_min": float(pred["dist"][min_h_idx]),
            "v_at_min": float(pred["v"][min_h_idx]),
            "w_at_min": float(pred["w"][min_h_idx]),
            "v_des0": float(pred["v_des0"]) if pred["v_des0"] is not None else None,
            "alpha_align0": float(pred["alpha_align0"]) if pred["alpha_align0"] is not None else None,
            "safe_gate0": float(pred["safe_gate0"]) if pred["safe_gate0"] is not None else None,
            "side_pref": float(side_pref),
        }

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

        # Parameters & CBF hysteresis
        if self.obstacle_xy is None:
            obs_xy = np.array([0.0, 0.0])
            cbf_on_val = np.array([0.0])
            side_pref_val = np.array([1.0])
        else:
            obs_xy = self.obstacle_xy

            # Compute h_now to update hysteresis
            dmin = self.R_robot + self.R_obs + self.r_buf
            r_now = self.x_current[0:2] - obs_xy
            dist_now = float(np.linalg.norm(r_now))
            h_now = dist_now**2 - dmin**2
            h_on  = self.hyst_d_on**2  - dmin**2
            h_off = self.hyst_d_off**2 - dmin**2

            if not self.cbf_active and h_now <= h_on:
                self.cbf_active = True
                self.get_logger().info(f"🟡 CBF engaged (h={h_now:.3f} ≤ h_on={h_on:.3f}, dist={dist_now:.3f})")
            elif self.cbf_active and h_now >= h_off:
                self.cbf_active = False
                self.get_logger().info(f"🟢 CBF released (h={h_now:.3f} ≥ h_off={h_off:.3f}, dist={dist_now:.3f})")

            cbf_on_val = np.array([1.0 if self.cbf_active else 0.0])

            # choose side based on ref y relative to obstacle
            if not hasattr(self, "side_pref_lat"):
                self.side_pref_lat = 1.0  # default to left

            # Keep same side while CBF is active
            if self.obstacle_xy is None or not self.cbf_active:
                # update side when free / released
                self.side_pref_lat = 1.0 if (self.ref[1] - obs_xy[1]) >= 0.0 else -1.0

            side_pref_val = np.array([self.side_pref_lat])

        params = np.concatenate([
            self.x_current, self.ref, self.u_last,
            [self.v_ref], obs_xy, cbf_on_val, side_pref_val,
        ])

        # Solve
        solved = True
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
            solved = False
            self.get_logger().warn(f"Solver failed: {e}")
            u0 = np.array([0.05, 0.0])
            self.U_prev = None

        v_cmd_unclipped = float(u0[0])
        w_cmd_unclipped = float(u0[1])

        v_cmd = float(np.clip(v_cmd_unclipped, 0.0, self.v_max))
        w_cmd = float(np.clip(w_cmd_unclipped, -self.w_max, self.w_max))

        wl_unclip = (2 * v_cmd - w_cmd * self.b_track) / (2 * self.r_wheel)
        wr_unclip = (2 * v_cmd + w_cmd * self.b_track) / (2 * self.r_wheel)

        wl = float(np.clip(wl_unclip, -self.omega_max, self.omega_max))
        wr = float(np.clip(wr_unclip, -self.omega_max, self.omega_max))

        sat_v = (abs(v_cmd - v_cmd_unclipped) > 1e-6)
        sat_w = (abs(w_cmd - w_cmd_unclipped) > 1e-6)
        sat_wheels = (abs(wl - wl_unclip) > 1e-6) or (abs(wr - wr_unclip) > 1e-6)

        self.pub_L.publish(Float32(data=float(wl)))
        self.pub_R.publish(Float32(data=float(wr)))
        self.u_last = np.array([v_cmd, w_cmd])

        t = time.time() - self.start_wall_time

        # ---------- Prediction diagnostics ----------
        diag = None
        if (self.obstacle_xy is not None) and (self.U_prev is not None) and solved:
            diag = self._rollout_debug(
                x0=self.x_current.copy(),
                Ustar=self.U_prev.copy(),
                ref=self.ref.copy(),
                v_ref=self.v_ref,
                obs_xy=self.obstacle_xy.copy(),
                side_pref=float(side_pref_val[0])
            )

        # Compact DEBUG line each step
        self.get_logger().info(
            f"[DEBUG] t={t:5.2f}s v={v_cmd:.3f}({v_cmd_unclipped:.3f}) w={w_cmd:.3f}({w_cmd_unclipped:.3f}) "
            f"WL={wl:.2f}({wl_unclip:.2f}) WR={wr:.2f}({wr_unclip:.2f}) "
            f"SAT[v,w,wheels]=[{int(sat_v)},{int(sat_w)},{int(sat_wheels)}] "
            f"solved={solved} cbf_active={int(self.cbf_active)}"
        )

        # Tracking error snapshot
        e_long = np.cos(self.ref[2])*(self.x_current[0]-self.ref[0]) + np.sin(self.ref[2])*(self.x_current[1]-self.ref[1])
        e_lat  = -np.sin(self.ref[2])*(self.x_current[0]-self.ref[0]) + np.cos(self.ref[2])*(self.x_current[1]-self.ref[1])

        # CBF live monitor at current state
        if self.obstacle_xy is not None:
            dmin = self.R_robot + self.R_obs + self.r_buf
            r_now = self.x_current[0:2] - self.obstacle_xy
            dist_now = float(np.linalg.norm(r_now))
            h_now = float(np.dot(r_now, r_now) - dmin**2)
            rel_dir = r_now[0]*np.cos(self.x_current[2]) + r_now[1]*np.sin(self.x_current[2])
            rel_toward = max(-rel_dir, 0.0)  # only toward
            dh_now = -2.0 * rel_toward * v_cmd
            cbf_val_now = dh_now + self.alpha_cbf * h_now

            # Publish extended CBF monitor
            msg = Float32MultiArray()
            cbf_margin_min = diag["cbf_margin_min"] if diag else 0.0
            min_h = diag["min_h"] if diag else h_now
            min_h_idx = float(diag["min_h_idx"]) if diag else 0.0
            sin_at_min = diag["sin_at_min"] if diag else 0.0
            cos_toward_at_min = diag["cos_toward_at_min"] if diag else 0.0
            near_gate_at_min = diag["near_gate_at_min"] if diag else 0.0
            msg.data = [
                h_now, v_cmd, w_cmd,
                cbf_margin_min, min_h, min_h_idx,
                sin_at_min, cos_toward_at_min, near_gate_at_min
            ]
            self.pub_cbf.publish(msg)

            # One-line DIAG always
            if diag:
                self.get_logger().info(
                    f"[DIAG] dist_now={dist_now:.3f} h_now={h_now:.3f} cbf_now={cbf_val_now:.3f} | "
                    f"min_h={diag['min_h']:.3f}@{diag['min_h_idx']} cbf_margin_min={diag['cbf_margin_min']:.3f} "
                    f"near@idx={diag['first_near_idx']} side_pref={'+L' if diag['side_pref']>0 else '-R'} "
                    f"sin@min={diag['sin_at_min']:.3f} cosTow@min={diag['cos_toward_at_min']:.3f} "
                    f"v_des0={diag['v_des0']:.3f} α_align0={diag['alpha_align0']:.2f} safe_gate0={diag['safe_gate0']:.2f}"
                )
            else:
                self.get_logger().info(
                    f"[DIAG] dist_now={dist_now:.3f} h_now={h_now:.3f} cbf_now={cbf_val_now:.3f} (no horizon diag)"
                )
        # ----- Extended Diagnostic Snapshot -----
        if diag:
            self.get_logger().info(
                f"[EXT] t={t:5.2f}s | "
                f"CBF={'ON' if self.cbf_active else 'OFF'} "
                f"dist={dist_now:.3f} h={h_now:.3f} cbf_now={cbf_val_now:.3f} "
                f"min_h={diag['min_h']:.3f} cbf_margin_min={diag['cbf_margin_min']:.3f} "
                f"sin@min={diag['sin_at_min']:.3f} cosTow@min={diag['cos_toward_at_min']:.3f} "
                f"near_gate@min={diag['near_gate_at_min']:.3f} "
                f"v_des0={diag['v_des0']:.3f} v_cmd={v_cmd:.3f} "
                f"α_align0={diag['alpha_align0']:.2f} safe_gate0={diag['safe_gate0']:.2f} "
                f"side={'L' if diag['side_pref']>0 else 'R'}"
            )

        # --- Log when CBF engages or disengages (only once per event) ---
        if not hasattr(self, "last_cbf_state"):
            self.last_cbf_state = self.cbf_active
        if self.cbf_active != self.last_cbf_state:
            state_str = "ENGAGED" if self.cbf_active else "RELEASED"
            self.get_logger().info(
                f"[EVENT] 🔁 CBF {state_str} at t={t:5.2f}s | "
                f"dist={dist_now:.3f} h={h_now:.3f} cbf_now={cbf_val_now:.3f} "
                f"cosTow@min={diag['cos_toward_at_min']:.3f} sin@min={diag['sin_at_min']:.3f} "
                f"v_cmd={v_cmd:.3f} w_cmd={w_cmd:.3f}"
            )
            self.last_cbf_state = self.cbf_active

        # --- Log when velocity drops sharply near obstacle (potential stall) ---
        if self.cbf_active and v_cmd < 0.02:
            self.get_logger().warn(
                f"[STALL?] Near obstacle dist={dist_now:.3f} h={h_now:.3f} "
                f"v_cmd={v_cmd:.3f} w_cmd={w_cmd:.3f} "
                f"cosTow@min={diag['cos_toward_at_min']:.3f} sin@min={diag['sin_at_min']:.3f}"
            )
        # Verbose block every debug_every steps
        self.step_count += 1
        if self.step_count % self.debug_every == 0:
            if self.obstacle_xy is None:
                self.get_logger().info("[VERBOSE] No obstacle: CBF OFF, pure tracking/damping.")
            else:
                self.get_logger().info(
                    "[VERBOSE] "
                    f"x=({self.x_current[0]:.2f},{self.x_current[1]:.2f},{self.x_current[2]:.2f}) "
                    f"ref=({self.ref[0]:.2f},{self.ref[1]:.2f},{self.ref[2]:.2f}) v_ref={self.v_ref:.2f} "
                    f"obs=({self.obstacle_xy[0]:.2f},{self.obstacle_xy[1]:.2f}) "
                    f"errors: e_long={e_long:.3f}, e_lat={e_lat:.3f}"
                )
                if diag:
                    self.get_logger().info(
                        "[VERBOSE] Horizon summary → "
                        f"min_h={diag['min_h']:.3f} at k={diag['min_h_idx']} (dist={diag['dist_at_min']:.3f}) | "
                        f"cbf_margin_min={diag['cbf_margin_min']:.3f} | "
                        f"first_near_idx={diag['first_near_idx']} | "
                        f"sin_tangent@min={diag['sin_at_min']:.3f} "
                        f"cos_toward@min={diag['cos_toward_at_min']:.3f} "
                        f"near_gate@min={diag['near_gate_at_min']:.3f} | "
                        f"v@min={diag['v_at_min']:.3f}, w@min={diag['w_at_min']:.3f} | "
                        f"side_pref={'LEFT(+1)' if diag['side_pref']>0 else 'RIGHT(-1)'} "
                    )

        # Final short state line
        self.get_logger().info(
            f"[STATE] v={v_cmd:.3f}, w={w_cmd:.3f} | "
            f"WL={wl:.2f}, WR={wr:.2f} | x={self.x_current[0]:.2f}, y={self.x_current[1]:.2f}"
        )

    # ---------------- main ----------------
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
