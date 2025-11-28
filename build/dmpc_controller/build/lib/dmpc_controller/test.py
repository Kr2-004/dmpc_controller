#!/usr/bin/env python3
import time
import numpy as np
from typing import Optional, Dict, List

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Float32MultiArray
import casadi as ca

def quat_to_yaw(qz: float, qw: float)  -> float:
    return 2.0 * np.arctan2(qz, qw)


class DMPCNode(Node):
    def __init__(self) -> None:
        super().__init__('dmpc_cbf_multirobot')

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter('robot_name', 'puzzlebot1')
        self.declare_parameter('robot_list', ['puzzlebot1', 'puzzlebot2'])

        self.robot_name: str = self.get_parameter('robot_name').value
        all_robots: List[str] = self.get_parameter('robot_list').value

        # Remove myself from peer list
        self.peer_names: List[str] = [r for r in all_robots if r != self.robot_name]

        # -----------------------------
        # Robot dynamics parameters
        # -----------------------------
        self.r_wheel = 0.05
        self.b_track = 0.085
        self.v_max   = 0.20
        self.w_max   = 6.0
        self.omega_max = 7.0

        # MPC horizon
        self.Ts = 0.01
        self.N  = 50

        # Costs
        self.Qp  = np.diag([8.0, 25.0])
        self.Qth = 2.0
        self.Qv  = 0.8
        self.R   = np.diag([1.0, 0.5])
        self.Sdu = 2.0

        # -----------------------------
        # CBF parameters
        # -----------------------------
        self.R_robot   = 0.20
        self.margin    = 0.05
        self.alpha_cbf = 20.0
        self.rho_slack = 1e4

        # -----------------------------
        # State variables
        # -----------------------------
        self.x_current = None
        self.ref       = None
        self.u_last    = np.array([0.05, 0.0])
        self.v_ref     = 0.25
        self.start_wall_time = time.time()

        # All peer positions (dictionary)
        # peer_name → np.array([x, y])
        self.peers: Dict[str, np.ndarray] = {}

        # Warm start
        self.U_prev = None

        # -----------------------------
        # Build solver
        # -----------------------------
        self._build_solver()

        # -----------------------------
        # ROS I/O
        #------------------------------
        # Sub: own pose
        self.create_subscription(
            PoseStamped,
            f'/vicon/{self.robot_name}/{self.robot_name}/pose',
            self.pose_cb,
            20
        )

        # Sub: reference
        self.create_subscription(
            PoseStamped,
            f'/vs/reference/{self.robot_name}',
            self.vs_cb,
            20
        )

        # Sub: reference speed
        self.create_subscription(
            Float32,
            '/vs/reference_speed',
            self.vref_cb,
            10
        )

        # Sub: dynamic peer subscriptions
        for peer in self.peer_names:
            self.create_subscription(
                PoseStamped,
                f'/vicon/{peer}/{peer}/pose',
                lambda msg, name=peer: self.peer_cb(msg, name),
                20
            )

        # Publishers
        self.pub_L = self.create_publisher(Float32, f'/{self.robot_name}/VelocitySetL', 10)
        self.pub_R = self.create_publisher(Float32, f'/{self.robot_name}/VelocitySetR', 10)
        self.pub_cbf = self.create_publisher(Float32MultiArray, '/cbf_monitor', 10)

        self.create_timer(self.Ts, self.control_step)
        self.get_logger().info(f"🚀 Symmetric Multi-Robot MPC+CBF ready for {self.robot_name}")


    # --------------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------------
    def pose_cb(self, msg: PoseStamped):
        self.x_current = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            quat_to_yaw(msg.pose.orientation.z, msg.pose.orientation.w)
        ], dtype=float)

    def vs_cb(self, msg: PoseStamped):
        self.ref = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            quat_to_yaw(msg.pose.orientation.z, msg.pose.orientation.w)
        ], dtype=float)

    def vref_cb(self, msg: Float32):
        self.v_ref = float(msg.data)

    def peer_cb(self, msg: PoseStamped, peer_name: str):
        self.peers[peer_name] = np.array([msg.pose.position.x, msg.pose.position.y], dtype=float)


    # --------------------------------------------------------------
    # Build MPC + Multi-CBF Solver
    # --------------------------------------------------------------
    def _build_solver(self):
        nx, nu, N, Ts = 3, 2, self.N, self.Ts

        # Decision variables U and slack delta
        U = ca.SX.sym('U', nu*N)
        delta = ca.SX.sym('delta')

        # Parameters
        x0 = ca.SX.sym('x0', nx)
        ref = ca.SX.sym('ref', nx)
        u_last = ca.SX.sym('u_last', nu)
        v_ref = ca.SX.sym('v_ref')

        # Pack all peers inside parameter vector
        # For k peers → vector of length 2*K
        num_peers = len(self.peer_names)
        peer_vec = ca.SX.sym('peer_vec', 2*num_peers)
        cbf_on_vec = ca.SX.sym('cbf_on_vec', num_peers)

        def f(x, u):
            th = x[2]
            v, w = u[0], u[1]
            return ca.vertcat(
                x[0] + Ts * v * ca.cos(th),
                x[1] + Ts * v * ca.sin(th),
                x[2] + Ts * w
            )

        # -------------------------
        # COST
        # -------------------------
        J = 0
        Xpred = []
        xk = x0

        for j in range(N):
            uj = U[j*nu:(j+1)*nu]
            xk = f(xk, uj)
            Xpred.append(xk)

            # Predict future reference along path
            ref_jx = ref[0] + v_ref * Ts * j * ca.cos(ref[2])
            ref_jy = ref[1] + v_ref * Ts * j * ca.sin(ref[2])
            ref_jth = ref[2]

            dx = xk[0] - ref_jx
            dy = xk[1] - ref_jy
            e_th = ca.fmod(xk[2] - ref_jth + ca.pi, 2*ca.pi) - ca.pi

            e_long = ca.cos(ref[2]) * dx + ca.sin(ref[2]) * dy
            e_lat  = -ca.sin(ref[2]) * dx + ca.cos(ref[2]) * dy

            # MPC tracking costs
            J += self.Qp[0,0]*(e_long**2) + self.Qp[1,1]*(e_lat**2)
            J += self.Qth*(e_th**2)
            J += ca.mtimes([uj.T, self.R, uj])

            # Adaptive forward-velocity tracking
            align_err = (e_lat**2 + 0.5*(e_th**2))
            alpha_gate = ca.exp(-4.0 * align_err)
            v_des = alpha_gate * v_ref
            J += self.Qv * ((uj[0] - v_des)**2)

            # Δu smoothness
            if j == 0:
                du = uj - u_last
            else:
                du = uj - U[(j-1)*nu:j*nu]
            J += self.Sdu * ca.mtimes([du.T, du])

        # Slack cost
        J += self.rho_slack * (delta**2)

        # -------------------------------------------------
        # CONSTRAINTS
        # -------------------------------------------------
        g = []

        # Wheel & actuator limits
        for j in range(N):
            v, w = U[j*2], U[j*2+1]
            wl = (2*v - w*self.b_track) / (2*self.r_wheel)
            wr = (2*v + w*self.b_track) / (2*self.r_wheel)

            g += [
                v, self.v_max - v,
                w + self.w_max, self.w_max - w,
                self.omega_max - wl,
                self.omega_max + wl,
                self.omega_max - wr,
                self.omega_max + wr
            ]

        # ----------------------------------------------
        # Multi-robot CBF: For each peer, for each step
        # ----------------------------------------------
        dmin = 2*self.R_robot + self.margin

        for k in range(num_peers):
            px = peer_vec[2*k]
            py = peer_vec[2*k+1]
            cbf_on_k = cbf_on_vec[k]

            for j in range(N):
                xj = Xpred[j]
                vj = U[j*2]
                thj = xj[2]
                pj = xj[0:2]

                peer_p = ca.vertcat(px, py)
                r = pj - peer_p
                h = ca.mtimes([r.T, r]) - dmin**2

                e_th_vec = ca.vertcat(ca.cos(thj), ca.sin(thj))
                dh = 2.0 * ca.mtimes([r.T, e_th_vec]) * vj

                g.append(cbf_on_k*(dh + self.alpha_cbf*h) + delta)

        # Slack >= 0
        g.append(delta)

        # NLP build
        Z = ca.vertcat(U, delta)
        params = ca.vertcat(
            x0, ref, u_last, v_ref,
            peer_vec, cbf_on_vec
        )

        nlp = {
            'x': Z,
            'f': J,
            'g': ca.vertcat(*g),
            'p': params
        }

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 40,
            'ipopt.tol': 1e-3
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.U_len = 2*self.N
        self.g_dim = len(g)

    # --------------------------------------------------------------
    # Control Loop
    # --------------------------------------------------------------
    def control_step(self):
        if self.x_current is None or self.ref is None:
            return

        # Warm start
        if self.U_prev is not None:
            U0 = np.hstack([self.U_prev[2:], self.U_prev[-2:]])
        else:
            U0 = np.tile(self.u_last, self.N)

        # Pack peer vector
        peer_vec = []
        cbf_on_vec = []

        for name in self.peer_names:
            if name in self.peers:
                peer_vec += self.peers[name].tolist()
                cbf_on_vec += [1.0]
            else:
                peer_vec += [0.0, 0.0]
                cbf_on_vec += [0.0]

        peer_vec = np.array(peer_vec)
        cbf_on_vec = np.array(cbf_on_vec)

        Z0 = np.concatenate([U0, [0.0]])  # d0 = 0

        params = np.concatenate([
            self.x_current, self.ref, self.u_last,
            [self.v_ref],
            peer_vec,
            cbf_on_vec
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
            delta = Zstar[-1]
            u0 = Ustar[:2]
            self.U_prev = Ustar

        except Exception as e:
            self.get_logger().warn(f"Solver failed: {e}")
            u0 = np.array([0.05, 0.0])
            delta = 0.0
            self.U_prev = None

        # Apply control
        v_cmd = float(np.clip(u0[0], 0.0, self.v_max))
        w_cmd = float(np.clip(u0[1], -self.w_max, self.w_max))

        wl = (2*v_cmd - w_cmd*self.b_track) / (2*self.r_wheel)
        wr = (2*v_cmd + w_cmd*self.b_track) / (2*self.r_wheel)
        wl = np.clip(wl, -self.omega_max, self.omega_max)
        wr = np.clip(wr, -self.omega_max, self.omega_max)

        self.pub_L.publish(Float32(data=wl))
        self.pub_R.publish(Float32(data=wr))
        self.u_last = np.array([v_cmd, w_cmd])

        # Publish CBF diagnostics
        msg = Float32MultiArray()
        msg.data = [float(delta)]
        self.pub_cbf.publish(msg)

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