#!/usr/bin/env python3
import rclpy
import numpy as np
import math
import traceback
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool, String, Float32
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from giu_f1t_interfaces.msg import VehicleStateArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from .logger_utils import KAYNLogger, LogLevel
from .controllers.bicycle_model import BicycleModel
from .controllers.lqr import LQRController
from .controllers.mpc import MPCController, ACADOS_AVAILABLE
from .controllers.stanley import StanleyController
from .controllers.params import WHEELBASE, DELTA_MAX, A_MAX, V_MAX
from .supervisor.curvature import CurvatureEstimator
from .supervisor.fsm import FSM

try:
    from tf_transformations import euler_from_quaternion
except ImportError:
    KAYNLogger(self, "Kayn").warn("tf_transformations not found -> using custom quaternion to euler conversion")
    def euler_from_quaternion(q):
        x, y, z, w = q
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        return 0.0, 0.0, math.atan2(siny, cosy)



_CURVE_STATES = {'CURVE', 'BLEND_OUT', 'BLEND_IN'}


class _NullMPC:
    """Placeholder used when acados is unavailable; FSM never calls it after reconfiguration."""
    def compute_control(self, *args, **kwargs):
        KAYNLogger(self, "MPC").error("Attempted to call MPC controller, but acados is not installed!")
        raise RuntimeError("acados unavailable — MPC must not be called")


class KAYNNode(Node):
    def __init__(self):
        super().__init__('kayn_controller_node')
        self._log = KAYNLogger(self, "Kayn")
        self._declare_params()
        self._load_params()

        try:
            self._log.set_level(LogLevel.DEBUG if self.debug else LogLevel.NORMAL)
        except Exception:
            self._log.set_level(LogLevel.NORMAL)

        self._build_controllers()
        self._init_state()
        self._setup_subs()
        self._setup_pubs()
        self._setup_timers()
        self._log.startup(
            f"KAYN ready | {self.control_hz}Hz | "
            f"warmup={self.warmup_ctrl} straight={self.straight_ctrl} "
            f"curve={self.curve_ctrl} fallback={self.fallback_ctrl}"
        )

    def _p(self, name: str):
        return self.get_parameter(name).value

    def _declare_params(self):
        p = self.declare_parameter
        p('wheelbase', WHEELBASE);    p('dt', 0.02)
        p('lqr.q_px', 5.0);          p('lqr.q_py', 5.0)
        p('lqr.q_theta', 6.0);       p('lqr.q_v', 1.0)
        p('lqr.r_delta', 4.0);       p('lqr.r_a', 0.3)
        p('mpc.horizon_n', 20);      p('mpc.timeout_ms', 5.0);  p('mpc.dt', 0.02)
        p('mpc.q_px', 5.0);          p('mpc.q_py', 5.0)
        p('mpc.q_theta', 6.0);       p('mpc.q_v', 1.0)
        p('mpc.r_delta', 4.0);       p('mpc.r_a', 0.3)
        p('stanley.k', 1.5)
        p('fsm.warmup_controller',   'stanley')
        p('fsm.straight_controller', 'lqr')
        p('fsm.curve_controller',    'mpc')
        p('fsm.fallback_controller', 'stanley')
        p('fsm.warmup_steps', 50)
        p('fsm.enter_threshold', 0.10); p('fsm.exit_threshold', 0.06)
        p('fsm.confirm_steps', 3);   p('fsm.blend_window', 5)
        p('fsm.lookahead', 10)
        p('fsm.v_warmup_min', 0.2);  p('fsm.v_stop', 0.05)
        p('fsm.stop_confirm_steps', 20)
        p('max_speed', V_MAX);       p('max_steering', DELTA_MAX); p('max_accel', A_MAX)
        p('control_hz', 50.0)
        p('odom_topic', '/odom')
        p('trajectory_topic', '/horizon_mapper/reference_trajectory')
        p('path_ready_topic', '/horizon_mapper/path_ready')
        p('drive_topic', '/kayn/drive')
        p('debug', False);            p('log_every_n', 25)
        p('reverse_direction', False)
        p('straight_speed_scale', 1.0)
        p('curve_speed_scale',    1.0)
        p('publish_diagnostics', False)
        p('publish_controller_mode', True)
        p('publish_fsm_state', True)
        p('publish_curvature', False)
        p('publish_cross_track_error', False)

    def _load_params(self):
        self.wheelbase     = self._p('wheelbase')
        self.dt            = self._p('dt')
        self.mpc_n         = self._p('mpc.horizon_n')
        self.mpc_dt        = self._p('mpc.dt')
        self.max_speed     = self._p('max_speed')
        self.max_steering  = self._p('max_steering')
        self.max_accel     = self._p('max_accel')
        self.control_hz    = self._p('control_hz')
        self.odom_topic    = self._p('odom_topic')
        self.traj_topic    = self._p('trajectory_topic')
        self.ready_topic   = self._p('path_ready_topic')
        self.drive_topic   = self._p('drive_topic')
        self.debug         = self._p('debug')
        self.log_every_n   = self._p('log_every_n')
        self.publish_diagnostics = self._p('publish_diagnostics')
        self.publish_controller_mode = self._p('publish_controller_mode')
        self.publish_fsm_state = self._p('publish_fsm_state')
        self.publish_curvature = self._p('publish_curvature')
        self.publish_cross_track_error = self._p('publish_cross_track_error')

        self.lqr_Q = np.diag([self._p('lqr.q_px'), self._p('lqr.q_py'),
                               self._p('lqr.q_theta'), self._p('lqr.q_v')])
        self.lqr_R = np.diag([self._p('lqr.r_delta'), self._p('lqr.r_a')])
        self.mpc_Q = np.diag([self._p('mpc.q_px'), self._p('mpc.q_py'),
                               self._p('mpc.q_theta'), self._p('mpc.q_v')])
        self.mpc_R = np.diag([self._p('mpc.r_delta'), self._p('mpc.r_a')])

        self.stanley_k      = self._p('stanley.k')
        self.warmup_ctrl    = self._p('fsm.warmup_controller')
        self.straight_ctrl  = self._p('fsm.straight_controller')
        self.curve_ctrl     = self._p('fsm.curve_controller')
        self.fallback_ctrl  = self._p('fsm.fallback_controller')
        self.warmup_steps   = self._p('fsm.warmup_steps')
        self.confirm_steps       = self._p('fsm.confirm_steps')
        self.blend_window        = self._p('fsm.blend_window')
        self.curv_lookahead      = self._p('fsm.lookahead')
        self.v_warmup_min        = self._p('fsm.v_warmup_min')
        self.v_stop              = self._p('fsm.v_stop')
        self.stop_confirm_steps  = self._p('fsm.stop_confirm_steps')
        self.enter_threshold = self._p('fsm.enter_threshold')
        self.exit_threshold  = self._p('fsm.exit_threshold')
        self.mpc_timeout_s      = self._p('mpc.timeout_ms') / 1000.0
        self.reverse_direction  = self._p('reverse_direction')
        self.straight_speed_scale = self._p('straight_speed_scale')
        self.curve_speed_scale    = self._p('curve_speed_scale')

        if not ACADOS_AVAILABLE and self.curve_ctrl == 'mpc':
            self._log.critical("acados is not installed — MPC controller unavailable")
            self._log.warn(
                f"MPC skipped — curve controller falling back to '{self.fallback_ctrl}'"
            )
            self.curve_ctrl = self.fallback_ctrl

    def _build_controllers(self):
        model = BicycleModel(L=self.wheelbase, dt=self.dt,
                             delta_max=self.max_steering,
                             a_max=self.max_accel,
                             v_max=self.max_speed)
        curv_est = CurvatureEstimator(
            lookahead=self.curv_lookahead,
            enter_threshold=self.enter_threshold,
            exit_threshold=self.exit_threshold,
        )
        mpc_ctrl = (
            MPCController(model, N=self.mpc_n, dt=self.mpc_dt,
                          Q=self.mpc_Q, R=self.mpc_R, v_max=self.max_speed)
            if ACADOS_AVAILABLE
            else _NullMPC()
        )
        self.fsm = FSM(
            lqr=LQRController(model, Q=self.lqr_Q, R=self.lqr_R),
            mpc=mpc_ctrl,
            stanley=StanleyController(k=self.stanley_k, model=model),
            curvature_estimator=curv_est,
            warmup_steps=self.warmup_steps,
            warmup_ctrl=self.warmup_ctrl,
            straight_ctrl=self.straight_ctrl,
            curve_ctrl=self.curve_ctrl,
            fallback_ctrl=self.fallback_ctrl,
            confirm_steps=self.confirm_steps,
            blend_window=self.blend_window,
            mpc_timeout_s=self.mpc_timeout_s,
            v_warmup_min=self.v_warmup_min,
            v_stop=self.v_stop,
            stop_confirm_steps=self.stop_confirm_steps,
            dt=self.dt,
            logger=KAYNLogger(self, "FSM"),
        )

    def _init_state(self):
        self.x_curr     = None
        self.trajectory = []
        self.path_ready = False
        self.ref_idx    = 0
        self._iter      = 0
        self._last_block = None

    def _setup_subs(self):
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        rel_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE, depth=10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos_profile)
        self.create_subscription(VehicleStateArray, self.traj_topic, self._traj_cb, rel_qos)
        self.create_subscription(Bool, self.ready_topic, self._ready_cb, rel_qos)

    def _setup_pubs(self):
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE, depth=10)

        self._pub_drive = self.create_publisher(AckermannDriveStamped, self.drive_topic, qos)
        
        if (self.publish_controller_mode):
            self._pub_mode  = self.create_publisher(String,  '/kayn/mode', qos)
        
        if (self.publish_fsm_state):
            self._pub_fsm_state = self.create_publisher(String, '/kayn/fsm_state', qos)
        
        if (self.publish_cross_track_error):
            self._pub_cte   = self.create_publisher(Float32, '/kayn/cross_track_error', qos)
        
        if (self.publish_curvature):
            self._pub_kappa = self.create_publisher(Float32, '/kayn/curvature', qos)
        
        if (self.publish_diagnostics):
            self._pub_diag  = self.create_publisher(DiagnosticArray, '/kayn/diagnostics', qos)

    def _setup_timers(self):
        self.create_timer(1.0 / self.control_hz, self._control_cb)
        if self.publish_diagnostics:
            self.create_timer(1.0, self._diag_cb)

    def _odom_cb(self, msg: Odometry):
        try:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            self.x_curr = np.array([p.x, p.y, yaw, math.sqrt(vx*vx + vy*vy)])
        except Exception as e:
            self._log.error("odom_cb", exception=e)

    def _traj_cb(self, msg: VehicleStateArray):
        if self.reverse_direction:
            self.trajectory = [
                {'x': s.x, 'y': s.y,
                 'theta': BicycleModel.normalize_angle(s.theta + math.pi),
                 'v': s.v}
                for s in msg.states
            ]
        else:
            self.trajectory = [
                {'x': s.x, 'y': s.y, 'theta': s.theta, 'v': s.v}
                for s in msg.states
            ]

    def _ready_cb(self, msg: Bool):
        prev = self.path_ready
        self.path_ready = msg.data
        if self.path_ready != prev:
            self._log.event("path_ready", f"ready={self.path_ready}")

    def _control_cb(self):
        self._iter += 1

        if not self.path_ready:
            self._block("path not ready"); return
        if self.x_curr is None:
            self._block("no odometry"); return
        if len(self.trajectory) < 5:
            self._block("trajectory too short"); return

        try:
            pts = np.array([[wp['x'], wp['y']] for wp in self.trajectory])
            self.ref_idx = int(np.argmin(np.linalg.norm(pts - self.x_curr[:2], axis=1)))

            scale = (self.curve_speed_scale
                     if self.fsm.state_name in _CURVE_STATES
                     else self.straight_speed_scale)
            traj = ([{**wp, 'v': wp['v'] * scale} for wp in self.trajectory]
                    if scale != 1.0 else self.trajectory)

            u = self.fsm.step(self.x_curr, traj, self.ref_idx)
            self._last_block = None

            kappa = None
            cte = None

            # Drive command
            drive = AckermannDriveStamped()
            drive.header.stamp = self.get_clock().now().to_msg()
            drive.header.frame_id = 'base_link'
            drive.drive.steering_angle = float(np.clip(u[0], -self.max_steering, self.max_steering))
            drive.drive.acceleration   = float(np.clip(u[1], -self.max_accel, self.max_accel))
            drive.drive.speed = float(np.clip(
                traj[self.ref_idx]['v'], 0.0, self.max_speed))
            self._pub_drive.publish(drive)

            # Telemetry
            if self.publish_controller_mode:
                self._pub_mode.publish(String(data=self.fsm.state_name))
            
            if self.publish_fsm_state:
                self._pub_fsm_state.publish(String(data=self.fsm.state_name))

            if self.publish_cross_track_error or self.publish_curvature:
                wp = self.trajectory[self.ref_idx]
                perp = np.array([-math.sin(wp['theta']), math.cos(wp['theta'])])
                cte  = float(np.dot(self.x_curr[:2] - np.array([wp['x'], wp['y']]), perp))
                kappa = float(self.fsm.curv_est.estimate(self.trajectory, self.ref_idx))

                if self.publish_cross_track_error:
                    self._pub_cte.publish(Float32(data=cte))

                if self.publish_curvature:
                    self._pub_kappa.publish(Float32(data=kappa))

            if self.debug or self._iter % self.log_every_n == 0:
                kappa_str = f"{kappa:.3f}" if kappa is not None else "n/a"
                cte_str = f"{cte:.3f}" if cte is not None else "n/a"
                self._log.info(
                    f"[{self._iter}] mode={self.fsm.state_name} "
                    f"v={self.x_curr[3]:.2f} steer={u[0]:.3f} "
                    f"kappa={kappa_str} cte={cte_str}",
                    LogLevel.DEBUG,
                )

        except Exception as e:
            self._log.error("control_cb", exception=e)
            if self.debug:
                traceback.print_exc()

    def _block(self, reason: str):
        if reason != self._last_block:
            self._log.warn(f"KAYN blocked: {reason}")
            self._last_block = reason
        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.drive.speed = 0.0
        self._pub_drive.publish(drive)

    def _diag_cb(self):
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        s = DiagnosticStatus()
        s.name = 'kayn_controller'; s.hardware_id = 'kayn'
        s.level = DiagnosticStatus.OK; s.message = self.fsm.state_name
        s.values = [
            KeyValue(key='mode',       value=self.fsm.state_name),
            KeyValue(key='path_ready', value=str(self.path_ready)),
            KeyValue(key='traj_len',   value=str(len(self.trajectory))),
            KeyValue(key='iter',       value=str(self._iter)),
        ]
        msg.status.append(s)
        self._pub_diag.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KAYNNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._log.shutdown("Shutting down KAYN controller node")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
