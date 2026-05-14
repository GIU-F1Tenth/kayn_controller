"""
KAYN FSM Supervisor
===================

6 states: WARMUP, STRAIGHT, BLEND_OUT, CURVE, BLEND_IN, FALLBACK

Each state has an independently configurable controller slot (kayn_params.yaml):
  fsm.warmup_controller   : controller run during warmup       (default: stanley)
  fsm.straight_controller : controller run on straights        (default: lqr)
  fsm.curve_controller    : controller run on curves           (default: mpc)
  fsm.fallback_controller : controller run when curve fails    (default: stanley)

Valid values: "stanley" | "lqr" | "mpc"

Transition table:
  WARMUP    → STRAIGHT  : WARMUP_STEPS elapsed
  STRAIGHT  → BLEND_OUT : kappa > 0.10 for CONFIRM_STEPS consecutive samples
  STRAIGHT  → FALLBACK  : straight_ctrl=lqr AND LQR raises exception or returns invalid output
  BLEND_OUT → CURVE     : BLEND_WINDOW steps complete (blends straight→curve ctrl)
  BLEND_OUT → FALLBACK  : straight_ctrl=lqr AND LQR fails during blend-out
  CURVE     → BLEND_IN  : kappa < 0.06 for CONFIRM_STEPS consecutive samples
  CURVE     → FALLBACK  : curve_ctrl=mpc AND (solve_time > 5ms OR status != 0)
  BLEND_IN  → STRAIGHT  : BLEND_WINDOW steps complete (blends curve→straight ctrl)
  BLEND_IN  → FALLBACK  : straight_ctrl=lqr AND LQR fails during blend-in
  FALLBACK  → CURVE     : curve_ctrl=mpc AND CONFIRM_STEPS healthy MPC solves AND kappa > 0.10
  FALLBACK  → STRAIGHT  : curve_ctrl=mpc AND CONFIRM_STEPS healthy MPC solves AND kappa <= 0.10

LQR failure escalation (avoids loops):
  If fallback_ctrl=lqr and LQR also fails in FALLBACK → Stanley used as safe backup
  If Stanley also fails                                → zeroed command published

Blend zones: linear interpolation over BLEND_WINDOW steps, preventing steering jumps.
"""

import numpy as np
from enum import Enum, auto
from typing import List, Dict

from .curvature import CurvatureEstimator
from .state_handoff import handoff

_STANLEY_V_KP = 2.0  # proportional gain for Stanley speed control [1/s]

BLEND_WINDOW  = 5       # steps for output blending at transitions
CONFIRM_STEPS = 3       # consecutive samples needed to confirm a transition
MPC_TIMEOUT_S = 0.005   # 5ms solver budget
WARMUP_STEPS  = 50      # warmup steps before handing off to straight controller (1s at 50Hz)

_VALID_CTRLS = {'stanley', 'lqr', 'mpc'}


class KAYNState(Enum):
    WARMUP    = auto()
    STRAIGHT  = auto()
    BLEND_OUT = auto()   # straight → curve transition
    CURVE     = auto()
    BLEND_IN  = auto()   # curve → straight transition
    FALLBACK  = auto()


class FSM:
    def __init__(self, lqr, mpc, stanley, curvature_estimator: CurvatureEstimator,
                 warmup_steps: int = WARMUP_STEPS,
                 warmup_ctrl: str = 'stanley',
                 straight_ctrl: str = 'lqr',
                 curve_ctrl: str = 'mpc',
                 fallback_ctrl: str = 'stanley',
                 confirm_steps: int = CONFIRM_STEPS,
                 blend_window: int = BLEND_WINDOW,
                 mpc_timeout_s: float = MPC_TIMEOUT_S,
                 v_warmup_min: float = 0.2,
                 v_stop: float = 0.05,
                 stop_confirm_steps: int = 20,
                 dt: float = 0.005,
                 logger=print):
        for slot, val in [('warmup', warmup_ctrl), ('straight', straight_ctrl),
                          ('curve', curve_ctrl),   ('fallback', fallback_ctrl)]:
            if val not in _VALID_CTRLS:
                raise ValueError(f"fsm.{slot}_controller={val!r} — must be one of {_VALID_CTRLS}")

        self.lqr      = lqr
        self.mpc      = mpc
        self.stanley  = stanley
        self.curv_est = curvature_estimator
        self._warmup_steps       = warmup_steps
        self._warmup_ctrl        = warmup_ctrl
        self._straight_ctrl      = straight_ctrl
        self._curve_ctrl         = curve_ctrl
        self._fallback_ctrl      = fallback_ctrl
        self._confirm_steps      = confirm_steps
        self._blend_window       = blend_window
        self._mpc_timeout_s      = mpc_timeout_s
        # Velocity thresholds [m/s] used for warmup and stop detection.
        self._speed_moving = v_warmup_min
        self._speed_stopped = v_stop
        self._stop_confirm_steps = stop_confirm_steps

        if callable(logger) and not hasattr(logger, 'event'):
            _fn = logger
            class _Compat:
                def event(self, name, details='', level=None): _fn(f"[KAYN] {name} | {details}")
                def warn(self, msg, level=None): _fn(f"[KAYN] WARNING: {msg}")
            self._log = _Compat()
        else:
            self._log = logger

        self.state = KAYNState.WARMUP
        self._warmup_count   = 0
        self._confirm_count  = 0
        self._blend_step     = 0
        self._recovery_count = 0
        self._stop_count     = 0

    def step(self, x_curr: np.ndarray, trajectory: List[Dict],
             ref_idx: int) -> np.ndarray:
        """One FSM control step. Returns u = [delta, a]."""
        kappa = self.curv_est.estimate(trajectory, ref_idx)
        speed = float(x_curr[3])

        # Re-enter WARMUP if the vehicle slows to a stop in any running state
        if self.state != KAYNState.WARMUP:
            if speed < self._speed_stopped:
                self._stop_count += 1
                if self._stop_count >= self._stop_confirm_steps:
                    self._transition(KAYNState.WARMUP,
                                     f"stopped speed={speed:.3f}m/s",
                                     x_curr, trajectory, ref_idx)
            else:
                self._stop_count = 0

        if self.state == KAYNState.WARMUP:
            return self._step_warmup(x_curr, trajectory, ref_idx, speed)
        elif self.state == KAYNState.STRAIGHT:
            return self._step_straight(x_curr, trajectory, ref_idx, kappa)
        elif self.state == KAYNState.BLEND_OUT:
            return self._step_blend_out(x_curr, trajectory, ref_idx)
        elif self.state == KAYNState.CURVE:
            return self._step_curve(x_curr, trajectory, ref_idx, kappa)
        elif self.state == KAYNState.BLEND_IN:
            return self._step_blend_in(x_curr, trajectory, ref_idx)
        elif self.state == KAYNState.FALLBACK:
            return self._step_fallback(x_curr, trajectory, ref_idx, kappa)
        return np.zeros(2)

    @property
    def state_name(self) -> str:
        return self.state.name

    def _step_warmup(self, x_curr, trajectory, ref_idx, speed: float = 0.0) -> np.ndarray:
        u, _, _ = self._ctrl_u(self._warmup_ctrl, x_curr, trajectory, ref_idx)
        # Pre-heat straight controller so it's ready immediately after warmup
        try:
            self._ctrl_u(self._straight_ctrl, x_curr, trajectory, ref_idx)
        except Exception:
            pass
        # Hysteresis for warmup counting:
        # - increment when speed exceeds the 'moving' threshold
        # - reset only when speed falls below the 'stopped' threshold
        if speed >= self._speed_moving:
            self._warmup_count += 1
        elif speed < self._speed_stopped:
            self._warmup_count = 0
        if self._warmup_count >= self._warmup_steps:
            self._transition(KAYNState.STRAIGHT,
                             f"warmup_complete motion_steps={self._warmup_count}",
                             x_curr, trajectory, ref_idx)
        return u

    def _step_straight(self, x_curr, trajectory, ref_idx, kappa) -> np.ndarray:
        u, _, status = self._ctrl_u(self._straight_ctrl, x_curr, trajectory, ref_idx)
        if self._straight_ctrl == 'lqr' and status == -1:
            self._log.warn(
                f"LQR failed in STRAIGHT — switching to fallback controller ({self._fallback_ctrl})"
            )
            self._transition(KAYNState.FALLBACK, "lqr_failed", x_curr, trajectory, ref_idx)
            return self._safe_fallback_u(x_curr, trajectory, ref_idx)
        if kappa > self.curv_est.enter_threshold:
            self._confirm_count += 1
            if self._confirm_count >= self._confirm_steps:
                self._transition(KAYNState.BLEND_OUT,
                                 f"kappa={kappa:.3f}", x_curr, trajectory, ref_idx)
        else:
            self._confirm_count = 0
        return u

    def _step_blend_out(self, x_curr, trajectory, ref_idx) -> np.ndarray:
        alpha = self._blend_step / self._blend_window
        u_out, _, status_out = self._ctrl_u(self._straight_ctrl, x_curr, trajectory, ref_idx)
        if self._straight_ctrl == 'lqr' and status_out == -1:
            self._log.warn(
                f"LQR failed in BLEND_OUT — switching to fallback controller ({self._fallback_ctrl})"
            )
            self._transition(KAYNState.FALLBACK, "lqr_failed_blend_out",
                             x_curr, trajectory, ref_idx)
            return self._safe_fallback_u(x_curr, trajectory, ref_idx)
        u_in, _, _ = self._ctrl_u(self._curve_ctrl, x_curr, trajectory, ref_idx)
        u = (1 - alpha) * u_out + alpha * u_in
        self._blend_step += 1
        if self._blend_step >= self._blend_window:
            self._transition(KAYNState.CURVE, "blend_complete", x_curr, trajectory, ref_idx)
        return u

    def _step_curve(self, x_curr, trajectory, ref_idx, kappa) -> np.ndarray:
        u, solve_time, status = self._ctrl_u(self._curve_ctrl, x_curr, trajectory, ref_idx)

        if self._curve_ctrl == 'mpc' and (status != 0 or solve_time > self._mpc_timeout_s):
            reason = (f"solver_timeout={solve_time*1000:.1f}ms"
                      if solve_time > self._mpc_timeout_s else f"infeasible status={status}")
            self._log.warn(
                f"MPC skipped: {reason} — fallback controller '{self._fallback_ctrl}' taking over"
            )
            self._transition(KAYNState.FALLBACK, reason, x_curr, trajectory, ref_idx)
            u_fb, _, _ = self._ctrl_u(self._fallback_ctrl, x_curr, trajectory, ref_idx)
            return u_fb

        if kappa < self.curv_est.exit_threshold:
            self._confirm_count += 1
            if self._confirm_count >= self._confirm_steps:
                self._transition(KAYNState.BLEND_IN, f"kappa={kappa:.3f}",
                                 x_curr, trajectory, ref_idx)
        else:
            self._confirm_count = 0
        return u

    def _step_blend_in(self, x_curr, trajectory, ref_idx) -> np.ndarray:
        alpha = self._blend_step / self._blend_window
        u_out, solve_time, status     = self._ctrl_u(self._curve_ctrl,    x_curr, trajectory, ref_idx)
        u_in,  _,          status_in  = self._ctrl_u(self._straight_ctrl, x_curr, trajectory, ref_idx)

        if self._curve_ctrl == 'mpc' and (status != 0 or solve_time > self._mpc_timeout_s):
            self._log.warn(
                f"MPC infeasible during BLEND_IN (status={status}, t={solve_time*1000:.1f}ms) "
                f"— transitioning to STRAIGHT"
            )
            self._transition(KAYNState.STRAIGHT, "blend_in_mpc_infeasible",
                             x_curr, trajectory, ref_idx)
            return u_in

        if self._straight_ctrl == 'lqr' and status_in == -1:
            self._log.warn(
                f"LQR failed in BLEND_IN — switching to fallback controller ({self._fallback_ctrl})"
            )
            self._transition(KAYNState.FALLBACK, "lqr_failed_blend_in",
                             x_curr, trajectory, ref_idx)
            return self._safe_fallback_u(x_curr, trajectory, ref_idx)

        u = (1 - alpha) * u_out + alpha * u_in
        self._blend_step += 1
        if self._blend_step >= self._blend_window:
            self._transition(KAYNState.STRAIGHT, "blend_complete", x_curr, trajectory, ref_idx)
        return u

    def _step_fallback(self, x_curr, trajectory, ref_idx, kappa) -> np.ndarray:
        u, _, status = self._ctrl_u(self._fallback_ctrl, x_curr, trajectory, ref_idx)

        if self._fallback_ctrl == 'lqr' and status == -1:
            self._log.warn(
                f"Fallback controller (LQR) failed in FALLBACK state — using Stanley as safe backup"
            )
            try:
                u, _, _ = self._ctrl_u('stanley', x_curr, trajectory, ref_idx)
            except Exception as exc:
                self._log.warn(f"Stanley backup also failed: {exc} — publishing zeroed command")
                u = np.zeros(2)
            return u

        # Recovery probe only makes sense when the curve controller is MPC
        if self._curve_ctrl == 'mpc':
            try:
                ref_slice = trajectory[ref_idx:]
                if len(ref_slice) < 2:
                    ref_slice = trajectory[-2:]
                _, bg_time, bg_status = self.mpc.compute_control(x_curr, ref_slice)
                if bg_status == 0 and bg_time < self._mpc_timeout_s:
                    self._recovery_count += 1
                else:
                    self._recovery_count = 0
            except Exception:
                self._recovery_count = 0

            if self._recovery_count >= self._confirm_steps:
                target = KAYNState.CURVE if kappa > self.curv_est.enter_threshold else KAYNState.STRAIGHT
                self._transition(target, f"solver_recovered kappa={kappa:.3f}",
                                 x_curr, trajectory, ref_idx)
        return u

    def _ctrl_u(self, name: str, x_curr: np.ndarray,
                trajectory: List[Dict], ref_idx: int):
        """Run named controller. Returns (u: np.ndarray, solve_time: float, status: int).

        status == -1 signals an LQR exception — callers treat this as a failure that
        should trigger fallback.  MPC keeps its own status codes (0 = ok, != 0 = infeasible).
        """
        if name == 'mpc':
            # Pass full remaining track — MPCController builds time-advance ref internally
            ref_slice = trajectory[ref_idx:]
            if len(ref_slice) < 2:
                ref_slice = trajectory[-2:]
            return self.mpc.compute_control(x_curr, ref_slice)
        elif name == 'lqr':
            try:
                kappa = self.curv_est.estimate(trajectory, ref_idx)
                delta_ff = float(np.arctan(self.lqr.model.L * kappa))
                u = self.lqr.compute_control(x_curr, self._ref_state(trajectory, ref_idx),
                                              np.array([delta_ff, 0.0]))
                return u, 0.0, 0
            except Exception as exc:
                self._log.warn(f"[KAYN] LQR raised exception: {exc}")
                return np.zeros(2), 0.0, -1
        else:  # stanley
            delta = self.stanley.compute_control(x_curr, trajectory)
            v_ref = trajectory[min(ref_idx, len(trajectory) - 1)]['v']
            a = float(np.clip(_STANLEY_V_KP * (v_ref - x_curr[3]),
                              -self.stanley.model.a_max, self.stanley.model.a_max))
            return np.array([delta, a]), 0.0, 0

    def _safe_fallback_u(self, x_curr: np.ndarray,
                         trajectory: List[Dict], ref_idx: int) -> np.ndarray:
        """Call the configured fallback controller.

        If fallback_ctrl is LQR and it also fails, escalate to Stanley rather than
        looping.  If Stanley fails too, publish a zeroed command.
        """
        u, _, status = self._ctrl_u(self._fallback_ctrl, x_curr, trajectory, ref_idx)
        if self._fallback_ctrl == 'lqr' and status == -1:
            self._log.warn(
                f"Fallback controller (LQR) also failed — using Stanley as safe backup"
            )
            try:
                u, _, _ = self._ctrl_u('stanley', x_curr, trajectory, ref_idx)
            except Exception as exc:
                self._log.warn(f"Stanley backup also failed: {exc} — publishing zeroed command")
                u = np.zeros(2)
        return u

    def _transition(self, new_state: KAYNState, reason: str,
                    x_curr: np.ndarray, trajectory: List[Dict],
                    ref_idx: int) -> None:
        self._log.event(
            f"{self.state.name} → {new_state.name}",
            details=f"{reason} | idx={ref_idx}",
        )
        self.state = new_state
        self._confirm_count  = 0
        self._blend_step     = 0
        self._recovery_count = 0
        self._stop_count     = 0
        if new_state == KAYNState.WARMUP:
            self._warmup_count = 0

        if new_state in (KAYNState.CURVE, KAYNState.BLEND_OUT):
            handoff(self.mpc, x_curr, trajectory, ref_idx)
        elif new_state in (KAYNState.STRAIGHT, KAYNState.BLEND_IN):
            handoff(self.lqr, x_curr, trajectory, ref_idx)

    def _ref_state(self, trajectory: List[Dict], idx: int) -> np.ndarray:
        wp = trajectory[min(idx, len(trajectory) - 1)]
        return np.array([wp['x'], wp['y'], wp['theta'], wp['v']])
