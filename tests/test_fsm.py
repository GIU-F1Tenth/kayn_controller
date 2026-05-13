import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kayn_controller.controllers.bicycle_model import BicycleModel
from kayn_controller.controllers.lqr import LQRController
from kayn_controller.controllers.mpc import MPCController, ACADOS_AVAILABLE
from kayn_controller.controllers.stanley import StanleyController
from kayn_controller.supervisor.curvature import CurvatureEstimator
from kayn_controller.supervisor.fsm import FSM, KAYNState, CONFIRM_STEPS, BLEND_WINDOW, WARMUP_STEPS
from simulation.track import straight_track, curve_track


class MockMPC:
    """Drop-in MPC stub for tests that don't need real acados."""
    def compute_control(self, x, traj):
        return np.zeros(2), 0.001, 0


def _make_fsm(**kwargs):
    model = BicycleModel()
    mpc = MockMPC() if not ACADOS_AVAILABLE else MPCController(model)
    return FSM(
        lqr=LQRController(model),
        mpc=mpc,
        stanley=StanleyController(model=model),
        curvature_estimator=CurvatureEstimator(lookahead=10),
        **kwargs,
    )


def test_initial_state_is_warmup():
    fsm = _make_fsm()
    assert fsm.state == KAYNState.WARMUP


def test_warmup_transitions_to_straight():
    """After warmup_steps Stanley steps, FSM must enter STRAIGHT."""
    fsm = _make_fsm()
    track = straight_track(length=200.0, v_ref=2.0, n_points=300)
    x_curr = np.array([track[5]['x'], track[5]['y'], track[5]['theta'], 2.0])
    for _ in range(WARMUP_STEPS):
        fsm.step(x_curr, track, 5)
    assert fsm.state == KAYNState.STRAIGHT, f"Expected STRAIGHT, got {fsm.state.name}"


def test_straight_to_blend_out_on_high_curvature():
    """3 consecutive high-kappa samples must trigger STRAIGHT → BLEND_OUT."""
    fsm = _make_fsm()
    fsm.state = KAYNState.STRAIGHT   # skip warmup
    track = curve_track(radius=3.0, sweep_deg=180.0, v_ref=2.0, n_points=300)
    x_curr = np.array([track[10]['x'], track[10]['y'], track[10]['theta'], 2.0])

    for _ in range(CONFIRM_STEPS):
        fsm.step(x_curr, track, 10)

    assert fsm.state in (KAYNState.BLEND_OUT, KAYNState.CURVE), \
        f"Expected BLEND_OUT or CURVE, got {fsm.state.name}"


def test_blend_out_completes_to_curve():
    """After BLEND_WINDOW blend steps, state must be CURVE."""
    fsm = _make_fsm()
    fsm.state = KAYNState.STRAIGHT   # skip warmup
    track = curve_track(radius=3.0, sweep_deg=180.0, v_ref=2.0, n_points=300)
    x_curr = np.array([track[10]['x'], track[10]['y'], track[10]['theta'], 2.0])

    # Get into BLEND_OUT
    for _ in range(CONFIRM_STEPS):
        fsm.step(x_curr, track, 10)

    if fsm.state == KAYNState.BLEND_OUT:
        for _ in range(BLEND_WINDOW):
            fsm.step(x_curr, track, 10)
        assert fsm.state == KAYNState.CURVE, f"Expected CURVE, got {fsm.state.name}"


def test_no_steering_jump_at_transition():
    """Steering output must not jump more than 0.05 rad at any transition."""
    fsm = _make_fsm()
    fsm.state = KAYNState.STRAIGHT   # skip warmup
    model = BicycleModel()
    track = curve_track(radius=3.0, sweep_deg=180.0, v_ref=2.0, n_points=300)
    x_curr = np.array([track[5]['x'], track[5]['y'], track[5]['theta'], 2.0])

    last_delta = None
    for i in range(5, min(50, len(track))):
        u = fsm.step(x_curr, track, i)
        delta = u[0]
        if last_delta is not None:
            jump = abs(delta - last_delta)
            assert jump < 0.05, \
                f"Steering jump {jump:.4f} rad at step {i}, state={fsm.state.name}"
        last_delta = delta
        x_curr = model.step_rk4(x_curr, u)


def test_curve_slot_lqr_no_fallback(monkeypatch):
    """curve_controller=lqr must never trigger FALLBACK even if MPC would time out."""
    model = BicycleModel()
    fsm = FSM(
        lqr=LQRController(model),
        mpc=MockMPC(),
        stanley=StanleyController(model=model),
        curvature_estimator=CurvatureEstimator(lookahead=10),
        curve_ctrl='lqr',
    )
    fsm.state = KAYNState.CURVE
    track = curve_track(radius=3.0, sweep_deg=180.0, v_ref=2.0, n_points=300)
    x_curr = np.array([track[10]['x'], track[10]['y'], track[10]['theta'], 2.0])
    monkeypatch.setattr(fsm.mpc, 'compute_control',
                        lambda *a, **kw: (np.zeros(2), 0.010, 0))
    fsm.step(x_curr, track, 10)
    assert fsm.state != KAYNState.FALLBACK, \
        f"curve_ctrl=lqr should not enter FALLBACK, got {fsm.state.name}"


def test_invalid_controller_slot_raises():
    """Passing an unknown controller name must raise ValueError at construction."""
    model = BicycleModel()
    import pytest
    with pytest.raises(ValueError):
        FSM(
            lqr=LQRController(model),
            mpc=MockMPC(),
            stanley=StanleyController(model=model),
            curvature_estimator=CurvatureEstimator(lookahead=10),
            curve_ctrl='pure_pursuit',
        )


def test_fallback_on_mpc_timeout(monkeypatch):
    """Monkeypatched slow MPC must trigger CURVE → FALLBACK immediately."""
    fsm = _make_fsm()
    track = curve_track(radius=3.0, sweep_deg=180.0, v_ref=2.0, n_points=300)
    x_curr = np.array([track[10]['x'], track[10]['y'], track[10]['theta'], 2.0])
    fsm.state = KAYNState.CURVE
    monkeypatch.setattr(fsm.mpc, 'compute_control',
                        lambda *a, **kw: (np.zeros(2), 0.010, 0))
    fsm.step(x_curr, track, 10)
    assert fsm.state == KAYNState.FALLBACK, f"Expected FALLBACK, got {fsm.state.name}"


def test_blend_in_mpc_infeasible_uses_lqr(monkeypatch):
    """During BLEND_IN, infeasible MPC must not contaminate output — must transition to STRAIGHT."""
    fsm = _make_fsm()
    track = straight_track(length=50.0, v_ref=2.0, n_points=100)
    x_curr = np.array([track[5]['x'], track[5]['y'], track[5]['theta'], 2.0])
    fsm.state = KAYNState.BLEND_IN
    fsm._blend_step = 2
    monkeypatch.setattr(fsm.mpc, 'compute_control',
                        lambda x, traj: (np.array([0.9, 2.0]), 0.001, 1))
    u = fsm.step(x_curr, track, 5)
    assert abs(u[0]) < 0.5, \
        f"Infeasible MPC in BLEND_IN must not pass through delta=0.9, got {u[0]:.4f}"
    assert fsm.state == KAYNState.STRAIGHT, \
        f"Infeasible MPC in BLEND_IN must transition to STRAIGHT, got {fsm.state.name}"


def test_lqr_uses_curvature_feedforward(monkeypatch):
    """In STRAIGHT state on a curve, LQR must receive non-zero u_ref (curvature feedforward)."""
    fsm = _make_fsm()
    fsm.state = KAYNState.STRAIGHT
    track = curve_track(radius=3.0, sweep_deg=90.0, v_ref=2.0, n_points=100)
    x_curr = np.array([track[10]['x'], track[10]['y'], track[10]['theta'], 2.0])

    received_u_ref = []
    orig = fsm.lqr.compute_control
    def spy(x_c, x_r, u_ref=None):
        received_u_ref.append(u_ref.copy() if u_ref is not None else None)
        return orig(x_c, x_r, u_ref)
    monkeypatch.setattr(fsm.lqr, 'compute_control', spy)

    fsm.step(x_curr, track, 10)

    assert len(received_u_ref) > 0, "LQR must have been called"
    assert received_u_ref[-1] is not None, \
        "u_ref must not be None when curvature is non-zero"
    assert abs(received_u_ref[-1][0]) > 1e-4, \
        f"Expected non-zero feedforward steering on curve, got {received_u_ref[-1][0]:.6f}"
