import numpy as np
import time
import sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kayn_controller.controllers.bicycle_model import BicycleModel
from kayn_controller.controllers.mpc import MPCController, ACADOS_AVAILABLE
from simulation.track import curve_track, straight_track

_needs_acados = pytest.mark.skipif(not ACADOS_AVAILABLE, reason="acados not installed")


def test_acados_available_is_bool():
    """ACADOS_AVAILABLE must be a bool regardless of whether acados is installed."""
    assert isinstance(ACADOS_AVAILABLE, bool)


@_needs_acados
def test_ocp_setup_no_error():
    model = BicycleModel()
    mpc = MPCController(model)
    assert mpc.solver is not None


@_needs_acados
def test_feasible_on_curve():
    model = BicycleModel()
    mpc = MPCController(model)
    track = curve_track(radius=3.0, sweep_deg=90.0, v_ref=2.0, n_points=100)
    x_curr = np.array([track[0]['x'], track[0]['y'], track[0]['theta'], 2.0])
    ref_traj = track[:mpc.N + 1]
    u, solve_time, status = mpc.compute_control(x_curr, ref_traj)
    assert status == 0, f"MPC infeasible (status={status})"
    assert u.shape == (2,)


@_needs_acados
def test_solve_time_within_budget():
    model = BicycleModel()
    mpc = MPCController(model)
    track = curve_track(radius=3.0, sweep_deg=90.0, v_ref=2.0, n_points=100)
    x_curr = np.array([track[0]['x'], track[0]['y'], track[0]['theta'], 2.0])
    ref_traj = track[:mpc.N + 1]
    mpc.compute_control(x_curr, ref_traj)
    _, solve_time, _ = mpc.compute_control(x_curr, ref_traj)
    assert solve_time < 0.010, f"MPC too slow: {solve_time*1000:.1f}ms"


@_needs_acados
def test_steering_output_within_limits():
    model = BicycleModel()
    mpc = MPCController(model)
    track = curve_track(radius=3.0, sweep_deg=90.0, v_ref=2.0, n_points=100)
    x_curr = np.array([track[0]['x'], track[0]['y'], track[0]['theta'], 2.0])
    ref_traj = track[:mpc.N + 1]
    u, _, _ = mpc.compute_control(x_curr, ref_traj)
    assert abs(u[0]) <= 0.4189 + 1e-6, f"delta={u[0]:.4f} exceeds limit"


@_needs_acados
def test_acados_matches_intuition_on_straight():
    model = BicycleModel()
    mpc = MPCController(model)
    track = straight_track(length=50.0, v_ref=2.0, n_points=200)
    x_curr = np.array([0.0, 0.0, 0.0, 2.0])
    ref_traj = track[:mpc.N + 1]
    u, _, status = mpc.compute_control(x_curr, ref_traj)
    assert status == 0
    assert abs(u[0]) < 0.05, f"Expected near-zero steering on straight, got {u[0]:.4f}"


@_needs_acados
def test_heading_normalization_near_pi():
    model = BicycleModel()
    mpc = MPCController(model)
    track = []
    for i in range(40):
        theta = np.pi if i % 2 == 0 else -np.pi
        track.append({'x': float(-i) * 0.3, 'y': 0.0, 'theta': theta, 'v': 2.0})
    x_curr = np.array([0.0, 0.0, np.pi, 2.0])
    u, _, status = mpc.compute_control(x_curr, track)
    assert status == 0, f"MPC should be feasible near ±π heading (status={status})"
    assert abs(u[0]) < 0.05, f"Expected near-zero steering near ±π, got delta={u[0]:.4f}"


@_needs_acados
def test_mpc_defaults_match_yaml():
    model = BicycleModel()
    mpc = MPCController(model)
    assert mpc.Q[0, 0] == 7.0
    assert mpc.Q[2, 2] == 5.0
    assert mpc.Q[3, 3] == 9.0
    assert mpc.R[0, 0] == 8.0
    assert mpc.R[1, 1] == 0.3
