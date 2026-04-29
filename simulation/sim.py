"""
KAYN multi-scenario closed-loop simulation.
Zero ROS2 imports.  Run with: python3 simulation/sim.py

Scenarios
---------
  straight_50m   : 50 m straight — baseline tracking test
  curve_90deg    : single 90-degree left corner
  hairpin        : straight → 180° hairpin → straight back
  slalom_4gates  : four alternating left-right 90° corners
  chicane        : two left corners with straights (the original mixed track)
  oval           : two 180° semicircles joined by straights

Controllers tested per scenario
--------------------------------
  stanley   : Stanley controller (steering) + proportional speed
  lqr       : LQR (steering + acceleration, tracks closest waypoint)
  mpc       : MPC via acados RTI (steering + acceleration, predictive)
  kayn_fsm  : Full KAYN FSM — LQR on straights, MPC on curves, Stanley fallback

Outputs
-------
  simulation/results/<scenario>_trajectory.png
  simulation/results/<scenario>_cte.png
  simulation/results/summary_cte.png
  (console) metrics table per scenario × controller
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from typing import List, Dict, Optional

from kayn_controller.controllers.bicycle_model import BicycleModel, A_MAX
from kayn_controller.controllers.lqr import LQRController
from kayn_controller.controllers.mpc import MPCController
from kayn_controller.controllers.stanley import StanleyController
from kayn_controller.supervisor.curvature import CurvatureEstimator
from kayn_controller.supervisor.fsm import FSM
from simulation.track import (
    straight_track, curve_track, mixed_track,
    hairpin_track, slalom_track, oval_track,
)


# ──────────────────────────────────────────────────────────────────────────────
# Safety limits
# ──────────────────────────────────────────────────────────────────────────────

MAX_CTE_ABORT = 4.0    # m   — abort if lateral error exceeds this
STUCK_STEPS   = 400    # sim steps without ref_idx advancing → stuck
V_KP          = 2.0    # proportional gain for Stanley speed control


# ──────────────────────────────────────────────────────────────────────────────
# Scenario registry
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS: Dict[str, callable] = {
    'straight_50m':  lambda: straight_track(length=50.0, v_ref=3.0, n_points=300),
    'curve_90deg':   lambda: curve_track(radius=4.0, sweep_deg=90.0, v_ref=2.0,
                                         n_points=120, direction=1),
    'hairpin':       lambda: hairpin_track(straight_len=20.0, radius=3.0,
                                           v_ref_straight=3.0, v_ref_curve=1.5),
    'slalom_4gates': lambda: slalom_track(n_gates=4, radius=3.5),
    'chicane':       lambda: mixed_track(),
    'oval':          lambda: oval_track(straight_len=30.0, radius=8.0),
}

CONTROLLERS = ['stanley', 'lqr', 'mpc', 'kayn_fsm']


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _closest_idx(x_curr: np.ndarray, trajectory: List[Dict]) -> int:
    pts = np.array([[wp['x'], wp['y']] for wp in trajectory])
    return int(np.argmin(np.linalg.norm(pts - x_curr[:2], axis=1)))


def _reset_mpc(mpc: MPCController, x0: np.ndarray, track: List[Dict]) -> None:
    """Warm-start MPC from initial state x0 along the time-advance reference."""
    try:
        ref_ta = mpc._time_advance_ref(track, mpc.N, mpc.dt)
        mpc.solver.set(0, 'lbx', x0)
        mpc.solver.set(0, 'ubx', x0)
        mpc.solver.set(0, 'x', x0)
        for k in range(1, mpc.N + 1):
            wp = ref_ta[min(k, len(ref_ta) - 1)]
            mpc.solver.set(k, 'x', np.array([wp['x'], wp['y'], wp['theta'], wp['v']]))
        for k in range(mpc.N):
            mpc.solver.set(k, 'u', np.zeros(2))
            wp = ref_ta[k]
            mpc.solver.set(k, 'yref',
                           np.array([wp['x'], wp['y'], wp['theta'], wp['v'], 0.0, 0.0]))
        wp_e = ref_ta[mpc.N]
        mpc.solver.set(mpc.N, 'yref',
                       np.array([wp_e['x'], wp_e['y'], wp_e['theta'], wp_e['v']]))
    except Exception:
        pass  # non-fatal — solver will self-correct


def _make_fsm(lqr: LQRController, mpc: MPCController,
              stanley: StanleyController) -> FSM:
    return FSM(
        lqr=lqr,
        mpc=mpc,
        stanley=stanley,
        curvature_estimator=CurvatureEstimator(lookahead=10),
    )


def _compute_u(ctrl_name: str,
               x_curr: np.ndarray,
               trajectory: List[Dict],
               ref_idx: int,
               lqr: LQRController,
               mpc: MPCController,
               stanley: StanleyController,
               fsm: Optional[FSM]):
    """Run one controller step.  Returns (u: [delta, a], mode_label: str)."""
    if ctrl_name == 'kayn_fsm':
        u = fsm.step(x_curr, trajectory, ref_idx)
        return u, fsm.state_name

    if ctrl_name == 'lqr':
        wp = trajectory[min(ref_idx, len(trajectory) - 1)]
        x_ref = np.array([wp['x'], wp['y'], wp['theta'], wp['v']])
        u = lqr.compute_control(x_curr, x_ref)
        return u, 'LQR'

    if ctrl_name == 'mpc':
        # Pass full remaining track — compute_control builds time-advance ref internally
        ref_slice = trajectory[ref_idx:]
        if len(ref_slice) < 2:
            ref_slice = trajectory[-2:]
        u, _, status = mpc.compute_control(x_curr, ref_slice)
        return u, 'MPC' if status == 0 else 'MPC_FAIL'

    if ctrl_name == 'stanley':
        delta = stanley.compute_control(x_curr, trajectory)
        v_ref = trajectory[min(ref_idx, len(trajectory) - 1)]['v']
        a = float(np.clip(V_KP * (v_ref - x_curr[3]), -A_MAX, A_MAX))
        return np.array([delta, a]), 'STANLEY'

    raise ValueError(f"Unknown controller: {ctrl_name!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Core runner
# ──────────────────────────────────────────────────────────────────────────────

def run_scenario(scenario_name: str,
                 track: List[Dict],
                 ctrl_name: str,
                 lqr: LQRController,
                 mpc: MPCController,
                 stanley: StanleyController,
                 dt: float = 0.02,
                 v_init: float = 2.0,
                 max_t: float = 120.0) -> dict:
    """
    Simulate ctrl_name on track for up to max_t seconds.
    Returns a result dict ready for plotting and metrics.
    """
    model   = BicycleModel(dt=dt)
    curv_est = CurvatureEstimator(lookahead=10)

    x0 = np.array([track[0]['x'], track[0]['y'], track[0]['theta'], v_init])

    # Fresh FSM per run (holds per-run state counters)
    fsm = _make_fsm(lqr, mpc, stanley) if ctrl_name == 'kayn_fsm' else None

    # Reset MPC warm-start so prior scenario doesn't pollute this one
    if ctrl_name in ('mpc', 'kayn_fsm'):
        _reset_mpc(mpc, x0, track)

    x_curr = x0.copy()

    xs, ys, thetas, vs = [], [], [], []
    deltas, accels     = [], []
    ctes, kappas       = [], []
    modes, times       = [], []

    t            = 0.0
    max_steps    = int(max_t / dt)
    prev_ref_idx = 0
    stuck_count  = 0
    abort_reason = 'timeout'
    ref_idx      = 0

    for _ in range(max_steps):
        ref_idx = _closest_idx(x_curr, track)

        if ref_idx >= len(track) - 1:
            abort_reason = 'completed'
            break

        # Stuck check
        if ref_idx <= prev_ref_idx:
            stuck_count += 1
            if stuck_count > STUCK_STEPS:
                abort_reason = 'stuck'
                break
        else:
            stuck_count  = 0
            prev_ref_idx = ref_idx

        # Controller step
        try:
            u, mode = _compute_u(ctrl_name, x_curr, track, ref_idx,
                                  lqr, mpc, stanley, fsm)
        except Exception as exc:
            print(f'    [WARN] {ctrl_name} step failed at t={t:.2f}s: {exc}')
            u    = np.zeros(2)
            mode = 'ERROR'

        # Lateral error (signed, perp to track heading)
        wp   = track[ref_idx]
        perp = np.array([-np.sin(wp['theta']), np.cos(wp['theta'])])
        cte  = float(np.dot(x_curr[:2] - np.array([wp['x'], wp['y']]), perp))
        kappa = curv_est.estimate(track, ref_idx)

        if abs(cte) > MAX_CTE_ABORT:
            abort_reason = f'off_track cte={cte:.2f}m'
            break

        xs.append(x_curr[0]);  ys.append(x_curr[1])
        thetas.append(x_curr[2]); vs.append(x_curr[3])
        deltas.append(u[0]);   accels.append(u[1])
        ctes.append(cte);      kappas.append(kappa)
        modes.append(mode);    times.append(t)

        x_curr = model.step_rk4(x_curr, u)
        t += dt

    completion = min(ref_idx, len(track) - 1) / max(len(track) - 1, 1)

    return {
        'scenario':   scenario_name,
        'controller': ctrl_name,
        'track':      track,
        'x':          np.array(xs),      'y':      np.array(ys),
        'theta':      np.array(thetas),  'v':      np.array(vs),
        'delta':      np.array(deltas),  'accel':  np.array(accels),
        'cte':        np.array(ctes),    'kappa':  np.array(kappas),
        'mode':       modes,             'time':   np.array(times),
        'completion': completion,
        'abort':      abort_reason,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ──────────────────────────────────────────────────────────────────────────────

def _metrics(r: dict) -> dict:
    ctes = r['cte']
    if len(ctes) == 0:
        return {'max_cte': float('nan'), 'rms_cte': float('nan'),
                'avg_v': float('nan'), 'steps': 0}
    return {
        'max_cte': float(np.max(np.abs(ctes))),
        'rms_cte': float(np.sqrt(np.mean(ctes ** 2))),
        'avg_v':   float(np.mean(r['v'])),
        'steps':   len(ctes),
    }


def print_summary(all_results: dict) -> None:
    print()
    print('═' * 88)
    print(f"{'Scenario':<18} {'Controller':<12} {'Compl%':>7} "
          f"{'MaxCTE m':>9} {'RMS_CTE m':>9} {'AvgV m/s':>9} {'Abort':<22}")
    print('─' * 88)
    for scenario_name, ctrl_results in all_results.items():
        for ctrl_name, r in ctrl_results.items():
            m = _metrics(r)
            print(f"{scenario_name:<18} {ctrl_name:<12} "
                  f"{r['completion']*100:7.1f} "
                  f"{m['max_cte']:9.3f} "
                  f"{m['rms_cte']:9.3f} "
                  f"{m['avg_v']:9.2f}  "
                  f"{r['abort']:<22}")
        print('─' * 88)
    print('═' * 88)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('KAYN Multi-Scenario Simulation')
    print('Scenarios :', ', '.join(SCENARIOS))
    print('Controllers:', ', '.join(CONTROLLERS))
    print('=' * 70)

    dt    = 0.02
    model = BicycleModel(dt=dt)

    print('\nBuilding controllers (acados MPC compilation — takes ~15 s first run)...')
    lqr     = LQRController(model)
    mpc     = MPCController(model, dt=dt)
    stanley = StanleyController(model=model)
    print('Controllers ready.\n')

    all_results: Dict[str, Dict[str, dict]] = {}

    for scenario_name, track_fn in SCENARIOS.items():
        track = track_fn()
        print(f"\n{'─'*66}")
        print(f"Scenario: {scenario_name}  ({len(track)} waypoints, "
              f"{track[-1]['x']:.1f} m × {track[-1]['y']:.1f} m)")
        print(f"{'─'*66}")
        all_results[scenario_name] = {}

        for ctrl_name in CONTROLLERS:
            t_wall = time.perf_counter()
            result = run_scenario(
                scenario_name, track, ctrl_name,
                lqr, mpc, stanley,
                dt=dt, v_init=2.0, max_t=120.0,
            )
            elapsed = time.perf_counter() - t_wall
            m = _metrics(result)

            print(f"  [{ctrl_name:<10}] "
                  f"compl={result['completion']*100:5.1f}%  "
                  f"max_cte={m['max_cte']:6.3f}m  "
                  f"rms_cte={m['rms_cte']:6.3f}m  "
                  f"abort={result['abort']:<22}  "
                  f"wall={elapsed:.1f}s")

            all_results[scenario_name][ctrl_name] = result

    print_summary(all_results)

    from simulation.plot import plot_all_results
    plot_all_results(all_results)
    print('\nDone. Plots saved in simulation/results/')


if __name__ == '__main__':
    main()
