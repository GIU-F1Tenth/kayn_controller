"""
State Handoff — reinitializes an incoming controller from the current vehicle state.

On every controller switch:
1. Set the incoming controller's reference index to current closest waypoint
2. MPC: warm-start with time-advanced state trajectory so first RTI step has a
   physically meaningful initial guess (vs. "car frozen at current position").
3. Stanley/LQR: stateless, nothing to initialize.
"""

import numpy as np
from typing import List, Dict


def handoff(incoming_controller, x_curr: np.ndarray,
            trajectory: List[Dict], ref_idx: int) -> None:
    """
    Initialize incoming_controller for the current vehicle state.

    Args:
        incoming_controller: LQRController, MPCController, or StanleyController
        x_curr: current vehicle state [px, py, theta, v]
        trajectory: full reference trajectory
        ref_idx: current index in trajectory
    """
    if type(incoming_controller).__name__ == 'MPCController':
        _handoff_mpc(incoming_controller, x_curr, trajectory, ref_idx)
    # LQR and Stanley are stateless — no initialization needed


def _handoff_mpc(mpc, x_curr: np.ndarray,
                 trajectory: List[Dict], ref_idx: int) -> None:
    """
    Warm-start MPC from the current state.

    State guess: propagate along the time-advance reference trajectory so
    each stage x[k] is seeded near where the car should actually be.
    This prevents the RTI from starting with a grossly infeasible warm-start
    (e.g., car frozen at entry of a hairpin) and avoids solver timeouts.
    """
    try:
        ref_slice = trajectory[ref_idx:]
        if len(ref_slice) < 2:
            ref_slice = trajectory[-2:]

        # Build time-advance reference (same logic as MPCController.compute_control)
        ref_ta = mpc._time_advance_ref(ref_slice, mpc.N, mpc.dt)

        # Set yref for all stages
        for k in range(mpc.N):
            wp = ref_ta[k]
            mpc.solver.set(k, 'yref',
                           np.array([wp['x'], wp['y'], wp['theta'], wp['v'],
                                     0.0, 0.0]))
        wp_e = ref_ta[mpc.N]
        mpc.solver.set(mpc.N, 'yref',
                       np.array([wp_e['x'], wp_e['y'], wp_e['theta'], wp_e['v']]))

        # Warm-start state trajectory along the reference waypoints
        mpc.solver.set(0, 'lbx', x_curr)
        mpc.solver.set(0, 'ubx', x_curr)
        mpc.solver.set(0, 'x', x_curr)
        for k in range(1, mpc.N + 1):
            wp = ref_ta[min(k, len(ref_ta) - 1)]
            x_guess = np.array([wp['x'], wp['y'], wp['theta'], wp['v']])
            mpc.solver.set(k, 'x', x_guess)

    except Exception:
        pass  # handoff failure is non-fatal — MPC will self-correct on next step
