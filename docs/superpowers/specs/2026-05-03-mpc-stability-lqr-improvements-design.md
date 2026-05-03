# MPC Straight-Line Stability & LQR Improvements

**Date:** 2026-05-03  
**Status:** Approved

## Problem Statement

MPC exhibits two failure modes on straight sections:

- **(a) Steering oscillation / chattering** — the car weaves even with no lateral error
- **(c) Solver infeasibility** — acados RTI returns non-zero status, FSM drops to FALLBACK or blends a bad output

LQR has two secondary inefficiencies:
- Cache invalidates on every step of a straight (DARE re-solves unnecessarily)
- No feedforward steering in blend zones — purely reactive at curve entry/exit

## Root Cause Analysis

### 1. Heading wrap bug in MPC (`mpc.py`)

`compute_control` loads `wp['theta']` raw into `yref`. The quadratic cost `(theta - theta_ref)²` does not handle ±π wrap. If the vehicle heading is `+2.9 rad` and the stored reference is `-3.1 rad` (same geometric direction), the cost sees `(2.9 - (-3.1))² = 36` instead of ≈0. The solver commands large corrective steering, overshoots, and oscillates. Worst on any straight near heading ±π (e.g. the return leg of a hairpin or oval).

### 2. MPC status ignored during BLEND_IN (`fsm.py`)

`_step_blend_in` calls both MPC and LQR and interpolates, but discards the MPC `status` return. A bad warm-start carried over from a curve makes the RTI diverge on the new straight reference. The infeasible MPC output is silently blended into the drive command.

### 3. Code defaults out of sync with `kayn_params.yaml`

`sim.py` instantiates `MPCController(model, dt=dt)` and `LQRController(model)` without explicit Q/R, using code defaults. The defaults differ materially from the production-tuned yaml values — most critically `mpc.r_delta=4.0` in code vs `8.0` in yaml. Lower `r_delta` means the solver uses aggressive steering to minimize position/heading errors, amplifying the oscillation from root cause 1.

### 4. LQR cache invalidates on every step of a straight (`lqr.py`)

`_should_recompute` compares full `x_ref = [px, py, theta, v]`. On a straight, `px` advances every step → DARE re-solves every step. The DARE solution depends only on `theta` and `v` (the dynamics-relevant states) — `px` and `py` do not appear in the Jacobians `A` or `B`.

### 5. No feedforward for LQR in FSM (`fsm.py`)

`_ctrl_u` calls `lqr.compute_control(x_curr, x_ref)` with `u_ref=None` (zeros). LQR is purely feedback. During BLEND_IN (curve→straight), LQR is entering with curvature still present — zero feedforward steering means it must wait for the lateral error to build before it can react, increasing overshoot.

## Design

### Fix 1 — Heading reference normalization in `mpc.py:compute_control`

Unwrap `theta_ref` against the previous stage's reference heading so the reference sequence is monotone along the horizon. Each stage's theta is normalized against the preceding one (not just against `x_curr[2]`), which correctly handles curves that sweep through large heading changes.

```python
theta_prev = x_curr[2]
for k in range(self.N):
    wp = ref_ta[k]
    theta_ref = theta_prev + ((wp['theta'] - theta_prev + np.pi) % (2*np.pi) - np.pi)
    yref = np.array([wp['x'], wp['y'], theta_ref, wp['v'], 0.0, 0.0])
    self.solver.set(k, 'yref', yref)
    theta_prev = theta_ref

wp_e = ref_ta[self.N]
theta_ref_e = theta_prev + ((wp_e['theta'] - theta_prev + np.pi) % (2*np.pi) - np.pi)
self.solver.set(self.N, 'yref', np.array([wp_e['x'], wp_e['y'], theta_ref_e, wp_e['v']]))
```

**Scope:** `mpc.py:compute_control` only. No change to OCP formulation or solver setup.

### Fix 2 — Sync code defaults to yaml

**`mpc.py`** — update `MPCController.__init__` defaults:
- `Q = diag([7.0, 7.0, 5.0, 9.0])` (was `[5.0, 5.0, 6.0, 6.0]`)
- `R = diag([8.0, 0.3])` (was `[4.0, 0.5]`)

**`lqr.py`** — update `LQRController.__init__` defaults:
- `Q = diag([6.0, 6.0, 8.0, 2.0])` (was `[5.0, 5.0, 6.0, 1.0]`)
- `R = diag([4.0, 0.3])` (already matches)

The P_f comment in `mpc.py` must be kept consistent (`10 * Q`).

### Fix 3 — LQR cache key uses dynamics-relevant states only (`lqr.py`)

Replace the full-vector norm comparison in `_should_recompute` with a comparison over only `[theta, v]` (indices 2 and 3 of `x_ref`):

```python
dyn_curr  = x_ref[2:4]
dyn_cache = self._cached_x_ref[2:4]
if np.linalg.norm(dyn_curr - dyn_cache) > self._cache_tol:
    return True
```

Full `x_ref` is still stored in `_cached_x_ref` (no structural change). The comparison simply ignores `px` and `py`.

### Fix 4 — Curvature feedforward for LQR in `fsm.py:_ctrl_u`

Compute the Ackermann feedforward steering angle from the current curvature estimate and pass as `u_ref` to LQR:

```python
elif name == 'lqr':
    kappa = self.curv_est.estimate(trajectory, ref_idx)
    delta_ff = float(np.arctan(self.lqr.model.L * kappa))
    u = self.lqr.compute_control(
        x_curr,
        self._ref_state(trajectory, ref_idx),
        np.array([delta_ff, 0.0])
    )
    return u, 0.0, 0
```

`delta_ff = atan(L × κ)` is the kinematic Ackermann angle for the current radius. On a straight `κ ≈ 0` so `delta_ff ≈ 0` — no change to straight-line behaviour. At curve entry (blend zones) it seeds LQR with the correct steering bias before lateral error builds.

### Fix 5 — BLEND_IN infeasibility guard in `fsm.py:_step_blend_in`

Check MPC status during BLEND_IN. On infeasibility, fast-forward the blend counter and return pure LQR output immediately:

```python
def _step_blend_in(self, x_curr, trajectory, ref_idx) -> np.ndarray:
    alpha = self._blend_step / self._blend_window
    u_out, solve_time, status = self._ctrl_u(self._curve_ctrl, x_curr, trajectory, ref_idx)
    u_in, _, _ = self._ctrl_u(self._straight_ctrl, x_curr, trajectory, ref_idx)

    if self._curve_ctrl == 'mpc' and (status != 0 or solve_time > self._mpc_timeout_s):
        self._blend_step = self._blend_window   # skip remaining blend
        return u_in

    u = (1 - alpha) * u_out + alpha * u_in
    self._blend_step += 1
    if self._blend_step >= self._blend_window:
        self._transition(KAYNState.STRAIGHT, "blend_complete", x_curr, trajectory, ref_idx)
    return u
```

The transition to STRAIGHT is not triggered on the infeasibility path — the next call to `_step_blend_in` will see `_blend_step == _blend_window` and trigger it naturally on the following step. This avoids double-transitioning.

## Files Changed

| File | Change |
|------|--------|
| `kayn_controller/controllers/mpc.py` | Fix 1 (heading norm), Fix 2 (defaults) |
| `kayn_controller/controllers/lqr.py` | Fix 2 (defaults), Fix 3 (cache key) |
| `kayn_controller/supervisor/fsm.py` | Fix 4 (LQR feedforward), Fix 5 (BLEND_IN guard) |

No changes to `bicycle_model.py`, `curvature.py`, `state_handoff.py`, `kayn_node.py`, `sim.py`, or `kayn_params.yaml`.

## Testing

Existing tests must continue to pass:
- `test_mpc.py::test_acados_matches_intuition_on_straight` — steering near zero on straight
- `test_mpc.py::test_feasible_on_curve` — status=0 on curve
- `test_lqr.py::test_straight_line_convergence` — lateral error < 0.05m in 3s
- `test_lqr.py::test_gain_caching` — K reused at same ref

New tests to add:
- `test_mpc.py::test_heading_normalization_near_pi` — MPC near-zero steering when heading ≈ ±π on a straight
- `test_lqr.py::test_cache_hit_on_straight` — K NOT recomputed when only px advances
- `test_fsm.py::test_blend_in_mpc_infeasible_uses_lqr` — BLEND_IN returns LQR output when MPC status≠0
