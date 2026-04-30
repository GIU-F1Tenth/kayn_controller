---
title: KAYN Controller — Repo Cleanup & Tuning Reference
date: 2026-04-30
status: approved
---

## Goal

Fix a dead-param wiring bug, clean up four code-quality issues, and rewrite the README
configuration section into a complete per-parameter tuning reference.

No restructuring. No new features.

---

## Section 1 — Param wiring

### Problem

`kayn_params.yaml` declares four parameters that the ROS node never reads:

| Yaml key | Hardcoded in | Current value |
|---|---|---|
| `fsm.enter_threshold` | `curvature.py:ENTER_THRESHOLD` | 0.10 |
| `fsm.exit_threshold` | `curvature.py:EXIT_THRESHOLD` | 0.06 |
| `fsm.confirm_steps` | `fsm.py:CONFIRM_STEPS` | 3 |
| `fsm.blend_window` | `fsm.py:BLEND_WINDOW` | 5 |

Editing the yaml has no effect on these values.

### Fix

Move the four values from module-level constants into constructor arguments.
Keep the current values as argument defaults so simulation and tests that omit them continue to work.

**`CurvatureEstimator.__init__`** gains `enter_threshold` and `exit_threshold` args.
**`FSM.__init__`** gains `confirm_steps` and `blend_window` args.
**`kayn_node._load_params`** reads all four from params and stores them.
**`kayn_node._build_controllers`** passes all four through to the constructors.

Module-level constants (`ENTER_THRESHOLD`, `EXIT_THRESHOLD`, `CONFIRM_STEPS`, `BLEND_WINDOW`)
stay in place as the default values — no change to simulation or test call sites.

---

## Section 2 — Code cleanup

Four targeted changes only. No restructuring.

### 2a — FSM transition logging

**Current:** `print(f"[KAYN] {self.state.name} → ...")` inside `fsm.py`

**Fix:** Add `log_fn` parameter to `FSM.__init__` (default: `print`).
The ROS node passes `self.get_logger().info`. Simulation and tests pass nothing.
No rclpy import inside fsm.py.

### 2b — kayn_node param loading

**Current:** `g = lambda n: self.get_parameter(n).value` — lambda assigned to variable.

**Fix:** Inline the calls directly. Readable, no clever shorthand.
Also wire the four previously-dead param reads (Section 1).

### 2c — state_handoff isinstance check

**Current:** `type(incoming_controller).__name__ == 'MPCController'` — string comparison on type name.

**Fix:** `isinstance(incoming_controller, MPCController)` — import MPCController at top of file.

### 2d — kayn_params.yaml default values

Restore the intended defaults that were overwritten:

| Key | Current (broken) | Correct |
|---|---|---|
| `fsm.straight_controller` | `"stanley"` | `"lqr"` |
| `fsm.curve_controller` | `"stanley"` | `"mpc"` |
| `fsm.fallback_controller` | `"stanley"` | `"stanley"` (unchanged) |
| `max_steering` | `0.8` | `0.4189` (matches `DELTA_MAX` in bicycle_model.py = 24°) |

The `dt: 0.005` / `control_hz: 200.0` combination is internally consistent (200 Hz × 0.005 s = 1),
so leave it. Add a comment making the relationship explicit.

---

## Section 3 — README parameter reference

Replace the current shallow "Configuration" section with a structured per-parameter
reference organized by subsystem. Each entry: what it is, units, default, one-line tuning guidance.

### Structure

**Vehicle**
- `wheelbase` [m] — kinematic model parameter; must match physical car
- `dt` [s] — integration timestep; must equal `1 / control_hz`
- `control_hz` [Hz] — control loop rate

**FSM supervisor — controller slots**
- `fsm.warmup_controller` — controller used during WARMUP (valid: `stanley` | `lqr` | `mpc`)
- `fsm.straight_controller` — controller used in STRAIGHT
- `fsm.curve_controller` — controller used in CURVE
- `fsm.fallback_controller` — controller used in FALLBACK

**FSM supervisor — timing**
- `fsm.warmup_steps` — steps before leaving WARMUP; at 50 Hz, 50 steps = 1 s
- `fsm.confirm_steps` — consecutive samples needed before any state transition; raise to reduce false triggers
- `fsm.blend_window` — steps to linearly interpolate between controllers at transitions; raise for smoother handoff, lower for faster response

**Curvature detection**
- `fsm.enter_threshold` [rad/m] — κ above which CURVE is entered; default 0.10 = R < 10 m
- `fsm.exit_threshold` [rad/m] — κ below which STRAIGHT is re-entered; default 0.06 = R > 16.7 m
- `fsm.lookahead` — waypoints ahead used for Menger curvature estimate; raise for earlier detection

Note on hysteresis: `exit_threshold` must be strictly less than `enter_threshold`.
Closing the gap (e.g., both 0.08) causes rapid oscillation at curve boundaries.

**Stanley**
- `stanley.k` — cross-track gain; raise for tighter tracking, lower if oscillating at speed

**LQR — state cost Q = diag([q_px, q_py, q_theta, q_v])**
- `lqr.q_px`, `lqr.q_py` — position error penalty; raise both to track the path more aggressively
- `lqr.q_theta` — heading error penalty; raise if the car wanders angularly on straights
- `lqr.q_v` — speed error penalty; raise to track reference speed more closely

**LQR — control cost R = diag([r_delta, r_a])**
- `lqr.r_delta` — steering effort penalty; raise to damp steering oscillations on straights
- `lqr.r_a` — acceleration effort penalty; raise for smoother acceleration profile

**MPC — cost (same Q/R structure as LQR)**
- `mpc.q_px`, `mpc.q_py`, `mpc.q_theta`, `mpc.q_v` — same semantics as LQR
- `mpc.r_delta` — raise (e.g., 8.0) to smooth MPC steering on tight curves
- `mpc.r_a` — raise for smoother acceleration through corners

**MPC — solver**
- `mpc.horizon_n` — prediction steps; longer horizon sees further ahead but increases solve time
- `mpc.timeout_ms` — solver budget [ms]; if exceeded, FSM falls back to `fallback_controller`

**Limits**
- `max_speed` [m/s] — hard speed cap applied to trajectory speed setpoint
- `max_steering` [rad] — must match `DELTA_MAX` in `bicycle_model.py` (0.4189 rad = 24°)
- `max_accel` [m/s²] — hard acceleration cap

### Common recipes

| Goal | Change |
|---|---|
| Run pure LQR everywhere | `curve_controller: "lqr"` |
| Run pure Stanley | `straight_controller: "stanley"`, `curve_controller: "stanley"` |
| Skip warmup | `fsm.warmup_steps: 0` |
| Detect curves earlier | lower `fsm.enter_threshold` (e.g., `0.07`) |
| Smoother MPC steering | raise `mpc.r_delta` (e.g., `8.0`) |
| Tighter straight tracking | raise `lqr.q_py` (e.g., `10.0`) |
| Prevent MPC fallback at high speed | raise `mpc.timeout_ms` (e.g., `10.0`) |

---

## Files changed

| File | Change |
|---|---|
| `config/kayn_params.yaml` | Restore defaults, fix comments, add tuning hints |
| `kayn_controller/kayn_node.py` | Wire 4 dead params, remove lambda shorthand |
| `kayn_controller/supervisor/fsm.py` | Accept `confirm_steps`, `blend_window`, `log_fn` as args |
| `kayn_controller/supervisor/curvature.py` | Accept `enter_threshold`, `exit_threshold` as args |
| `kayn_controller/supervisor/state_handoff.py` | Use `isinstance` instead of type name string |
| `README.md` | Rewrite Configuration section with full param reference |

---

## Out of scope

- No changes to controllers (`lqr.py`, `mpc.py`, `stanley.py`, `bicycle_model.py`)
- No changes to simulation (`sim.py`, `track.py`, `plot.py`)
- No changes to tests
- No changes to `math/` derivation docs
- No new files
