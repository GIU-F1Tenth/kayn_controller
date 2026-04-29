"""Plotting for KAYN simulation results — single-run and multi-scenario."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Per-FSM-state colours (used when plotting FSM run)
MODE_COLORS = {
    'WARMUP':   '#10B981',
    'STRAIGHT': '#4C9BE8',
    'BLEND_OUT':'#A78BFA',
    'CURVE':    '#EF4444',
    'BLEND_IN': '#F97316',
    'FALLBACK': '#22C55E',
    # standalone labels
    'LQR':      '#4C9BE8',
    'MPC':      '#EF4444',
    'MPC_FAIL': '#FF8800',
    'STANLEY':  '#10B981',
    'ERROR':    '#888888',
}

# Per-controller line colours for multi-controller overlay plots
CTRL_COLORS = {
    'stanley':  '#10B981',
    'lqr':      '#4C9BE8',
    'mpc':      '#EF4444',
    'kayn_fsm': '#7C3AED',
}

CTRL_LABELS = {
    'stanley':  'Stanley',
    'lqr':      'LQR',
    'mpc':      'MPC',
    'kayn_fsm': 'KAYN FSM',
}


def _savefig(fig, name: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy single-run plots (kept for backward compat)
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectory(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    track = data['track']
    ax.plot([wp['x'] for wp in track], [wp['y'] for wp in track],
            'k--', linewidth=1, alpha=0.4, label='Reference')
    for mode, color in MODE_COLORS.items():
        mask = np.array([m == mode for m in data['mode']])
        if mask.any():
            ax.scatter(data['x'][mask], data['y'][mask],
                       c=color, s=4, label=mode)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Trajectory (coloured by active mode)')
    ax.legend(loc='upper right', markerscale=3)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, 'trajectory.png')


def plot_cross_track_error(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    times, modes = data['time'], data['mode']
    prev_mode = None
    for i, (t, cte, mode) in enumerate(zip(times, data['cte'], modes)):
        color = MODE_COLORS.get(mode, 'gray')
        if i > 0:
            ax.plot([times[i - 1], t], [data['cte'][i - 1], cte],
                    color=color, linewidth=1)
        if mode != prev_mode and i > 0:
            ax.axvline(t, color='gray', linewidth=0.5, alpha=0.5)
        prev_mode = mode
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline( 0.05, color='red', linewidth=0.8, linestyle='--', label='±0.05 m')
    ax.axhline(-0.05, color='red', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('CTE [m]')
    ax.set_title('Cross-Track Error  (vertical lines = mode transitions)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, 'cross_track_error.png')


def plot_control_inputs(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    times = data['time']
    ax.plot(times, np.degrees(data['delta']),
            label='Steering [deg]', color='steelblue')
    ax2 = ax.twinx()
    ax2.plot(times, data['accel'],
             label='Accel [m/s²]', color='darkorange', alpha=0.7)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Steering [deg]')
    ax2.set_ylabel('Acceleration [m/s²]')
    ax.set_title('Control Inputs'); ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2)
    fig.tight_layout()
    _savefig(fig, 'control_inputs.png')


def plot_curvature_fsm(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    times = data['time']
    ax.plot(times, data['kappa'], color='purple', linewidth=1, label='κ [rad/m]')
    ax.axhline(0.10, color='red',    linewidth=0.8, linestyle='--', label='Enter MPC (0.10)')
    ax.axhline(0.06, color='orange', linewidth=0.8, linestyle='--', label='Exit MPC (0.06)')
    state_to_num = {'WARMUP': 0, 'STRAIGHT': 1, 'BLEND_OUT': 2,
                    'CURVE': 3, 'BLEND_IN': 4, 'FALLBACK': 5}
    ax2 = ax.twinx()
    ax2.step(times, [state_to_num.get(m, 1) for m in data['mode']],
             color='gray', alpha=0.4, linewidth=1, where='post')
    ax2.set_yticks(list(state_to_num.values()))
    ax2.set_yticklabels(list(state_to_num.keys()), fontsize=7)
    ax2.set_ylabel('FSM State')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Curvature κ [rad/m]')
    ax.set_title('Curvature Estimate + FSM State')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, 'curvature_fsm.png')


def plot_results(data: dict) -> None:
    """Original single-run plot entrypoint (backward compat)."""
    plot_trajectory(data)
    plot_cross_track_error(data)
    plot_control_inputs(data)
    plot_curvature_fsm(data)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-scenario plots
# ──────────────────────────────────────────────────────────────────────────────

def _ctrl_legend_handles():
    return [mlines.Line2D([], [], color=CTRL_COLORS[c], linewidth=2,
                          label=CTRL_LABELS[c])
            for c in CTRL_COLORS]


def plot_scenario_trajectory(scenario_name: str,
                              ctrl_results: dict) -> None:
    """Overlay trajectory of all controllers on one axes."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Reference track (use first available result's track)
    track = next(iter(ctrl_results.values()))['track']
    ax.plot([wp['x'] for wp in track], [wp['y'] for wp in track],
            'k--', linewidth=1.2, alpha=0.4, label='Reference', zorder=1)

    for ctrl_name, r in ctrl_results.items():
        if len(r['x']) == 0:
            continue
        color = CTRL_COLORS.get(ctrl_name, 'gray')
        ax.plot(r['x'], r['y'], color=color, linewidth=1.5,
                alpha=0.85, label=CTRL_LABELS.get(ctrl_name, ctrl_name))
        # Mark endpoint
        ax.plot(r['x'][-1], r['y'][-1], 'o', color=color,
                markersize=6, markeredgecolor='white', zorder=5)

    ax.set_title(f'Trajectory — {scenario_name}')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(loc='best', fontsize=9)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, f'{scenario_name}_trajectory.png')


def plot_scenario_cte(scenario_name: str, ctrl_results: dict) -> None:
    """CTE vs time for all controllers in one scenario."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for ctrl_name, r in ctrl_results.items():
        if len(r['time']) == 0:
            continue
        color = CTRL_COLORS.get(ctrl_name, 'gray')
        ax.plot(r['time'], r['cte'], color=color, linewidth=1.2, alpha=0.85,
                label=f"{CTRL_LABELS.get(ctrl_name, ctrl_name)} "
                      f"(rms={np.sqrt(np.mean(r['cte']**2)):.3f} m)")

    ax.axhline( 0.05, color='red', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axhline(-0.05, color='red', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('CTE [m]')
    ax.set_title(f'Cross-Track Error — {scenario_name}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, f'{scenario_name}_cte.png')


def plot_scenario_speed(scenario_name: str, ctrl_results: dict) -> None:
    """Speed profile for all controllers."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for ctrl_name, r in ctrl_results.items():
        if len(r['time']) == 0:
            continue
        color = CTRL_COLORS.get(ctrl_name, 'gray')
        ax.plot(r['time'], r['v'], color=color, linewidth=1.2, alpha=0.85,
                label=CTRL_LABELS.get(ctrl_name, ctrl_name))
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Speed [m/s]')
    ax.set_title(f'Speed Profile — {scenario_name}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, f'{scenario_name}_speed.png')


def plot_scenario_steering(scenario_name: str, ctrl_results: dict) -> None:
    """Steering angle for all controllers."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for ctrl_name, r in ctrl_results.items():
        if len(r['time']) == 0:
            continue
        color = CTRL_COLORS.get(ctrl_name, 'gray')
        ax.plot(r['time'], np.degrees(r['delta']), color=color,
                linewidth=1.2, alpha=0.85,
                label=CTRL_LABELS.get(ctrl_name, ctrl_name))
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Steering [deg]')
    ax.set_title(f'Steering — {scenario_name}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, f'{scenario_name}_steering.png')


def plot_fsm_states(scenario_name: str, ctrl_results: dict) -> None:
    """FSM state transitions for the kayn_fsm controller."""
    r = ctrl_results.get('kayn_fsm')
    if r is None or len(r['time']) == 0:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    ax1.plot(r['time'], r['kappa'], color='purple', linewidth=1, label='κ [rad/m]')
    ax1.axhline(0.10, color='red',    linewidth=0.8, linestyle='--', label='Enter MPC')
    ax1.axhline(0.06, color='orange', linewidth=0.8, linestyle='--', label='Exit MPC')
    ax1.set_ylabel('Curvature [rad/m]')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    state_map = {'WARMUP': 0, 'STRAIGHT': 1, 'BLEND_OUT': 2,
                 'CURVE': 3, 'BLEND_IN': 4, 'FALLBACK': 5}
    state_nums = [state_map.get(m, 1) for m in r['mode']]
    ax2.step(r['time'], state_nums, color='steelblue', linewidth=1.5, where='post')
    ax2.set_yticks(list(state_map.values()))
    ax2.set_yticklabels(list(state_map.keys()), fontsize=7)
    ax2.set_ylabel('FSM State')
    ax2.set_xlabel('Time [s]')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'KAYN FSM — {scenario_name}', fontsize=11)
    fig.tight_layout()
    _savefig(fig, f'{scenario_name}_fsm.png')


def plot_summary_bar(all_results: dict) -> None:
    """Bar chart: max CTE per scenario × controller."""
    scenarios   = list(all_results.keys())
    controllers = list(next(iter(all_results.values())).keys())

    x      = np.arange(len(scenarios))
    width  = 0.8 / len(controllers)
    offset = -(len(controllers) - 1) / 2 * width

    fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 2), 5))

    for i, ctrl_name in enumerate(controllers):
        max_ctes = []
        for sc in scenarios:
            r = all_results[sc].get(ctrl_name, {})
            ctes = r.get('cte', np.array([]))
            max_ctes.append(float(np.max(np.abs(ctes))) if len(ctes) > 0 else float('nan'))
        bars = ax.bar(x + offset + i * width, max_ctes, width * 0.9,
                      label=CTRL_LABELS.get(ctrl_name, ctrl_name),
                      color=CTRL_COLORS.get(ctrl_name, 'gray'), alpha=0.85)
        for bar, val in zip(bars, max_ctes):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=6)

    ax.axhline(0.05, color='red', linewidth=1, linestyle='--', alpha=0.6,
               label='±0.05 m target')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Max |CTE| [m]')
    ax.set_title('Max Cross-Track Error: All Scenarios × Controllers')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    _savefig(fig, 'summary_cte.png')


def plot_summary_completion(all_results: dict) -> None:
    """Bar chart: track completion % per scenario × controller."""
    scenarios   = list(all_results.keys())
    controllers = list(next(iter(all_results.values())).keys())

    x      = np.arange(len(scenarios))
    width  = 0.8 / len(controllers)
    offset = -(len(controllers) - 1) / 2 * width

    fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 2), 4))

    for i, ctrl_name in enumerate(controllers):
        compl = [all_results[sc].get(ctrl_name, {}).get('completion', 0) * 100
                 for sc in scenarios]
        ax.bar(x + offset + i * width, compl, width * 0.9,
               label=CTRL_LABELS.get(ctrl_name, ctrl_name),
               color=CTRL_COLORS.get(ctrl_name, 'gray'), alpha=0.85)

    ax.axhline(100, color='green', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Track Completion [%]')
    ax.set_title('Track Completion: All Scenarios × Controllers')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    _savefig(fig, 'summary_completion.png')


def plot_all_results(all_results: dict) -> None:
    """Generate all multi-scenario plots."""
    print('\nGenerating plots...')
    for scenario_name, ctrl_results in all_results.items():
        plot_scenario_trajectory(scenario_name, ctrl_results)
        plot_scenario_cte(scenario_name, ctrl_results)
        plot_scenario_speed(scenario_name, ctrl_results)
        plot_scenario_steering(scenario_name, ctrl_results)
        plot_fsm_states(scenario_name, ctrl_results)
    plot_summary_bar(all_results)
    plot_summary_completion(all_results)
