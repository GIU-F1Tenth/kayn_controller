import numpy as np
import matplotlib.pyplot as plt
import os

MODE_COLORS = {
    'WARMUP':   '#10B981',
    'STRAIGHT':  '#4C9BE8',
    'BLEND_OUT': '#A78BFA',
    'CURVE':     '#EF4444',
    'BLEND_IN':  '#F97316',
    'FALLBACK':  '#22C55E',
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def _savefig(fig, name: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


def plot_trajectory(data: dict):
    fig, ax = plt.subplots(figsize=(10, 7))
    track = data['track']
    tx = [wp['x'] for wp in track]
    ty = [wp['y'] for wp in track]
    ax.plot(tx, ty, 'k--', linewidth=1, alpha=0.5, label='Reference')
    for mode, color in MODE_COLORS.items():
        mask = np.array([m == mode for m in data['mode']])
        if mask.any():
            ax.scatter(data['x'][mask], data['y'][mask], c=color, s=4, label=mode)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Trajectory (coloured by active mode)')
    ax.legend(loc='upper right', markerscale=3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, 'trajectory.png')


def plot_cross_track_error(data: dict):
    fig, ax = plt.subplots(figsize=(12, 4))
    times = data['time']
    modes = data['mode']
    prev_mode = None
    for i, (t, cte, mode) in enumerate(zip(times, data['cte'], modes)):
        color = MODE_COLORS.get(mode, 'gray')
        if i > 0:
            ax.plot([times[i - 1], t], [data['cte'][i - 1], cte], color=color, linewidth=1)
        if mode != prev_mode and i > 0:
            ax.axvline(t, color='gray', linewidth=0.5, alpha=0.5)
        prev_mode = mode
    ax.axhline(0,      color='black', linewidth=0.5)
    ax.axhline( 0.05,  color='red', linewidth=0.8, linestyle='--', label='±0.05 m')
    ax.axhline(-0.05,  color='red', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('CTE [m]')
    ax.set_title('Cross-Track Error  (vertical lines = mode transitions)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, 'cross_track_error.png')


def plot_control_inputs(data: dict):
    fig, ax = plt.subplots(figsize=(12, 4))
    times = data['time']
    ax.plot(times, np.degrees(data['delta']), label='Steering [deg]', color='steelblue')
    ax2 = ax.twinx()
    ax2.plot(times, data['accel'], label='Accel [m/s²]', color='darkorange', alpha=0.7)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Steering [deg]')
    ax2.set_ylabel('Acceleration [m/s²]')
    ax.set_title('Control Inputs')
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2)
    fig.tight_layout()
    _savefig(fig, 'control_inputs.png')


def plot_curvature_fsm(data: dict):
    fig, ax = plt.subplots(figsize=(12, 4))
    times = data['time']
    ax.plot(times, data['kappa'], color='purple', linewidth=1, label='κ [rad/m]')
    ax.axhline(0.10, color='red',    linewidth=0.8, linestyle='--', label='Enter MPC (0.10)')
    ax.axhline(0.06, color='orange', linewidth=0.8, linestyle='--', label='Exit MPC (0.06)')
    state_to_num = {'WARMUP': 0, 'STRAIGHT': 1, 'BLEND_OUT': 2,
                    'CURVE': 3, 'BLEND_IN': 4, 'FALLBACK': 5}
    state_nums = [state_to_num.get(m, 1) for m in data['mode']]
    ax2 = ax.twinx()
    ax2.step(times, state_nums, color='gray', alpha=0.4, linewidth=1, where='post')
    ax2.set_yticks(list(state_to_num.values()))
    ax2.set_yticklabels(list(state_to_num.keys()), fontsize=7)
    ax2.set_ylabel('FSM State')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Curvature κ [rad/m]')
    ax.set_title('Curvature Estimate + FSM State')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, 'curvature_fsm.png')


def plot_results(data: dict):
    plot_trajectory(data)
    plot_cross_track_error(data)
    plot_control_inputs(data)
    plot_curvature_fsm(data)
