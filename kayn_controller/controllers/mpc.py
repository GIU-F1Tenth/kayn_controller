"""
MPC Controller using acados — see math/mpc.md for full derivation.

OCP:
    min   sum_{k=0}^{N-1} (x_k-x_r_k)^T Q (x_k-x_r_k) + u_k^T R u_k
        + (x_N-x_r_N)^T P_f (x_N-x_r_N)
    s.t.  x_{k+1} = f_rk4(x_k, u_k)
          |delta| <= 0.4189, |a| <= 5.0, v >= 0

Solved via acados RTI (one SQP step per control call, warm-started).
"""

import numpy as np
import time
import os
import tempfile
from typing import List, Dict, Tuple

# Pre-load acados shared libs with global visibility before acados_template
# tries to dlopen them.  Without this, libhpipm.so is not found unless
# /tmp/acados/lib is already in LD_LIBRARY_PATH (e.g. via the ROS env).
import ctypes as _ctypes
_ACADOS_LIB_DIR = '/tmp/acados/lib'
if os.path.isdir(_ACADOS_LIB_DIR):
    for _so in ('libblasfeo.so', 'libhpipm.so', 'libacados.so'):
        _so_path = os.path.join(_ACADOS_LIB_DIR, _so)
        if os.path.isfile(_so_path):
            try:
                _ctypes.CDLL(_so_path, mode=_ctypes.RTLD_GLOBAL)
            except OSError:
                pass

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import MX, vertcat, cos, sin, tan

from .bicycle_model import BicycleModel, DELTA_MAX, A_MAX, V_MAX


class MPCController:
    def __init__(self, model: BicycleModel,
                 N: int = 15,
                 dt: float = None,
                 Q: np.ndarray = None,
                 R: np.ndarray = None,
                 v_max: float = V_MAX):
        self.model = model
        self.N = N
        self.dt = dt or model.dt

        # Q[3] = 6.0 — strong velocity tracking to prevent MPC from
        # over-speeding on curves (was 1.0, caused 2x v_ref overshoot and
        # 2.7m CTE on hairpin because car entered corners at 6 m/s vs 1.5 m/s ref).
        self.Q = Q if Q is not None else np.diag([5.0, 5.0, 6.0, 6.0])
        self.R = R if R is not None else np.diag([4.0, 0.5])
        self.P_f = 10.0 * self.Q  # terminal cost — heavier to prevent horizon-end drift
        self.v_max = v_max

        self.solver = self._build_solver()

    def _build_acados_model(self) -> AcadosModel:
        """Define CasADi symbolic bicycle model — the ODE is visible here, no abstraction."""
        acados_model = AcadosModel()
        acados_model.name = 'kayn_bicycle'

        # Symbolic states: [px, py, theta, v]
        px    = MX.sym('px')
        py    = MX.sym('py')
        theta = MX.sym('theta')
        v     = MX.sym('v')
        x     = vertcat(px, py, theta, v)

        # Symbolic controls: [delta, a]
        delta = MX.sym('delta')
        a     = MX.sym('a')
        u     = vertcat(delta, a)

        L = self.model.L

        # Continuous bicycle dynamics — same equations as bicycle_model.py
        # acados uses these symbolically to auto-compute Jacobians for RTI linearization
        f_expl = vertcat(
            v * cos(theta),       # px_dot
            v * sin(theta),       # py_dot
            v * tan(delta) / L,   # theta_dot
            a,                    # v_dot
        )

        acados_model.x = x
        acados_model.u = u
        acados_model.f_expl_expr = f_expl
        return acados_model

    def _build_solver(self) -> AcadosOcpSolver:
        """Build the acados OCP solver with HPIPM QP backend and RTI scheme."""
        ocp = AcadosOcp()
        ocp.model = self._build_acados_model()

        nx = 4  # [px, py, theta, v]
        nu = 2  # [delta, a]

        ocp.solver_options.N_horizon = self.N   # replaces deprecated .N

        # LINEAR_LS cost: y = [x; u], cost = (y - y_ref)^T W (y - y_ref)
        ocp.cost.cost_type   = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        # Selection matrices: y = Vx * x + Vu * u
        Vx = np.zeros((nx + nu, nx))
        Vx[:nx, :] = np.eye(nx)
        Vu = np.zeros((nx + nu, nu))
        Vu[nx:, :] = np.eye(nu)

        ocp.cost.Vx   = Vx
        ocp.cost.Vu   = Vu
        ocp.cost.Vx_e = np.eye(nx)

        # Weight matrices [Q block-diag R] and terminal P_f
        ocp.cost.W   = np.block([[self.Q, np.zeros((4, 2))],
                                  [np.zeros((2, 4)), self.R]])
        ocp.cost.W_e = self.P_f

        # Zero reference (overwritten at each compute_control call)
        ocp.cost.yref   = np.zeros(nx + nu)
        ocp.cost.yref_e = np.zeros(nx)

        # Control bounds: |delta| <= DELTA_MAX, |a| <= A_MAX
        ocp.constraints.lbu    = np.array([-DELTA_MAX, -A_MAX])
        ocp.constraints.ubu    = np.array([ DELTA_MAX,  A_MAX])
        ocp.constraints.idxbu  = np.array([0, 1])

        # State bounds: 0 <= v <= v_max [m/s] — taken from max_speed in kayn_params.yaml
        ocp.constraints.lbx    = np.array([0.0])
        ocp.constraints.ubx    = np.array([self.v_max])
        ocp.constraints.idxbx  = np.array([3])

        # Initial state constraint (set at each call)
        ocp.constraints.x0 = np.zeros(nx)

        # Solver: RTI = one SQP iteration per call, warm-started from previous solution
        ocp.solver_options.tf              = self.N * self.dt
        ocp.solver_options.integrator_type = 'ERK'        # explicit Runge-Kutta (RK4)
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'   # real-time iteration
        ocp.solver_options.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'  # interior-point QP
        ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
        ocp.solver_options.print_level     = 0

        # Build generated C code in temp dir to keep workspace clean
        build_dir = os.path.join(tempfile.gettempdir(), 'kayn_acados')
        os.makedirs(build_dir, exist_ok=True)
        ocp.code_gen_opts.code_export_directory = build_dir
        ocp.code_gen_opts.json_file = os.path.join(build_dir, 'kayn_ocp.json')  # explicit path

        solver = AcadosOcpSolver(ocp)
        return solver

    @staticmethod
    def _time_advance_ref(full_traj: List[Dict], N: int, dt: float) -> List[Dict]:
        """
        Return N+1 time-advanced reference waypoints.

        Instead of spatial indexing (ref[k] = closest[k]), this gives the
        waypoint the car *should* reach at time k*dt given the reference speed
        profile.  This is critical on tight curves where N spatial steps cover
        far less arc than the full curve — spatial indexing would see only the
        first few metres of a hairpin, time-advance sees all the way around it.
        """
        n = len(full_traj)
        if n < 2:
            return [full_traj[0]] * (N + 1)

        pts = np.array([[wp['x'], wp['y']] for wp in full_traj])
        diffs = np.diff(pts, axis=0)
        arc = np.concatenate([[0.0],
                               np.cumsum(np.linalg.norm(diffs, axis=1))])

        result: List[Dict] = []
        s = 0.0
        for _ in range(N + 1):
            idx = int(np.searchsorted(arc, s, side='left'))
            idx = min(idx, n - 1)
            wp = full_traj[idx]
            result.append(wp)
            s += wp['v'] * dt   # advance s by v_ref * dt

        return result

    def compute_control(self, x_curr: np.ndarray,
                        ref_traj: List[Dict]) -> Tuple[np.ndarray, float, int]:
        """
        Run one RTI step.

        Args:
            x_curr:   current state [px, py, theta, v]
            ref_traj: remaining trajectory from current ref_idx to end
                      (pass the full remaining track, not just N+1 points —
                      _time_advance_ref selects the right waypoints internally)

        Returns:
            (u, solve_time_s, status)
            u:          [delta, a] clipped to physical limits
            solve_time: seconds taken by acados solve()
            status:     0 = feasible, nonzero = infeasible or failure
        """
        # Build time-advanced reference so MPC looks far ahead on tight curves
        ref_ta = self._time_advance_ref(ref_traj, self.N, self.dt)

        # Pin initial state via equality constraints
        self.solver.set(0, 'lbx', x_curr)
        self.solver.set(0, 'ubx', x_curr)

        # Load reference trajectory into every stage cost
        for k in range(self.N):
            wp = ref_ta[k]
            yref = np.array([wp['x'], wp['y'], wp['theta'], wp['v'], 0.0, 0.0])
            self.solver.set(k, 'yref', yref)

        wp_e = ref_ta[self.N]
        self.solver.set(self.N, 'yref',
                        np.array([wp_e['x'], wp_e['y'], wp_e['theta'], wp_e['v']]))

        t0 = time.perf_counter()
        status = self.solver.solve()
        solve_time = time.perf_counter() - t0

        u = self.solver.get(0, 'u')
        u[0] = np.clip(u[0], -DELTA_MAX, DELTA_MAX)
        u[1] = np.clip(u[1], -A_MAX, A_MAX)
        return u, solve_time, status
