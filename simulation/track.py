"""Waypoint generators for simulation and testing."""
import numpy as np
from typing import List, Dict


def _normalize(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def straight_track(length: float = 50.0, v_ref: float = 3.0,
                   n_points: int = 200) -> List[Dict]:
    """Straight line along x-axis."""
    xs = np.linspace(0, length, n_points)
    return [{'x': x, 'y': 0.0, 'theta': 0.0, 'v': v_ref} for x in xs]


def curve_track(radius: float = 3.0, sweep_deg: float = 90.0,
                v_ref: float = 2.0, n_points: int = 100,
                start_x: float = 0.0, start_y: float = 0.0,
                start_theta: float = 0.0,
                direction: int = 1) -> List[Dict]:
    """
    Circular arc.
    direction: +1 = left/CCW (default), -1 = right/CW.
    Center is placed perpendicular to the current heading,
    on the left (+1) or right (-1) side.
    """
    sweep = np.radians(sweep_deg)
    if direction >= 0:
        cx = start_x + radius * np.cos(start_theta + np.pi / 2)
        cy = start_y + radius * np.sin(start_theta + np.pi / 2)
        start_angle = start_theta - np.pi / 2
        angles = np.linspace(start_angle, start_angle + sweep, n_points)
        def theta_fn(a): return a + np.pi / 2
    else:
        cx = start_x + radius * np.cos(start_theta - np.pi / 2)
        cy = start_y + radius * np.sin(start_theta - np.pi / 2)
        start_angle = start_theta + np.pi / 2
        angles = np.linspace(start_angle, start_angle - sweep, n_points)
        def theta_fn(a): return a - np.pi / 2

    waypoints = []
    for a in angles:
        x = cx + radius * np.cos(a)
        y = cy + radius * np.sin(a)
        theta = _normalize(theta_fn(a))
        waypoints.append({'x': float(x), 'y': float(y),
                          'theta': float(theta), 'v': v_ref})
    return waypoints


def mixed_track() -> List[Dict]:
    """
    Straight → left-curve → straight → left-curve → straight (Z-chicane).
    Used for full FSM simulation: exercises straight→curve→straight transitions.
    """
    track = []

    s1 = straight_track(length=20.0, v_ref=3.0, n_points=100)
    track.extend(s1)

    last = s1[-1]
    c1 = curve_track(radius=4.0, sweep_deg=90.0, v_ref=2.0, n_points=80,
                     start_x=last['x'], start_y=last['y'],
                     start_theta=last['theta'], direction=1)
    track.extend(c1)

    last = c1[-1]
    dx, dy = np.cos(last['theta']), np.sin(last['theta'])
    s2 = [{'x': last['x'] + dx * i * 0.2,
            'y': last['y'] + dy * i * 0.2,
            'theta': last['theta'], 'v': 3.0} for i in range(75)]
    track.extend(s2)

    last = s2[-1]
    c2 = curve_track(radius=4.0, sweep_deg=90.0, v_ref=2.0, n_points=80,
                     start_x=last['x'], start_y=last['y'],
                     start_theta=last['theta'], direction=1)
    track.extend(c2)

    last = c2[-1]
    dx, dy = np.cos(last['theta']), np.sin(last['theta'])
    s3 = [{'x': last['x'] + dx * i * 0.25,
            'y': last['y'] + dy * i * 0.25,
            'theta': last['theta'], 'v': 3.0} for i in range(80)]
    track.extend(s3)

    return track


def hairpin_track(straight_len: float = 20.0, radius: float = 3.0,
                  v_ref_straight: float = 3.0, v_ref_curve: float = 1.5,
                  n_pts_straight: int = 100, n_pts_curve: int = 160) -> List[Dict]:
    """Straight → 180° left hairpin → straight back (heading reversed)."""
    track = []

    s1 = straight_track(length=straight_len, v_ref=v_ref_straight,
                        n_points=n_pts_straight)
    track.extend(s1)

    last = s1[-1]
    h = curve_track(radius=radius, sweep_deg=180.0, v_ref=v_ref_curve,
                    n_points=n_pts_curve,
                    start_x=last['x'], start_y=last['y'],
                    start_theta=last['theta'], direction=1)
    track.extend(h)

    last = h[-1]
    dx, dy = np.cos(last['theta']), np.sin(last['theta'])
    s2 = [{'x': last['x'] + dx * i * 0.25,
            'y': last['y'] + dy * i * 0.25,
            'theta': last['theta'], 'v': v_ref_straight}
           for i in range(n_pts_straight)]
    track.extend(s2)

    return track


def slalom_track(n_gates: int = 4, straight_entry: float = 10.0,
                 straight_between: float = 8.0, radius: float = 3.5,
                 v_ref_s: float = 3.0, v_ref_c: float = 2.0) -> List[Dict]:
    """
    Alternating left-right S-curves (slalom).
    Odd gates turn left, even gates turn right.
    """
    track = []

    s0 = straight_track(length=straight_entry, v_ref=v_ref_s, n_points=50)
    track.extend(s0)

    for i in range(n_gates):
        last = track[-1]
        direction = 1 if (i % 2 == 0) else -1
        c = curve_track(radius=radius, sweep_deg=90.0, v_ref=v_ref_c,
                        n_points=70,
                        start_x=last['x'], start_y=last['y'],
                        start_theta=last['theta'], direction=direction)
        track.extend(c)

        last = track[-1]
        dx, dy = np.cos(last['theta']), np.sin(last['theta'])
        n = max(1, int(straight_between / 0.2))
        s = [{'x': last['x'] + dx * j * 0.2,
               'y': last['y'] + dy * j * 0.2,
               'theta': last['theta'], 'v': v_ref_s} for j in range(n)]
        track.extend(s)

    return track


def oval_track(straight_len: float = 30.0, radius: float = 8.0,
               v_ref_straight: float = 3.5, v_ref_curve: float = 2.0) -> List[Dict]:
    """
    Oval: straight → 180° semicircle → straight back → 180° semicircle.
    The two straights are 2*radius apart in y.
    Not a closed loop for simulation: ends at the second semicircle exit.
    """
    track = []

    s1 = straight_track(length=straight_len, v_ref=v_ref_straight, n_points=150)
    track.extend(s1)

    last = s1[-1]
    c1 = curve_track(radius=radius, sweep_deg=180.0, v_ref=v_ref_curve,
                     n_points=160,
                     start_x=last['x'], start_y=last['y'],
                     start_theta=last['theta'], direction=1)
    track.extend(c1)

    last = c1[-1]
    dx, dy = np.cos(last['theta']), np.sin(last['theta'])
    n = max(1, int(straight_len / 0.2))
    s2 = [{'x': last['x'] + dx * i * 0.2,
            'y': last['y'] + dy * i * 0.2,
            'theta': last['theta'], 'v': v_ref_straight} for i in range(n)]
    track.extend(s2)

    last = s2[-1]
    c2 = curve_track(radius=radius, sweep_deg=180.0, v_ref=v_ref_curve,
                     n_points=160,
                     start_x=last['x'], start_y=last['y'],
                     start_theta=last['theta'], direction=1)
    track.extend(c2)

    return track
