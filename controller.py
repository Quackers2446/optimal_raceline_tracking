import numpy as np
from numpy.typing import ArrayLike
import os

from racetrack import RaceTrack


class Controller:

    def __init__(self, raceline_path: str | None = None):
        self.raceline_path = raceline_path
        self._raceline: np.ndarray | None = None

        if raceline_path is not None:
            self._load_raceline(raceline_path)

        # === Low-level gains ===
        self.k_delta = 15.0   # steering rate gain
        self.k_v = 2.0       # velocity gain
        self.v_min = 5.0    # minimum target speed (m/s)

        # === High-level params ===
        # Lateral acceleration limit (≈ 1 g): controls corner speed
        self.a_y_max = 20.0               # m/s^2
        # Straight-line cap: close to car max (100 m/s)
        self.v_straight_cap = 100.0        # m/s

        # Curvature threshold: treat tiny curvature as straight
        self.k_straight_eps = 5e-4

        # Lookahead distances
        self.lookahead_straight = 30.0    # long on straights
        self.lookahead_curve = 15.0       # shorter in turns

        # Minimum geometric distance to target to avoid huge steering
        self.min_L_look = 3.0

        # Steering damping at high speed (to reduce wobble)
        self.steer_damping_gain = 0.015   # mild

        # Steering smoothing to kill small oscillations
        self.delta_smooth_alpha = 0.3
        self.prev_delta_r: float | None = None

        # Optional velocity smoothing (to avoid jerk)
        self.v_smooth_beta = 0.4
        self.prev_v_r: float | None = None

    # ---------- internal helpers ----------

    def _load_raceline(self, raceline_path: str | None = None) -> np.ndarray:
        """Load raceline from CSV file and cache it."""
        if raceline_path is None:
            raceline_path = self.raceline_path

        if raceline_path is None:
            raise ValueError("Raceline path not provided.")

        if self._raceline is None or raceline_path != self.raceline_path:
            if os.path.exists(raceline_path):
                data = np.loadtxt(raceline_path, comments="#", delimiter=",")
                self._raceline = data[:, 0:2]  # Nx2 (rx, ry)
                self.raceline_path = raceline_path
            else:
                raise ValueError(f"Raceline file not found: {raceline_path}")

        return self._raceline

    def _find_closest_index(self, position: np.ndarray) -> int:
        raceline = self._load_raceline()
        distances = np.linalg.norm(raceline - position, axis=1)
        return int(np.argmin(distances))

    def _compute_curvature(self, idx: int) -> float:
        """
        Smooth curvature using three-point circumcircle approximation.
        Less noisy than heading-difference / arc-length.
        """
        raceline = self._load_raceline()
        n = len(raceline)
        if n < 3:
            return 0.0

        p0 = raceline[(idx - 1) % n]
        p1 = raceline[idx]
        p2 = raceline[(idx + 1) % n]

        a = p1 - p0
        b = p2 - p1
        c = p2 - p0

        area2 = a[0] * b[1] - a[1] * b[0]  # 2 * signed area
        denom = np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c)
        if denom < 1e-6:
            return 0.0

        k = area2 / denom   # curvature
        return float(k)

    def _get_lookahead_point(self, closest_idx: int, lookahead_distance: float):
        """Forward arc-length search along raceline."""
        raceline = self._load_raceline()
        n = len(raceline)
        accumulated_distance = 0.0

        for i in range(n):
            idx = (closest_idx + i) % n
            nxt = (idx + 1) % n

            seg_len = np.linalg.norm(raceline[nxt] - raceline[idx])
            accumulated_distance += seg_len

            if accumulated_distance >= lookahead_distance:
                return raceline[nxt]

        # fallback if we loop the track
        return raceline[closest_idx]

    # ---------- high-level controller ----------

    def high_level(
        self,
        state: ArrayLike,
        parameters: ArrayLike,
        racetrack: RaceTrack | None = None,
    ) -> np.ndarray:
        """
        Compute desired steering angle and velocity:
        - pure pursuit steering
        - curvature-based speed planning
        - dynamic lookahead (straight vs curve)
        - mild speed-based steering damping
        """
        sx = float(state[0])
        sy = float(state[1])
        phi = float(state[4])
        v = float(state[3])
        L = float(parameters[0])  # wheelbase (3.6)

        # 1. find closest raceline point
        current_pos = np.array([sx, sy])
        closest_idx = self._find_closest_index(current_pos)

        # 2. local curvature
        k = self._compute_curvature(closest_idx)

        # Treat tiny curvature as straight so we never slow on straights
        if abs(k) < self.k_straight_eps:
            k = 0.0

        # 3. choose lookahead distance
        if k == 0.0:
            lookahead_distance = self.lookahead_straight
        else:
            lookahead_distance = self.lookahead_curve

        # 4. lookahead point
        lookahead_point = self._get_lookahead_point(closest_idx, lookahead_distance)
        px, py = float(lookahead_point[0]), float(lookahead_point[1])

        # 5. transform to body frame
        dx = px - sx
        dy = py - sy

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        x_b = cos_phi * dx + sin_phi * dy
        y_b = -sin_phi * dx + cos_phi * dy

        L_look = float(np.sqrt(x_b**2 + y_b**2))
        if L_look < self.min_L_look:
            L_look = self.min_L_look

        # 6. pure pursuit steering
        if L_look > 1e-6:
            delta_r = float(np.arctan2(2.0 * L * y_b, L_look**2))
        else:
            delta_r = 0.0

        # 7. speed-based damping to reduce high-speed wobble
        # (at 50 m/s, denominator ~ 1 + 0.75 = 1.75 → ~40% reduction)
        delta_r = delta_r / (1.0 + self.steer_damping_gain * max(v, 0.0))

        # # 7b. low-pass steer to knock down residual oscillation
        # if self.prev_delta_r is None:
        #     smoothed_delta = delta_r
        # else:
        #     alpha = self.delta_smooth_alpha
        #     smoothed_delta = self.prev_delta_r + alpha * (delta_r - self.prev_delta_r)
        # self.prev_delta_r = smoothed_delta
        # delta_r = smoothed_delta

        # 8. curvature-based velocity planning
        if k == 0.0:
            # Straight: aim for straight-line cap
            v_target = self.v_straight_cap
        else:
            # Corner speed from lateral accel limit
            v_max_curve = np.sqrt(self.a_y_max / abs(k))
            v_target = min(self.v_straight_cap, v_max_curve)

        v_target = max(v_target, self.v_min)

        # 9. smooth desired speed a bit to avoid jerk
        if self.prev_v_r is None:
            v_r = v_target
        else:
            beta = self.v_smooth_beta
            v_r = self.prev_v_r + beta * (v_target - self.prev_v_r)
        self.prev_v_r = float(v_r)

        return np.array([delta_r, v_r])

    # ---------- low-level controller ----------

    def low_level(
        self,
        state: ArrayLike,
        desired: ArrayLike,
        parameters: ArrayLike,
    ) -> np.ndarray:
        """
        Convert desired (delta_r, v_r) into (v_delta, a).
        Rate/accel will be clipped by RaceCar.normalize_system.
        """
        assert desired.shape == (2,)

        delta_r = float(desired[0])
        v_r = float(desired[1])

        delta = float(state[2])
        v = float(state[3])

        v_delta = self.k_delta * (delta_r - delta)
        a = self.k_v * (v_r - v)

        return np.array([v_delta, a])


# ============================================================
# Module-level wrappers for backwards compatibility
# ============================================================

_controller_instance: Controller | None = None


def init_controller(raceline_path: str) -> None:
    """Call this once from main.py to set up the controller."""
    global _controller_instance
    _controller_instance = Controller(raceline_path)


def _get_controller() -> Controller:
    global _controller_instance
    if _controller_instance is None:
        raise RuntimeError("Controller not initialized. Call init_controller() first.")
    return _controller_instance


def controller(
    state: ArrayLike,
    parameters: ArrayLike,
    racetrack: RaceTrack,
) -> np.ndarray:
    ctrl = _get_controller()
    return ctrl.high_level(state, parameters, racetrack)


def lower_controller(
    state: ArrayLike,
    desired: ArrayLike,
    parameters: ArrayLike,
) -> np.ndarray:
    ctrl = _get_controller()
    return ctrl.low_level(state, desired, parameters)
