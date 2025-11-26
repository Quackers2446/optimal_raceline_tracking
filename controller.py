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
        self.k_delta = 10.0   # steering rate gain
        self.k_v = 3.0        # velocity gain
        self.v_min = 3.0      # minimum target speed (m/s)

        # === High-level params ===
        # Lateral acceleration limit (controls corner speed)
        self.a_y_max = 6.5               # m/s^2

        # Straight-line cap: full car capability
        self.v_straight_cap = 100.0      # m/s

        # For really tight corners we liked ~75 before
        self.v_turn_cap = 80.0           # max speed when a significant corner is coming

        # Curvature thresholds
        self.k_straight_eps = 2e-4          # for steering
        self.k_straight_eps_speed = 3e-4    # for speed

        # Lookahead distances (steering)
        self.lookahead_straight = 25.0    # steering lookahead on straights
        self.lookahead_curve = 7.0        # steering lookahead in turns

        # Speed-planning horizon (meters)
        self.speed_horizon_distance = 175.0  # how far ahead we look for tight turns

        # Physical braking capability (from RaceCar)
        self.max_brake = 20.0  # m/s^2

        # Minimum geometric distance to target to avoid huge steering
        self.min_L_look = 3.0

        # Steering damping at high speed (to reduce wobble)
        self.steer_damping_gain = 0.022   # mild

        # Steering smoothing (currently unused, can enable if needed)
        self.delta_smooth_alpha = 0.3
        self.prev_delta_r: float | None = None

        # Velocity smoothing (to avoid jerk)
        self.v_smooth_beta = 0.6
        self.prev_v_r: float | None = None

    # ---------- internal helpers ----------

    def get_raceline(self) -> np.ndarray | None:
        """Get the current raceline data."""
        return self._raceline

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
        """
        Steering lookahead based on arc length along the raceline.
        """
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

        return raceline[closest_idx]

    def _curvature_ahead(
        self,
        closest_idx: int,
        base_window: int = 12,
        v: float = 0.0,
    ) -> float:
        """
        Effective curvature for STEERING: uses a fixed number of raceline
        points scaled with speed. This is local-ish and meant for shaping
        the steering, not long-range speed planning.
        """
        raceline = self._load_raceline()
        n = len(raceline)

        # 1) local curvature at current point
        k_local = abs(self._compute_curvature(closest_idx))

        # 2) choose how far we look ahead (in indices)
        window = int(base_window + min(max(v / 8.0, 0.0), 10.0))

        k_max = k_local
        for i in range(1, window):
            idx = (closest_idx + i) % n
            k_i = abs(self._compute_curvature(idx))
            if k_i > k_max:
                k_max = k_i

        # 3) blend local vs ahead curvature
        alpha = np.clip(v / 50.0, 0.2, 0.6)   # 20%..60% weight on k_max
        k_eff = (1.0 - alpha) * k_local + alpha * k_max

        return k_eff

    def _curvature_max_in_distance(
        self,
        start_idx: int,
        max_distance: float,
    ) -> tuple[float, float]:
        """
        Scan forward along the raceline up to a given arc length (meters),
        and return:
          - max curvature magnitude in that horizon
          - distance along the raceline where that max occurs

        This is for SPEED planning: we want to see tight turns early.
        """
        raceline = self._load_raceline()
        n = len(raceline)

        accumulated_distance = 0.0
        k_max = 0.0
        dist_at_max = max_distance

        for i in range(n):
            idx = (start_idx + i) % n
            nxt = (idx + 1) % n

            seg = raceline[nxt] - raceline[idx]
            seg_len = np.linalg.norm(seg)
            accumulated_distance += seg_len

            k_i = abs(self._compute_curvature(idx))
            if k_i > k_max:
                k_max = k_i
                dist_at_max = accumulated_distance

            if accumulated_distance >= max_distance:
                break

        return float(k_max), float(dist_at_max)

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
        - short-horizon curvature for steering
        - long-horizon curvature + braking distance for speed
        """
        sx, sy, delta, v, phi = state
        L = float(parameters[0])  # wheelbase (3.6)

        # 1. find closest raceline point
        current_pos = np.array([sx, sy])
        closest_idx = self._find_closest_index(current_pos)

        # 2. curvature for steering (shorter horizon)
        k_steer = self._curvature_ahead(closest_idx, base_window=10, v=float(v))
        if abs(k_steer) < self.k_straight_eps:
            k_steer = 0.0

        # 2b. curvature for speed planning (long horizon in METERS)
        k_speed, dist_to_corner = self._curvature_max_in_distance(
            closest_idx, self.speed_horizon_distance
        )
        if abs(k_speed) < self.k_straight_eps_speed:
            k_speed = 0.0

        # 3. choose lookahead distance (for steering) based on k_steer
        if k_steer == 0.0:
            lookahead_distance = self.lookahead_straight
        else:
            lookahead_distance = self.lookahead_curve

        # 4. lookahead point for steering
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
        delta_r = delta_r / (1.0 + self.steer_damping_gain * max(v, 0.0))

        # (optional) steering smoothing could go here if needed

        # 8. curvature-based velocity planning with braking logic
        if k_speed == 0.0:
            # No significant turn in horizon: go full straight-line speed
            v_target = self.v_straight_cap
        else:
            # Corner speed from lateral accel limit (cap at v_turn_cap)
            v_corner = np.sqrt(self.a_y_max / k_speed)
            v_corner = min(v_corner, self.v_turn_cap)

            # Simple braking distance check
            v_curr = max(float(v), 0.0)
            if v_curr > v_corner:
                d_brake = (v_curr**2 - v_corner**2) / (2.0 * self.max_brake + 1e-6)
            else:
                d_brake = 0.0

            # If we are closer to the corner than the ideal braking distance,
            # be extra conservative and ramp down toward v_corner.
            if d_brake > 0.0 and dist_to_corner < d_brake:
                # Scale based on how "late" we are:
                # - if dist_to_corner == d_brake: factor ~ 1.0
                # - if much less: factor goes down to ~0.5
                ratio = dist_to_corner / d_brake
                ratio = max(0.5, min(1.0, ratio))
                v_target = v_corner * ratio
            else:
                v_target = v_corner

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


def get_raceline() -> np.ndarray | None:
    """Get the raceline data from the controller."""
    ctrl = _get_controller()
    return ctrl.get_raceline()
