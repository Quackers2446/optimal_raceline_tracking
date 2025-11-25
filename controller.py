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

        self.k_delta = 5.0   # steering rate gain
        self.k_v = 2.0       # velocity gain

        # high-level params
        self.lookahead_distance = 15.0          # meters
        self.a_y_max = 0.4 * 9.81               # max lateral accel
        self.v_straight_cap = 40.0              # m/s

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
        raceline = self._load_raceline()
        n = len(raceline)

        p0 = raceline[(idx - 1) % n]
        p1 = raceline[idx]
        p2 = raceline[(idx + 1) % n]

        v1 = p1 - p0
        v2 = p2 - p1

        ds1 = np.linalg.norm(v1)
        ds2 = np.linalg.norm(v2)
        arc_length = (ds1 + ds2) / 2.0

        if arc_length < 1e-6:
            return 0.0

        h1 = np.arctan2(v1[1], v1[0])
        h2 = np.arctan2(v2[1], v2[0])
        dtheta = np.arctan2(np.sin(h2 - h1), np.cos(h2 - h1))

        return dtheta / arc_length

    def _get_lookahead_point(self, closest_idx: int, lookahead_distance: float):
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
        Compute desired steering angle and velocity.
        (This is your old `controller(...)` logic.)
        """
        sx = state[0]
        sy = state[1]
        phi = state[4]
        L = parameters[0]  # wheelbase

        # 1. find closest raceline point
        current_pos = np.array([sx, sy])
        closest_idx = self._find_closest_index(current_pos)

        # 2. lookahead point
        lookahead_point = self._get_lookahead_point(
            closest_idx, self.lookahead_distance
        )
        px, py = lookahead_point

        # 3. transform to body frame
        dx = px - sx
        dy = py - sy
        x_b = np.cos(phi) * dx + np.sin(phi) * dy
        y_b = -np.sin(phi) * dx + np.cos(phi) * dy
        L_look = np.sqrt(x_b**2 + y_b**2)

        # 4. pure pursuit steering
        if L_look > 1e-6:
            delta_r = np.arctan2(2 * L * y_b, L_look**2)
        else:
            delta_r = 0.0

        # 5. curvature-based velocity planning
        k = self._compute_curvature(closest_idx)
        if abs(k) > 1e-6:
            v_max = np.sqrt(self.a_y_max / abs(k))
        else:
            v_max = self.v_straight_cap

        v_r = min(self.v_straight_cap, v_max)
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
        (This is your old `lower_controller(...)` logic.)
        """
        assert desired.shape == (2,)

        delta_r = desired[0]
        v_r = desired[1]

        delta = state[2]
        v = state[3]

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

# This to be used so our class-based structure is compatible with existing code in main

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