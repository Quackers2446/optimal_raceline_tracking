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

        # === Low-level gains : ===

        # K_delta: steering rate gain
        # K_v: velocity gain
        # v_min: minimum speed ( car accelerates/breakes toward at least this speed )

        self.k_delta = 10.0
        self.k_v = 4.0
        self.v_min = 3.0

        # === High-level params : control pure pursuit steering and curvature-based speed planning ===
        
        # a_y_max: max lateral acceleration (m/s²)
        # v_straight_cap: max speed on straight segments (m/s)
        # k_straight_eps: curvature threshold to treat as straight 

        # lookahead_straight: lookahead distance on straights (m) 
            # larger: smoother but less responsive
            # smaller : more responsive but possibly oscillatory
        # lookahead_curve: lookahead distance on curves (m)
            # larger: more damping in corners
            # smaller: more aggressive cornering
        
        # steer_damping_gain: gain for speed-based steering damping

        self.a_y_max = 8.0
        self.v_straight_cap = 100.0

        self.k_straight_eps = 8e-5

        # Lookahead distances
        self.lookahead_straight = 40.0 
        self.lookahead_curve   = 6.0

        self.min_L_look = 3.0

        self.steer_damping_gain = 0.022

        # Velocity smoothing: respond faster to speed changes (less “laggy” braking).

        # === Braking preview params (physics-based) ===
        # Approximate comfortable braking decel (m/s^2)
        self.a_x_brake_est = 20.0

        # Scale factor on stopping distance ( > 1 is safer )
        self.brake_preview_scale = 1.2

        # Clamp preview distance so it does not explode on very high speed
        self.brake_preview_min_dist = 0.0    # allow near-zero at low speed
        self.brake_preview_max_dist = 200.0  # tweak as needed

        self.brake_preview_v_min = 25.0    # m/s, ~65 km/h, tweak as needed

        # Only use future limit if we are significantly above it
        self.future_limit_margin = 2.5     # m/s margin

        # Only apply advanced braking / preview logic when above this speed
        self.preview_enable_v = 77.0  # m/s

        self.prev_v_r: float | None = None

    def get_raceline(self) -> np.ndarray | None:
        return self._raceline

    def _load_raceline(self, raceline_path: str | None = None) -> np.ndarray:
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
        raceline = self._load_raceline()
        n = len(raceline)

        # 1) local curvature at current point
        k_local = abs(self._compute_curvature(closest_idx))

        # 2) choose how far we look ahead (shorter than before)
        #    ~12 .. 22 points depending on speed
        window = int(base_window + min(max(v / 8.0, 0.0), 10.0))

        k_max = k_local
        for i in range(1, window):
            idx = (closest_idx + i) % n
            k_i = abs(self._compute_curvature(idx))
            if k_i > k_max:
                k_max = k_i

        # 3) blend local vs ahead curvature
        #    - at low speed: mostly local
        #    - at high speed: more influenced by k_max
        alpha = np.clip(v / 50.0, 0.2, 0.6)   # 20%..60% weight on k_max
        k_eff = (1.0 - alpha) * k_local + alpha * k_max

        return k_eff

    def _max_curvature_in_distance(self, closest_idx: int, distance: float) -> float:
        raceline = self._load_raceline()
        n = len(raceline)

        travelled = 0.0
        idx = closest_idx
        k_max = 0.0

        while travelled < distance:
            nxt = (idx + 1) % n
            seg = raceline[nxt] - raceline[idx]
            seg_len = float(np.linalg.norm(seg))

            if seg_len < 1e-6:
                break

            travelled += seg_len
            k_here = abs(self._compute_curvature(nxt))
            if k_here > k_max:
                k_max = k_here

            idx = nxt
            if idx == closest_idx:
                # we wrapped the whole lap, no need to continue
                break

        return k_max
    
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
        sx, sy, delta, v, phi = state

        L = float(parameters[0])  # wheelbase (3.6)

        # 1. find closest raceline point
        current_pos = np.array([sx, sy])
        closest_idx = self._find_closest_index(current_pos)

        # 2. local curvature
        # k = self._compute_curvature(closest_idx)
        k = self._curvature_ahead(closest_idx, base_window=10, v=float(v))

        # Treat tiny curvature as straight so we never slow on straights
        if abs(k) < self.k_straight_eps:
            k = 0.0

        # 3. choose lookahead distance based on speed and curvature
        v_abs = max(float(v), 0.0)

        if k == 0.0:
            # On straights: look further ahead at high speed for smoother steering
            base = self.lookahead_straight
            extra = 0.2 * v_abs
            lookahead_distance = np.clip(base + extra, base, 120.0)
        else:
            # In curves: increase lookahead a bit with speed, but not too much
            base = self.lookahead_curve
            extra = 0.05 * v_abs
            lookahead_distance = np.clip(base + extra, base, 40.0)


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

        # 8. curvature-based velocity planning with conditional preview
        v_abs = max(float(v), 0.0)

        # Local curvature speed limit (what we need *right here*)
        if abs(k) < self.k_straight_eps:
            v_limit_local = self.v_straight_cap
        else:
            v_limit_local = min(self.v_straight_cap,
                                np.sqrt(self.a_y_max / abs(k)))

        # Default target: just obey local limit
        v_target = max(self.v_min, v_limit_local)

        # Only apply advanced braking / preview when we are truly fast
        if v_abs > self.preview_enable_v and self.a_x_brake_est > 0.0:
            stop_dist = (v_abs ** 2) / (2.0 * self.a_x_brake_est)
            preview_dist = self.brake_preview_scale * stop_dist
            preview_dist = float(np.clip(preview_dist, 0.0, self.brake_preview_max_dist))

            k_future = self._max_curvature_in_distance(closest_idx, preview_dist)

            if k_future < self.k_straight_eps:
                v_limit_future = self.v_straight_cap
            else:
                v_limit_future = min(self.v_straight_cap,
                                     np.sqrt(self.a_y_max / k_future))

            # Only let future limit override if we are actually above it by some margin
            if v_abs > v_limit_future + self.future_limit_margin:
                v_target = max(self.v_min, min(v_limit_local, v_limit_future))
            else:
                v_target = max(self.v_min, v_limit_local)

        # 9. smooth desired speed: react faster when slowing down than speeding up
        if self.prev_v_r is None:
            v_r = v_target
        else:
            beta_accel = 0.3   # smoother when speeding up
            beta_brake = 0.9   # aggressive when slowing down

            if v_target < self.prev_v_r:
                beta = beta_brake
            else:
                beta = beta_accel

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
    ctrl = _get_controller()
    return ctrl.get_raceline()
