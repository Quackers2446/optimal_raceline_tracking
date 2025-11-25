import numpy as np
from numpy.typing import ArrayLike
import os

from racetrack import RaceTrack

# Global raceline array (loaded on first use)
_raceline = None
_raceline_path = None

def _load_raceline(raceline_path: str):
    """Load raceline from CSV file and store as global array."""
    global _raceline, _raceline_path
    
    if _raceline is None or _raceline_path != raceline_path:
        if raceline_path and os.path.exists(raceline_path):
            data = np.loadtxt(raceline_path, comments="#", delimiter=",")
            _raceline = data[:, 0:2]  # Nx2 array of (rx, ry)
            _raceline_path = raceline_path
        else:
            raise ValueError(f"Raceline file not found: {raceline_path}")
    
    return _raceline

def _find_closest_index(position: np.ndarray, raceline: np.ndarray) -> int:
    """Find the closest point on raceline to current position."""
    distances = np.linalg.norm(raceline - position, axis=1)
    return np.argmin(distances)

def _compute_curvature(raceline: np.ndarray, idx: int) -> float:
    """Compute curvature at index idx using three consecutive points."""
    n = len(raceline)
    
    # Get three consecutive points (with wrapping)
    p0 = raceline[(idx - 1) % n]
    p1 = raceline[idx]
    p2 = raceline[(idx + 1) % n]
    
    # Compute vectors
    v1 = p1 - p0
    v2 = p2 - p1
    
    # Compute arc length
    ds1 = np.linalg.norm(v1)
    ds2 = np.linalg.norm(v2)
    arc_length = (ds1 + ds2) / 2.0
    
    if arc_length < 1e-6:
        return 0.0
    
    # Compute heading change
    heading1 = np.arctan2(v1[1], v1[0])
    heading2 = np.arctan2(v2[1], v2[0])
    delta_heading = heading2 - heading1
    
    # Normalize angle difference to [-pi, pi]
    delta_heading = np.arctan2(np.sin(delta_heading), np.cos(delta_heading))
    
    # Curvature = delta_heading / arc_length
    k = delta_heading / arc_length
    
    return k

def _get_lookahead_point(raceline: np.ndarray, closest_idx: int, lookahead_distance: float) -> tuple:
    """Get lookahead point on raceline based on arc length."""
    n = len(raceline)
    current_idx = closest_idx
    accumulated_distance = 0.0
    
    # Search forward along raceline
    for i in range(n):
        next_idx = (current_idx + i) % n
        next_next_idx = (next_idx + 1) % n
        
        segment_length = np.linalg.norm(raceline[next_next_idx] - raceline[next_idx])
        accumulated_distance += segment_length
        
        if accumulated_distance >= lookahead_distance:
            # Interpolate between points if needed
            if i > 0:
                prev_idx = (next_idx - 1) % n
                prev_segment_length = np.linalg.norm(raceline[next_idx] - raceline[prev_idx])
                remaining = lookahead_distance - (accumulated_distance - segment_length)
                
                if remaining < prev_segment_length:
                    # Interpolate on previous segment
                    t = remaining / prev_segment_length if prev_segment_length > 0 else 0
                    lookahead_point = raceline[prev_idx] + t * (raceline[next_idx] - raceline[prev_idx])
                    return lookahead_point, accumulated_distance - segment_length + remaining
                else:
                    # Use current point
                    return raceline[next_idx], accumulated_distance
            else:
                return raceline[next_idx], accumulated_distance
    
    # If we've gone all the way around, return the point at lookahead_distance
    return raceline[closest_idx], lookahead_distance

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    """
    Low-level controller: converts desired steering angle and velocity to steering rate and acceleration.
    
    Args:
        state: [sx, sy, delta, v, phi]
        desired: [desired_steering_angle, desired_velocity]
        parameters: vehicle parameters (parameters[0] = wheelbase L)
    
    Returns:
        [steering_rate, acceleration]
    """
    assert(desired.shape == (2,))
    
    # Extract desired values
    delta_r = desired[0]  # desired steering angle
    v_r = desired[1]      # desired velocity
    
    # Extract current state
    delta = state[2]  # current steering angle
    v = state[3]      # current velocity
    
    # Controller gains (tunable)
    k_delta = 5.0  # steering rate gain
    k_v = 2.0      # velocity gain
    
    # Proportional controllers
    v_delta = k_delta * (delta_r - delta)  # steering rate
    a = k_v * (v_r - v)                    # acceleration
    
    return np.array([v_delta, a])

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack, raceline_path : str = None
) -> ArrayLike:
    """
    High-level controller: computes desired steering angle and velocity using pure pursuit and curvature-based planning.
    
    Args:
        state: [sx, sy, delta, v, phi]
        parameters: vehicle parameters (parameters[0] = wheelbase L)
        racetrack: RaceTrack object
        raceline_path: path to raceline CSV file
    
    Returns:
        [desired_steering_angle, desired_velocity]
    """
    # Load raceline
    raceline = _load_raceline(raceline_path)
    
    # Extract state
    sx = state[0]  # x position
    sy = state[1]  # y position
    phi = state[4] # heading
    
    # Extract parameters
    L = parameters[0]  # wheelbase
    
    # Controller parameters (tunable)
    lookahead_distance = 15.0  # meters
    a_y_max = 0.4 * 9.81      # maximum lateral acceleration (m/s^2)
    v_straight_cap = 40.0      # maximum velocity on straight sections (m/s)
    
    # Step 1: Find closest point on raceline
    current_position = np.array([sx, sy])
    closest_idx = _find_closest_index(current_position, raceline)
    
    # Step 2: Get lookahead point
    lookahead_point, _ = _get_lookahead_point(raceline, closest_idx, lookahead_distance)
    px, py = lookahead_point
    
    # Step 3: Convert lookahead point to body frame
    dx = px - sx
    dy = py - sy
    
    x_b = np.cos(phi) * dx + np.sin(phi) * dy
    y_b = -np.sin(phi) * dx + np.cos(phi) * dy
    
    L_look = np.sqrt(x_b**2 + y_b**2)
    
    # Step 4: Pure Pursuit steering law
    # Using exact formula: delta_r = atan2(2 * L * y_b, L_look^2)
    if L_look > 1e-6:
        delta_r = np.arctan2(2 * L * y_b, L_look**2)
    else:
        delta_r = 0.0
    
    # Step 5: Compute curvature at current position
    k = _compute_curvature(raceline, closest_idx)
    
    # Step 6: Curvature-based velocity planning
    if abs(k) > 1e-6:
        v_max = np.sqrt(a_y_max / abs(k))
    else:
        v_max = v_straight_cap
    
    v_r = min(v_straight_cap, v_max)
    
    return np.array([delta_r, v_r])
