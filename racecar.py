import numpy as np
from numpy.typing import ArrayLike

class RaceCar:

    @staticmethod
    def normalize_system(state : ArrayLike, input : ArrayLike, parameters : ArrayLike):
        assert(state.shape == (5,) and input.shape == (2,))
        assert(parameters.shape == (11,))

        state[2:4] = np.clip(state[2:4], parameters[1:3], parameters[4:6])
        state[4] = np.arctan2(np.sin(state[4]), np.cos(state[4]))
        input = np.clip(input, parameters[7:9], parameters[9:11])

        return state, input 

    @staticmethod
    def vehicle_kin(state : ArrayLike, input : ArrayLike, parameters : ArrayLike):
        state, input = RaceCar.normalize_system(state, input, parameters)

        # Kinematic Model (Refer to documentation for math).
        return np.array([
            state[3] * np.cos(state[4]),
            state[3] * np.sin(state[4]),
            input[0], input[1],
            (state[3]/parameters[0])*(np.tan(state[2]))
        ])

    def __init__(self, initial_state : ArrayLike):
        # Car Parameters
        self.wheelbase = 3.6 # m

        # Longitudinal Parameters
        # This acceleration is ~2g
        self.max_acceleration = 20 # m/s^{2}

        # Minimum Velocity allowed on the car
        self.min_velocity = -10 # m/s

        # Maximum velocity allowed
        self.max_velocity = 100 # m/s

        # Steering Parameters
        self.max_steering_angle = 0.9 # rad
        self.max_steering_vel = 0.4 # rad/s

        # Parameter packing
        self.parameters = np.array([
            self.wheelbase, # Car Wheelbase
            -self.max_steering_angle, # x3
            self.min_velocity, # x4
            -np.pi, # x5
            self.max_steering_angle,
            self.max_velocity,
            np.pi,
            -self.max_steering_vel, # u1
            -self.max_acceleration, # u2
            self.max_steering_vel,
            self.max_acceleration
        ])

        # States are specified in the associated documentation.
        assert(initial_state.shape == (5,))
        self.state = initial_state

        # Integration time step for RK4
        self.time_step = 1e-1

    def update(self, u : ArrayLike):
        assert(u.shape == (2,))

        s1 = RaceCar.vehicle_kin(self.state, u, self.parameters)
        s2_state = self.state + self.time_step*(s1/2)

        s2 = RaceCar.vehicle_kin(s2_state, u, self.parameters)
        s3_state = self.state + self.time_step*(s2/2)

        s3 = RaceCar.vehicle_kin(s3_state, u, self.parameters)
        s4_state = self.state + self.time_step*s3

        s4 = RaceCar.vehicle_kin(s4_state, u, self.parameters)
        self.state = self.state + self.time_step*0.1666*(s1 + 2*s2 + 2*s3 + s4)

        self.state, _ = RaceCar.normalize_system(self.state, u, self.parameters)

