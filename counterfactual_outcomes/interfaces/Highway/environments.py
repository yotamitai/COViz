import numpy as np

from highway_env.envs import HighwayEnv, Action
from gym.envs.registration import register
from highway_env.utils import lmap
from highway_env.vehicle.controller import ControlledVehicle


class Plain(HighwayEnv):
    """rewarded for driving in parallel to a car"""

    def _reward(self, action: Action) -> float:
        obs = self.observation_type.observe()
        other_cars = obs[1:]
        dist_closest_car_in_lane = [x[1] for x in other_cars if x[1] > 0 and abs(x[2]) <= 0.05]
        scaled_speed = lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])

        # safety distance from car in same lane
        if not dist_closest_car_in_lane or dist_closest_car_in_lane[0] > 0.02:
            keeping_distance = 1
        else:
            keeping_distance = -1

        reward = \
            + self.config["keep_distance_reward"] * keeping_distance \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
            + self.config["collision_reward"] * self.vehicle.crashed

        reward = -10 if not self.vehicle.on_road else reward
        return reward


register(
    id='Plain-v0',
    entry_point='counterfactual_outcomes.interfaces.Highway.environments:Plain',
)