import gymnasium as gym
from gymnasium import spaces

import numpy as np


class TradingEnv(gym.Env):
    """
    Simple trading environment for reinforcement learning.

    Actions:
    0: Hold
    1: Buy
    2: Sell
    """

    def __init__(self, data):

        super(TradingEnv, self).__init__()

        self.data = data

        self.current_step = 0


        # Action space
        self.action_space = spaces.Discrete(3)


        # Observation:
        # [price, volume]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        )


        self.balance = 10000
        self.position = 0


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0

        self.balance = 10000
        self.position = 0


        observation = self._get_observation()

        return observation, {}


    def _get_observation(self):

        price = self.data.iloc[self.current_step]["Close"]

        volume = self.data.iloc[self.current_step]["Volume"]


        return np.array(
            [
                price,
                volume
            ],
            dtype=np.float32
        )


    def step(self, action):

        current_price = self.data.iloc[
            self.current_step
        ]["Close"]


        old_balance = self.balance


        # Buy
        if action == 1:

            if self.balance > current_price:

                self.position += 1

                self.balance -= current_price


        # Sell
        elif action == 2:

            if self.position > 0:

                self.position -= 1

                self.balance += current_price



        self.current_step += 1


        terminated = (
            self.current_step >= len(self.data)-1
        )


        new_value = (
            self.balance +
            self.position *
            current_price
        )


        reward = new_value - old_balance


        observation = self._get_observation()


        return (
            observation,
            reward,
            terminated,
            False,
            {}
        )
