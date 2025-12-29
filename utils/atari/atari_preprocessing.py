"""
Derived from https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
Implementation of Atari 2600 Preprocessing following the guidelines of Machado et al., 2018.
"""
import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

class AtariPreprocessing(gym.Wrapper):
    def __init__(self, env, noop_max=30, frame_skip=4, screen_size=64):
        super().__init__(env)
        self.noop_max = noop_max
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        # Buffer for max pooling over most recent two observations
        self.obs_buffer = [
            np.empty(env.observation_space.shape, dtype=np.uint8),
            np.empty(env.observation_space.shape, dtype=np.uint8),
        ]
        self.lives = 0

        _shape = (screen_size, screen_size, 3)
        self.observation_space = Box(low=0, high=255, shape=_shape, dtype=np.uint8)

    @property
    def ale(self):
        return self.env.unwrapped.ale

    def step(self, action):
        total_reward, terminated, truncated, info = 0.0, False, False, {}
        life_loss = False

        for t in range(self.frame_skip):
            _, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if self.ale.lives() < self.lives:
                life_loss = True
                self.lives = self.ale.lives()

            if terminated or truncated:
                break

            # Buffer frames for max pooling (to remove flicker)
            if t == self.frame_skip - 2:
                self.ale.getScreenRGB(self.obs_buffer[1])
            elif t == self.frame_skip - 1:
                self.ale.getScreenRGB(self.obs_buffer[0])

        info["life_loss"] = life_loss
        obs = self._get_obs()
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = self.env.unwrapped.np_random.integers(1, self.noop_max + 1) if self.noop_max > 0 else 0
        for _ in range(noops):
            _, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                self.env.reset(**kwargs)

        self.lives = self.ale.lives()
        self.ale.getScreenRGB(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)
        return self._get_obs(), {}

    def _get_obs(self):
        if self.frame_skip > 1:
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        
        # Resize to your Dreamer4 target resolution
        obs = cv2.resize(self.obs_buffer[0], (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        return np.asarray(obs, dtype=np.uint8)