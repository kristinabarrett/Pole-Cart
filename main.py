"""
Implementation of reinforcement learning to play the cart pole game
"""

import gymnasium as gym

__author__ = "Kristina Barrett"


def sample_simulation():
    env = gym.make("CartPole-v1", render_mode="human")

    observation, info = env.reset(seed=42)
    for _ in range(1):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def random_movement():
    env = gym.make("CartPole-v1", render_mode="human")

    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = 1
        observation, reward, terminated, truncated, info = env.step(action)

    env.close()


if __name__ == "__main__":
    random_movement()
