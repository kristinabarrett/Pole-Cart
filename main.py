"""
Implementation of reinforcement learning to play the cart pole game
"""
import random
from time import sleep

# PyNoInspect requirement
import gymnasium as gym

__author__ = "Kristina Barrett"

PUSH_RIGHT = 1
PUSH_LEFT = 0
MAX_ANGLE = 0.418
MAX_POSITION = 4.8


def sample_simulation():
    env = gym.make("CartPole-v1", render_mode="human")

    observation, info = env.reset(seed=42)
    for _ in range(1):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def simulate(control_algorithm):
    """
    This function runs random movement commands
    :return:
    """
    env = gym.make("CartPole-v1", render_mode="human")

    observation, info = env.reset(seed=42)
    cart_pos = 0
    cart_vel = 0
    pole_angle = 0
    pole_vel = 0
    for _ in range(1000):
        #sleep(0.2)
        action = control_algorithm(cart_pos, cart_vel, pole_angle, pole_vel)
        print("Moving {}".format("right" if action > 0 else "left"))
        observation, reward, terminated, truncated, info = env.step(action)
        cart_pos, cart_vel, pole_angle, pole_vel = observation

    env.close()


deltaV = -2
firstV = 0


def simple_ai(cart_pos: float, cart_vel: float, pole_angle: float, pole_vel: float) -> int:
    """
    Returns either 0 or 1 or the direction to move the cart.
    :param cart_pos: Position of the cart
    :param cart_vel: Velocity of the cart
    :param pole_angle: Angle of the pole
    :param pole_vel: Velocity of the pole
    :return: 0 to move left, 1 to move right
    """
    global deltaV, firstV
    if deltaV == -2:
        deltaV = -1
        firstV = cart_vel
        return 1
    if deltaV == -1:
        deltaV = cart_vel - firstV
        print("Delta V", deltaV)

    if pole_angle + pole_vel > 0:  # Pole will be falling right
        if cart_vel > 0:  # Cart is moving right
            if cart_vel + deltaV > 0:
                return 1
            return 0
        return 1
    elif pole_angle + pole_vel < 0:  # pole will be falling left
        if cart_vel < 0:  # Cart is moving left
            if cart_vel + deltaV < 0:
                return 0
            return 1
        return 0
    else:  # Guess
        return random.Random.randint(0, 1)


def pid_control(cart_pos: float, cart_vel: float, pole_angle: float, pole_vel: float) -> int:
    expected_angle = pole_angle + pole_vel
    expected_position = cart_pos + cart_vel

    okay_angles = 0.2 * MAX_ANGLE  # Allow angles within x% of the target
    okay_position = 0.1 * MAX_POSITION  # Allow positions within x% of the target

    if abs(expected_angle) > okay_angles:
        if expected_angle > 0:
            return PUSH_RIGHT
        elif expected_angle < 0:
            return PUSH_LEFT
    elif abs(expected_position) > okay_position:
        if expected_position > 0:
            return PUSH_RIGHT
        elif expected_position < 0:
            return PUSH_LEFT
    return random.choice([PUSH_RIGHT, PUSH_LEFT])


def velocity_comparison(cart_pos: float, cart_vel: float, pole_angle: float, pole_vel: float) -> int:
    if pole_angle > 0:
        if pole_vel > 0:
            if cart_vel < pole_vel:
                return PUSH_RIGHT
            else:
                return PUSH_LEFT
        else:
            if cart_vel < 0 - pole_vel:
                return PUSH_LEFT
            else:
                return PUSH_RIGHT
    else:
        if pole_vel > 0:
            if cart_vel < pole_vel:
                return PUSH_RIGHT
            else:
                return PUSH_LEFT
        else:
            if cart_vel < 0 - pole_vel:
                return PUSH_LEFT
            else:
                return PUSH_RIGHT


if __name__ == "__main__":
    simulate(velocity_comparison)
