"""
Implementation of reinforcement learning to play the cart pole game
"""
import random
from matplotlib import pyplot as plt
import gymnasium as gym

import numpy as np
import time
import matplotlib.pyplot as plt

# import the class that implements the Q-Learning algorithm
#from venv import Q_Learning

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
    time_steps = 1000
    # Initialize plotting variables
    figure = plt.figure()
    plt.axis([0, 1000, 0, 1])

    i = 0
    data = []
    times = []

    # [1,2,3]
    # x = [[1,1,2], [2,3,5]
    # x = [[1,2], [1,3], [2,5]]
    # zip(x)

    env = gym.make("CartPole-v1", render_mode="human")

    observation, info = env.reset(seed=42)
    cart_pos = 0
    cart_vel = 0
    pole_angle = 0
    pole_vel = 0
    for i in range(time_steps):
        # sleep(0.2)
        action = control_algorithm(cart_pos, cart_vel, pole_angle, pole_vel)
        print("Moving {}".format("right" if action > 0 else "left"))
        observation, reward, terminated, truncated, info = env.step(action)
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        times.append(i)
        data.append(list(observation))
        if len(data) > 1:
            print(data)
            print(list(zip(data)))
            cps, cvs, pas, pvs = list(zip(*data))
        else:
            print(data)
            cps, cvs, pas, pvs = data[0]
        plt.plot(times, cps, 'r')
        plt.plot(times, cvs, 'b')
        plt.plot(times, pas, 'c')
        plt.plot(times, pvs, 'g')
        plt.pause(0.05)
    plt.show()

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
    """
    Kristina made this -- this comment was left by Kayla so I don't forget
    :param cart_pos:
    :param cart_vel:
    :param pole_angle:
    :param pole_vel:
    :return:
    """
    if pole_angle > 0:  # Pole leans right
        if pole_vel > 0:  # Pole falling down (right +)
            if cart_vel < pole_vel:  # Pole is falling faster than the cart is moving
                return PUSH_RIGHT
            else:  # Cart is moving faster than the pole falls
                return PUSH_LEFT
        else:  # Pole moving up (left -)
            if cart_vel < 0 - pole_vel:
                return PUSH_LEFT
            else:
                return PUSH_RIGHT
    else:
        if pole_vel > 0:
            if 0 - cart_vel < pole_vel:
                return PUSH_RIGHT
            else:
                return PUSH_LEFT
        else:
            if cart_vel < 0 - pole_vel:
                return PUSH_LEFT
            else:
                return PUSH_RIGHT


#if __name__ == "__main__":
#    simulate(velocity_comparison)




class Q_Learning:
    ###########################################################################
    #   START - __init__ function
    ###########################################################################
    # INPUTS:
    # env - Cart Pole environment
    # alpha - step size
    # gamma - discount rate
    # epsilon - parameter for epsilon-greedy approach
    # numberEpisodes - total number of simulation episodes

    # numberOfBins - this is a 4 dimensional list that defines the number of grid points
    # for state discretization
    # that is, this list contains number of bins for every state entry,
    # we have 4 entries, that is,
    # discretization for cart position, cart velocity, pole angle, and pole angular velocity

    # lowerBounds - lower bounds (limits) for discretization, list with 4 entries:
    # lower bounds on cart position, cart velocity, pole angle, and pole angular velocity

    # upperBounds - upper bounds (limits) for discretization, list with 4 entries:
    # upper bounds on cart position, cart velocity, pole angle, and pole angular velocity

    def __init__(self, env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds):
        import numpy as np

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actionNumber = env.action_space.n
        self.numberEpisodes = numberEpisodes
        self.numberOfBins = numberOfBins
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds

        # this list stores sum of rewards in every learning episode
        self.sumRewardsEpisode = []

        # this matrix is the action value function matrix
        self.Qmatrix = np.random.uniform(low=0, high=1, size=(
        numberOfBins[0], numberOfBins[1], numberOfBins[2], numberOfBins[3], self.actionNumber))

    ###########################################################################
    #   END - __init__ function
    ###########################################################################

    ###########################################################################
    # START: function "returnIndexState"
    # for the given 4-dimensional state, and discretization grid defined by
    # numberOfBins, lowerBounds, and upperBounds, this function will return
    # the index tuple (4-dimensional) that is used to index entries of the
    # of the QvalueMatrix

    # INPUTS:
    # state - state list/array, 4 entries:
    # cart position, cart velocity, pole angle, and pole angular velocity

    # OUTPUT: 4-dimensional tuple defining the indices of the QvalueMatrix
    # that correspond to "state" input

    ###############################################################################
    def returnIndexState(self, state):
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]

        cartPositionBin = np.linspace(self.lowerBounds[0], self.upperBounds[0], self.numberOfBins[0])
        cartVelocityBin = np.linspace(self.lowerBounds[1], self.upperBounds[1], self.numberOfBins[1])
        poleAngleBin = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.numberOfBins[2])
        poleAngleVelocityBin = np.linspace(self.lowerBounds[3], self.upperBounds[3], self.numberOfBins[3])

        indexPosition = np.maximum(np.digitize(state[0], cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(state[1], cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(state[2], poleAngleBin) - 1, 0)
        indexAngularVelocity = np.maximum(np.digitize(state[3], poleAngleVelocityBin) - 1, 0)

        return tuple([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])
        ###########################################################################

    #   END - function "returnIndexState"
    ###########################################################################

    ###########################################################################
    #    START - function for selecting an action: epsilon-greedy approach
    ###########################################################################
    # this function selects an action on the basis of the current state
    # INPUTS:
    # state - state for which to compute the action
    # index - index of the current episode
    def selectAction(self, state, index):

        # first 500 episodes we select completely random actions to have enough exploration
        if index < 500:
            return np.random.choice(self.actionNumber)

            # Returns a random real number in the half-open interval [0.0, 1.0)
        # this number is used for the epsilon greedy approach
        randomNumber = np.random.random()

        # after 7000 episodes, we slowly start to decrease the epsilon parameter
        if index > 7000:
            self.epsilon = 0.999 * self.epsilon

        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionNumber)

            # otherwise, we are selecting greedy actions
        else:
            # we return the index where Qmatrix[state,:] has the max value
            # that is, since the index denotes an action, we select greedy actions
            return np.random.choice(np.where(
                self.Qmatrix[self.returnIndexState(state)] == np.max(self.Qmatrix[self.returnIndexState(state)]))[0])
            # here we need to return the minimum index since it can happen
            # that there are several identical maximal entries, for example
            # import numpy as np
            # a=[0,1,1,0]
            # np.where(a==np.max(a))
            # this will return [1,2], but we only need a single index
            # that is why we need to have np.random.choice(np.where(a==np.max(a))[0])
            # note that zero has to be added here since np.where() returns a tuple

    ###########################################################################
    #    END - function selecting an action: epsilon-greedy approach
    ###########################################################################

    ###########################################################################
    #    START - function for simulating learning episodes
    ###########################################################################

    def simulateEpisodes(self):
        import numpy as np
        # here we loop through the episodes
        for indexEpisode in range(self.numberEpisodes):

            # list that stores rewards per episode - this is necessary for keeping track of convergence
            rewardsEpisode = []

            # reset the environment at the beginning of every episode
            (stateS, _) = self.env.reset()
            stateS = list(stateS)

            print("Simulating episode {}".format(indexEpisode))

            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState = False
            while not terminalState:
                # return a discretized index of the state

                stateSIndex = self.returnIndexState(stateS)

                # select an action on the basis of the current state, denoted by stateS
                actionA = self.selectAction(stateS, indexEpisode)

                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                # prime means that it is the next state
                (stateSprime, reward, terminalState, _, _) = self.env.step(actionA)

                rewardsEpisode.append(reward)

                stateSprime = list(stateSprime)

                stateSprimeIndex = self.returnIndexState(stateSprime)

                # return the max value, we do not need actionAprime...
                QmaxPrime = np.max(self.Qmatrix[stateSprimeIndex])

                if not terminalState:
                    # stateS+(actionA,) - we use this notation to append the tuples
                    # for example, for stateS=(0,0,0,1) and actionA=(1,0)
                    # we have stateS+(actionA,)=(0,0,0,1,0)
                    error = reward + self.gamma * QmaxPrime - self.Qmatrix[stateSIndex + (actionA,)]
                    self.Qmatrix[stateSIndex + (actionA,)] = self.Qmatrix[stateSIndex + (actionA,)] + self.alpha * error
                else:
                    # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0
                    error = reward - self.Qmatrix[stateSIndex + (actionA,)]
                    self.Qmatrix[stateSIndex + (actionA,)] = self.Qmatrix[stateSIndex + (actionA,)] + self.alpha * error

                # set the current state to the next state
                stateS = stateSprime

            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))

    ###########################################################################
    #    END - function for simulating learning episodes
    ###########################################################################

    ###########################################################################
    #    START - function for simulating the final learned optimal policy
    ###########################################################################
    # OUTPUT:
    # env1 - created Cart Pole environment
    # obtainedRewards - a list of obtained rewards during time steps of a single episode

    # simulate the final learned optimal policy
    def simulateLearnedStrategy(self):
        import gym
        import time
        env1 = gym.make('CartPole-v1', render_mode='human')
        (currentState, _) = env1.reset()
        env1.render()
        timeSteps = 1000
        # obtained rewards at every time step
        obtainedRewards = []

        for timeIndex in range(timeSteps):
            print(timeIndex)
            # select greedy actions
            actionInStateS = np.random.choice(np.where(self.Qmatrix[self.returnIndexState(currentState)] == np.max(
                self.Qmatrix[self.returnIndexState(currentState)]))[0])
            currentState, reward, terminated, truncated, info = env1.step(actionInStateS)
            obtainedRewards.append(reward)
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        return obtainedRewards, env1

    ###########################################################################
    #    END - function for simulating the final learned optimal policy
    ###########################################################################

    ###########################################################################
    #    START - function for simulating random actions many times
    #   this is used to evaluate the optimal policy and to compare it with a random policy
    ###########################################################################
    #  OUTPUT:
    # sumRewardsEpisodes - every entry of this list is a sum of rewards obtained by simulating the corresponding episode
    # env2 - created Cart Pole environment
    def simulateRandomStrategy(self):
        import gym
        import time
        import numpy as np
        env2 = gym.make('CartPole-v1')
        (currentState, _) = env2.reset()
        env2.render()
        # number of simulation episodes
        episodeNumber = 100
        # time steps in every episode
        timeSteps = 1000
        # sum of rewards in each episode
        sumRewardsEpisodes = []

        for episodeIndex in range(episodeNumber):
            rewardsSingleEpisode = []
            initial_state = env2.reset()
            print(episodeIndex)
            for timeIndex in range(timeSteps):
                random_action = env2.action_space.sample()
                observation, reward, terminated, truncated, info = env2.step(random_action)
                rewardsSingleEpisode.append(reward)
                if (terminated):
                    break
            sumRewardsEpisodes.append(np.sum(rewardsSingleEpisode))
        return sumRewardsEpisodes, env2
    ###########################################################################
    #    END - function for simulating random actions many times
    ###########################################################################


env=gym.make('CartPole-v1',render_mode='human')
#env = gym.make('CartPole-v1')
(state, _) = env.reset()
# env.render()
# env.close()

# here define the parameters for state discretization
upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
cartVelocityMin = -3
cartVelocityMax = 3
poleAngleVelocityMin = -10
poleAngleVelocityMax = 10
upperBounds[1] = cartVelocityMax
upperBounds[3] = poleAngleVelocityMax
lowerBounds[1] = cartVelocityMin
lowerBounds[3] = poleAngleVelocityMin

numberOfBinsPosition = 30
numberOfBinsVelocity = 30
numberOfBinsAngle = 30
numberOfBinsAngleVelocity = 30
numberOfBins = [numberOfBinsPosition, numberOfBinsVelocity, numberOfBinsAngle, numberOfBinsAngleVelocity]

# define the parameters
alpha = 0.1
gamma = 1
epsilon = 0.2
numberEpisodes = 15000

# create an object
Q1 = Q_Learning(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds)
# run the Q-Learning algorithm
Q1.simulateEpisodes()
# simulate the learned strategy
(obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()

plt.figure(figsize=(12, 5))
# plot the figure and adjust the plot parameters
plt.plot(Q1.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.yscale('log')
plt.show()
plt.savefig('convergence.png')

# close the environment
env1.close()
# get the sum of rewards
np.sum(obtainedRewardsOptimal)

# now simulate a random strategy
(obtainedRewardsRandom, env2) = Q1.simulateRandomStrategy()
plt.hist(obtainedRewardsRandom)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()

# run this several times and compare with a random learning strategy
(obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()
