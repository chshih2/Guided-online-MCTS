import sys

sys.path.append("../")
from collections import defaultdict
import math
from q_model.qtable import QTable
import random


class MultiArmedBandit():
    """ Select an action for this state given from a list given a Q-function """

    def select(self, state, actions, qfunction):
        abstract

    """ Reset a multi-armed bandit to its initial configuration """

    def reset(self):
        self.__init__()


class EpsilonGreedy(MultiArmedBandit):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def reset(self):
        pass

    def select(self, state, actions, qfunction):
        # Select a random action with epsilon probability
        if random.random() < self.epsilon:
            return random.choice(actions)
        (arg_max_q, _) = qfunction.get_max_q(state, actions)
        return arg_max_q


class EpsilonDecreasing(MultiArmedBandit):
    def __init__(self, epsilon=0.2, alpha=0.999):
        self.epsilon_greedy_bandit = EpsilonGreedy(epsilon)
        self.initial_epsilon = epsilon
        self.alpha = alpha

    def reset(self):
        self.epsilon_greedy_bandit = EpsilonGreedy(self.initial_epsilon)

    def select(self, state, actions, qfunction):
        result = self.epsilon_greedy_bandit.select(state, actions, qfunction)
        self.epsilon_greedy_bandit.epsilon *= self.alpha
        return result


class Softmax(MultiArmedBandit):
    def __init__(self, tau=1.0):
        self.tau = tau

    def reset(self):
        pass

    def select(self, state, actions, qfunction):

        # calculate the denominator for the softmax strategy
        total = 0.0
        for action in actions:
            total += math.exp(qfunction.get_q_value(state, action) / self.tau)

        rand = random.random()
        cumulative_probability = 0.0
        result = None
        for action in actions:
            probability = (
                    math.exp(qfunction.get_q_value(state, action) / self.tau) / total
            )
            if cumulative_probability <= rand <= cumulative_probability + probability:
                result = action
            cumulative_probability += probability

        return result


class UpperConfidenceBounds(MultiArmedBandit):
    def __init__(self):
        self.total = 0
        # number of times each action has been chosen
        self.times_selected = {}

    def select(self, state, actions, qfunction):

        # First execute each action one time
        for action in actions:
            if action not in self.times_selected.keys():
                self.times_selected[action] = 1
                self.total += 1
                return action

        max_actions = []
        max_value = float("-inf")
        for action in actions:
            value = qfunction.get_q_value(state, action) + math.sqrt(
                (2 * math.log(self.total)) / self.times_selected[action]
            )
            if value > max_value:
                max_actions = [action]
                max_value = value
            elif value == max_value:
                max_actions += [action]

        # if there are multiple actions with the highest value
        # choose one randomly
        result = random.choice(max_actions)
        self.times_selected[result] = self.times_selected[result] + 1
        self.total += 1
        return result


""" Run a bandit algorithm for a number of episodes, with each episode
being a set length.
"""


def run_bandit(bandit, episodes=1, episode_length=1000, drift=True):
    # The actions available
    arms = [0, 1, 2, 3, 4]
    print("arms", arms)
    probabilities = [0.1, 0.3, 0.7, 0.2, 0.1]
    print("probabilities", probabilities)
    if drift:
        print("probabilities", [0.5, 0.2, 0.0, 0.3, 0.3])
    # A dummy state
    state = 1

    rewards = []
    actions = []
    for _ in range(0, episodes):
        bandit.reset()

        # The probability of receiving a payoff of 1 for each action
        probabilities = [0.1, 0.3, 0.7, 0.2, 0.1]

        # The number of times each arm has been selected
        times_selected = defaultdict(lambda: 0)
        qtable = QTable()

        episode_rewards = []
        episode_actions = []
        for step in range(0, episode_length):

            # Halfway through the episode, change the probabilities
            if drift and step == episode_length / 2:
                probabilities = [0.5, 0.2, 0.0, 0.3, 0.3]

            # Select an action using the bandit
            action = bandit.select(state, arms, qtable)

            # Get the reward for that action
            reward = 0
            if random.random() < probabilities[action]:
                reward = 5

            episode_rewards += [reward]
            episode_actions += [action]

            times_selected[action] = times_selected[action] + 1
            qtable.update(
                state,
                action,
                (reward / times_selected[action])
                - (qtable.get_q_value(state, action) / times_selected[action]),
            )

        rewards += [episode_rewards]
        actions += [episode_actions]

    return rewards, actions


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    drift = False
    epsilon_greedy, epsilon_greedy_action = run_bandit(EpsilonGreedy(epsilon=0.1), drift=drift)
    epsilon_decreasing, epsilon_decreasing_action = run_bandit(EpsilonDecreasing(alpha=0.99), drift=drift)
    softmax, softmax_action = run_bandit(Softmax(tau=1.0), drift=drift)
    ucb, ucb_action = run_bandit(UpperConfidenceBounds(), drift=drift)

    # plot reward and action
    # ucb and epsilon_decreasing are consistent in choosing the best arm
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(np.cumsum(epsilon_greedy[-1]))
    ax1.set_title("epsilon_greedy_reward")
    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(epsilon_greedy_action[-1], 'o')
    ax2.set_title("epsilon_greedy_action")

    ax3 = plt.subplot(4, 2, 3)
    ax3.plot(np.cumsum(epsilon_decreasing[-1]))
    ax3.set_title("epsilon_decreasing_reward")
    ax4 = plt.subplot(4, 2, 4)
    ax4.plot(epsilon_decreasing_action[-1], 'o')
    ax4.set_title("epsilon_decreasing_action")

    ax5 = plt.subplot(4, 2, 5)
    ax5.plot(np.cumsum(softmax[-1]))
    ax5.set_title("softmax_reward")
    ax6 = plt.subplot(4, 2, 6)
    ax6.plot(softmax_action[-1], 'o')
    ax6.set_title("softmax_action")

    ax7 = plt.subplot(4, 2, 7)
    ax7.plot(np.cumsum(ucb[-1]))
    ax7.set_title("ucb_reward")
    ax8 = plt.subplot(4, 2, 8)
    ax8.plot(ucb_action[-1], 'o')
    ax8.set_title("ucb_action")

    # plt.plot(epsilon_decreasing_action[-1],label="epsilon_decreasing")
    # plt.plot(softmax_action[-1],label="softmax")
    # plt.plot(ucb_action[-1],label="ucb")
    # plt.legend()
    plt.show()
