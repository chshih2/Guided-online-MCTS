import copy
import numpy as np
from mcts_algo.mcts import Node
from mcts_algo.mcts import MCTS


class SingleAgentNode(Node):
    def __init__(
            self,
            mdp,
            parent,
            state,
            qfunction,
            bandit,
            reward=0.0,
            action=None
    ):
        super().__init__(mdp, parent, state, qfunction, bandit, reward, action)

        # A dictionary from actions to a set of node-probability pairs
        self.children = {}
        self.env = copy.deepcopy(mdp)

    """ Return true if and only if all child actions have been expanded """

    def is_fully_expanded(self):
        valid_actions = self.mdp.get_actions(self.state)
        if len(valid_actions) == len(self.children):
            return True
        else:
            return False

    """ Select a node that is not fully expanded """

    def select(self):
        # if len(list(self.children.keys()))==0 or self.mdp.is_terminal(self.state): # add constraints here, do not need to explore all
        if not self.is_fully_expanded() or self.mdp.is_terminal(
                self.state):  # add constraints here, do not need to explore all
            return self
        else:
            actions = list(self.children.keys())
            action = self.bandit.select(self.state, actions, self.qfunction)
            return self.get_outcome_child(action).select()

    """ Expand a node if it is not a terminal node """

    def expand(self):
        if not self.mdp.is_terminal(self.state):
            # Randomly select an unexpanded action to expand
            actions = self.mdp.get_actions(self.state) - self.children.keys()
            # action = random.choice(list(actions))
            action = self.bandit.select(self.state, actions, self.qfunction)
            self.children[action] = []
            return self.get_outcome_child(action)
        return self

    """ Backpropogate the reward back to the parent node """

    def back_propagate(self, reward, child, depth):
        action = child.action

        q_value = self.qfunction.get_q_value(self.state, action)
        delta = reward - q_value

        self.qfunction.update(self.state, action, delta)

        depth += 1

        if self.parent != None:
            depth = self.parent.back_propagate(self.reward + reward, self, depth)
        return depth

    """ Simulate the outcome of an action, and return the child node """

    def get_outcome_child(self, action):
        # Choose one outcome based on transition probabilities
        tmp = copy.deepcopy(self.env)
        (next_state, reward) = tmp.execute(self.state, action)

        # Find the corresponding state and return if this already exists
        for (child, _) in self.children[action]:
            if next_state == ("terminal", "terminal"):
                next_state = (-1, -1)
            if child.state == ("terminal", "terminal"):
                child.state = (-1, -1)
            if np.isclose(next_state, child.state).all():
                return child

        # This outcome has not occured from this state-action pair previously
        new_child = SingleAgentNode(
            tmp, self, next_state, self.qfunction, self.bandit, reward, action
        )

        # Find the probability of this outcome (only possible for model-based) for printing the search tree
        probability = 1.0
        self.children[action] += [(new_child, probability)]
        return new_child


class SingleAgentMCTS(MCTS):
    def create_root_node(self):
        return SingleAgentNode(
            self.mdp, None, self.mdp.get_initial_state(), self.qfunction, self.bandit
        )
