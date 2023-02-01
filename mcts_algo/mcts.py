import copy
import math
import time
import random
from collections import defaultdict


class Node:
    # Record a unique node id to distinguish duplicated states
    next_node_id = 0

    # Records the number of times states have been visited
    visits = defaultdict(lambda: 0)

    def __init__(self, mdp, parent, state, qfunction, bandit, reward=0.0, action=None):
        self.mdp = mdp
        self.parent = parent
        self.state = state
        self.id = Node.next_node_id
        Node.next_node_id += 1

        # The Q function used to store state-action values
        self.qfunction = qfunction

        # A multi-armed bandit for this node
        self.bandit = bandit

        # The immediate reward received for reaching this state, used for backpropagation
        self.reward = reward

        # The action that generated this node
        self.action = action

    """ Select a node that is not fully expanded """

    def select(self): abstract

    """ Expand a node if it is not a terminal node """

    def expand(self): abstract

    """ Backpropogate the reward back to the parent node """

    def back_propagate(self, reward, child, depth): abstract

    """ Return the value of this node """

    def get_value(self):
        (_, max_q_value) = self.qfunction.get_max_q(
            self.state, self.mdp.get_actions(self.state)
        )
        return max_q_value

    """ Get the number of visits to this state """

    def get_visits(self):
        return Node.visits[self.state]


class MCTS:
    def __init__(self, mdp, qfunction, bandit, simulation_depth=None):
        self.mdp = mdp
        self.qfunction = qfunction
        self.bandit = bandit
        self.simulation_depth = 10000 if not simulation_depth else simulation_depth

    """
    Execute the MCTS algorithm from the initial state given, with timeout in seconds
    """

    def mcts(self, timeout=1, max_depth=100, root_node=None):
        if root_node is None:
            root_node = self.create_root_node()

        start_time = time.time()
        current_time = time.time()
        while current_time < start_time + timeout:

            # Find a state node to expand
            selected_node = root_node.select()
            if not self.mdp.is_terminal(selected_node):
                child = selected_node.expand()
                reward = self.simulate(child)
                depth = selected_node.back_propagate(reward, child, 0)
                if depth >= max_depth and root_node.is_fully_expanded():
                    # from graph_visualisation import GraphVisualisation
                    # gv = GraphVisualisation(max_level=6)
                    # graph = gv.single_agent_mcts_to_graph(root_node, filename="mcts")
                    # graph.format = 'png'
                    # graph.render('Graph', view=True)
                    # time.sleep(1)
                    break
            current_time = time.time()

        return root_node

    """ Create a root node representing an initial state """

    def create_root_node(self):
        abstract

    """ Choose a random action. Heustics can be used here to improve simulations. """

    def choose(self, state):
        return random.choice(self.mdp.get_actions(state))

    """ Simulate until a terminal state """

    def simulate(self, node):
        state = node.state
        cumulative_reward = 0.0
        depth = 0
        tmp = copy.deepcopy(node.mdp)
        while not tmp.is_terminal(state):
            # Choose an action to execute
            action = self.choose(state)

            # Execute the action
            (next_state, reward) = tmp.execute(state, action)
            # Discount the reward
            cumulative_reward += pow(self.mdp.get_discount_factor(), depth) * reward

            state = next_state

            depth += 1
            if depth == self.simulation_depth:
                cutoff_state_val = self.qfunction.get_max_q(state, self.mdp.get_actions(state))[1]
                cumulative_reward += pow(self.mdp.get_discount_factor(), depth + 1) * cutoff_state_val
                break

        return cumulative_reward
