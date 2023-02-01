import copy

import torch
import torch.nn as nn
from q_model.qfunction import QFunction
from torch.optim import Adam
from acrobot.train_acrobot import DQN


class DeepQFunction(QFunction):
    """ A neural network to represent the Q-function.
        This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(
            self, mdp, state_space, action_space, hiddem_dim=64, alpha=0.001, pretrained_dir=None
    ) -> None:
        self.mdp = mdp
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha

        # Create a sequential neural network to represent the Q function
        if not pretrained_dir:
            self.q_network = nn.Sequential(
                nn.Linear(in_features=self.state_space, out_features=hiddem_dim),
                nn.ReLU(),
                nn.Linear(in_features=hiddem_dim, out_features=hiddem_dim),
                nn.ReLU(),
                nn.Linear(in_features=hiddem_dim, out_features=self.action_space),
            )
        else:
            self.q_network = DQN(state_space, action_space)
            self.q_network.load_state_dict(torch.load(pretrained_dir))
            # torch.load(pretrained_dir)

        self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha)

        self.target_net = copy.deepcopy(self.q_network)

        # A two-way mapping from actions to integer IDs for ordinal encoding
        actions = self.mdp.get_actions()
        self.action_to_id = {actions[i]: i for i in range(len(actions))}
        self.id_to_action = {
            action_id: action for action, action_id in self.action_to_id.items()
        }

    def update(self, state, action, delta):
        # Train the network based on the squared error.

        self.optimiser.zero_grad()  # Reset gradients to zero
        (delta ** 2).backward()  # Back-propagate the loss through the network
        self.optimiser.step()  # Do a gradient descent step with the optimiser

    def get_q_value(self, state, action, for_max=False):
        # Convert the state into a tensor
        state = self.encode_state(state)
        q_values = self.q_network(state)

        q_value = q_values[self.action_to_id[action]]  # Index q-values by action

        return q_value

    def get_max_q(self, state, actions):
        # Convert the state into a tensor
        state = torch.as_tensor(self.encode_state(state), dtype=torch.float32)

        # Since we have a multi-headed q-function, we only need to pass through the network once
        # call torch.no_grad() to avoid tracking the gradients for this network forward pass
        q_values = self.q_network(state)
        arg_max_q = None
        max_q = float("-inf")
        for action in actions:
            value = q_values[self.action_to_id[action]].item()
            if max_q < value:
                arg_max_q = action
                max_q = value
        return (arg_max_q, max_q)

    """
    Turn the state into a tensor.
    """

    @staticmethod
    def encode_state(state):
        if state == ("terminal", "terminal"):
            state = (-1, -1)
        return torch.as_tensor(state, dtype=torch.float32)
