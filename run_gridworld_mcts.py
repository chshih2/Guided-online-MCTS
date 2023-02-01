from environment.gridworld import GridWorld
from post_processing.graph_visualisation import GraphVisualisation
from q_model.qtable import QTable
from q_model.deep_qfunction import DeepQFunction
from q_model.qlearning import QLearning
from mcts_algo.single_agent_mcts import SingleAgentMCTS
from learning_algo.multi_armed_bandit import UpperConfidenceBounds, EpsilonGreedy
import numpy as np
import copy
import time


def plan_from_root(gridworld, qfunction, plot_graph, plot_policy):
    # search from root
    start_time = time.time()
    root_node = SingleAgentMCTS(gridworld, qfunction, UpperConfidenceBounds()).mcts(timeout=2.5)
    current_time = time.time()
    print("time spent", current_time - start_time)
    if plot_graph:
        gv = GraphVisualisation(max_level=6)
        graph = gv.single_agent_mcts_to_graph(root_node, filename="mcts")
        graph.format = 'png'
        graph.render('Graph', view=True)

    if plot_policy:
        gridworld.visualise_q_function(qfunction)
        policy = qfunction.extract_policy(gridworld)
        gridworld.visualise_policy(policy)


def plan_recursively(gridworld, qfunction, max_depth, simulation_depth, pretrained_q=False, plot_policy=False):
    # search recursively
    bandit = UpperConfidenceBounds()
    if pretrained_q:
        actions = gridworld.get_actions(gridworld.get_initial_state())
        bandit.total = len(actions)
        for action in actions:
            bandit.times_selected[action] = 1
    single_agent = SingleAgentMCTS(gridworld, qfunction, bandit, simulation_depth=simulation_depth)
    root_node = None
    start_time = time.time()
    while not root_node or not gridworld.is_terminal(current_state):
        done = False
        root_node = single_agent.mcts(timeout=10.03, max_depth=max_depth, root_node=root_node)
        current_state = root_node.state
        action, _ = qfunction.get_max_q(current_state, root_node.mdp.get_actions(current_state))
        tmp = copy.deepcopy(root_node.env)
        (next_state, reward) = tmp.execute(current_state, action)

        for (child, _) in root_node.children[action]:
            if next_state == ("terminal", "terminal") or child.state == ("terminal", "terminal"):
                done = True
                break
            if np.isclose(next_state, child.state).all():
                print(current_state, action)
                root_node = child
                break
        if done:
            current_time = time.time()
            print("reward", reward)
            print("time spent", current_time - start_time)
            break
    if plot_policy:
        gridworld.visualise_q_function(qfunction)
        policy = qfunction.extract_policy(gridworld)
        gridworld.visualise_policy(policy)


if __name__ == '__main__':
    plot_graph = True
    plot_policy = True
    deep_q = True
    recursive_planning = True
    lr = 5e-3
    max_depth = 5
    simulation_depth = 10
    pretrain_Q = False
    pretrain_Q_iter = 10

    gridworld = GridWorld(noise=0.0)
    if deep_q:
        qfunction = DeepQFunction(gridworld, state_space=len(gridworld.get_initial_state()), action_space=5,
                                  hiddem_dim=16, alpha=1e-3)
    else:
        qfunction = QTable()

    if recursive_planning:
        if pretrain_Q:

            QLearning(gridworld, EpsilonGreedy(), qfunction, alpha=1e-3).execute(episodes=pretrain_Q_iter)
            if plot_policy:
                gridworld.visualise_q_function(qfunction)
                policy = qfunction.extract_policy(gridworld)
                gridworld.visualise_policy_as_image(policy)
        from torch.optim import Adam

        qfunction.optimiser = Adam(qfunction.q_network.parameters(), lr=1e-3)
        plan_recursively(gridworld, qfunction, max_depth, simulation_depth,
                         pretrained_q=pretrain_Q, plot_policy=plot_policy)
    else:
        plan_from_root(gridworld, qfunction, plot_graph, plot_policy)
