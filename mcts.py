# mcts.py

import numpy as np
import tensorflow as tf
import copy

# Import GameState from your game_state module
from game_state import GameState  

# Define ACTIONS and NUM_ACTIONS
ACTIONS = {
    0: np.array([0, 1]),   # Up
    1: np.array([0, -1]),  # Down
    2: np.array([-1, 0]),  # Left
    3: np.array([1, 0])    # Right
}
NUM_ACTIONS = len(ACTIONS)

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state: GameState = state  # GameState object
        self.parent = parent
        self.children = {}  # key: action, value: MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 0.0  # From neural network policy head

    def is_expanded(self):
        return len(self.children) > 0

def mcts_search(root, model, num_simulations):
    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # Selection
        while node.is_expanded():
            action, node = select_child(node)
            search_path.append(node)

        # Expansion and Evaluation
        value = expand_and_evaluate(node, model)

        # Backpropagation
        backpropagate(search_path, value)

    # After simulations, select the action with the highest visit count
    max_visits = -1
    best_action = None
    for action, child in root.children.items():
        if child.visit_count > max_visits:
            max_visits = child.visit_count
            best_action = action
    return best_action

def select_child(node):
    """
    Selects the child with the highest UCB score.
    """
    total_visits = sum(child.visit_count for child in node.children.values())
    best_score = -float('inf')
    best_action = None
    best_child = None

    for action, child in node.children.items():
        # UCB formula
        ucb_score = (child.total_value / (child.visit_count + 1e-6)) + \
                    2 * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
        if ucb_score > best_score:
            best_score = ucb_score
            best_action = action
            best_child = child

    return best_action, best_child

def expand_and_evaluate(node, model):
    """
    Expands the node by adding all possible actions and evaluates the state using the neural network.
    """
    state_tensor = node.state.get_state_as_tensor()
    state_tensor = np.expand_dims(state_tensor, axis=0)  # Add batch dimension

    # Get policy and value from the model
    policy_logits, value = model.predict(state_tensor)
    policy_probs = tf.nn.softmax(policy_logits[0]).numpy()
    value = value[0][0]

    # Expand the node with new children
    for action_idx, action_prob in enumerate(policy_probs):
        # Apply action to create new state
        new_state = copy.deepcopy(node.state)
        move = np.array([ACTIONS[action_idx]])
        # For simplicity, assume only one snake (the agent)
        new_state.apply_moves(move)
        child_node = MCTSNode(new_state, parent=node)
        child_node.prior = action_prob
        node.children[action_idx] = child_node

    return value

def backpropagate(search_path, value):
    """
    Propagates the evaluation value up the search path.
    """
    for node in reversed(search_path):
        node.visit_count += 1
        node.total_value += value
        value = -value  # For zero-sum games
