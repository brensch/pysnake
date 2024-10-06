# mcts.py

import numpy as np
import tensorflow as tf
import copy

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
    def __init__(self, state, parent=None, action=None):
        self.state = state  # GameState object
        self.parent = parent
        self.action = action  # The joint action that led to this node
        self.children = {}  # key: joint action tuple, value: MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 0.0  # For multi-agent, can use average of priors

    def is_expanded(self):
        return len(self.children) > 0

def mcts_search(root, model, num_simulations, num_snakes):
    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # Selection
        while node.is_expanded():
            joint_action, node = select_child(node)
            search_path.append(node)

        # Expansion and Evaluation
        value = expand_and_evaluate(node, model, num_snakes)

        # Backpropagation
        backpropagate(search_path, value)

    # After simulations, select the joint action with the highest visit count
    max_visits = -1
    best_joint_action = None
    for joint_action, child in root.children.items():
        if child.visit_count > max_visits:
            max_visits = child.visit_count
            best_joint_action = joint_action

    # Return the actions for each snake
    return best_joint_action

def select_child(node):
    """
    Selects the child with the highest UCB score.
    """
    total_visits = sum(child.visit_count for child in node.children.values())
    best_score = -float('inf')
    best_joint_action = None
    best_child = None

    for joint_action, child in node.children.items():
        # UCB formula
        ucb_score = (child.total_value / (child.visit_count + 1e-6)) + \
                    2 * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
        if ucb_score > best_score:
            best_score = ucb_score
            best_joint_action = joint_action
            best_child = child

    return best_joint_action, best_child

def expand_and_evaluate(node, model, num_snakes):
    """
    Expands the node by adding all possible joint actions and evaluates the state using the neural network.
    """
    state_tensor = node.state.get_state_as_tensor()
    state_tensor = np.expand_dims(state_tensor, axis=0)  # Add batch dimension

    # Get policy logits and value from the model
    outputs = model.predict(state_tensor)
    policy_logits = outputs[:num_snakes]
    value = outputs[-1][0][0]  # Extract scalar value

    # Convert logits to probabilities
    policy_probs = [tf.nn.softmax(logits[0]).numpy() for logits in policy_logits]

    # Generate all possible joint actions (cartesian product)
    action_indices = [list(range(NUM_ACTIONS)) for _ in range(num_snakes)]
    joint_actions = list(np.array(np.meshgrid(*action_indices)).T.reshape(-1, num_snakes))

    for joint_action_indices in joint_actions:
        # Compute joint prior (average of individual priors)
        priors = [policy_probs[i][action_idx] for i, action_idx in enumerate(joint_action_indices)]
        joint_prior = np.mean(priors)

        # Apply joint action to create new state
        new_state = copy.deepcopy(node.state)
        moves = [ACTIONS[action_idx] for action_idx in joint_action_indices]
        new_state.apply_moves(np.array(moves))
        child_node = MCTSNode(new_state, parent=node, action=tuple(joint_action_indices))
        child_node.prior = joint_prior
        node.children[tuple(joint_action_indices)] = child_node

    return value

def backpropagate(search_path, value):
    """
    Propagates the evaluation value up the search path.
    """
    for node in reversed(search_path):
        node.visit_count += 1
        node.total_value += value
        value = -value  # For zero-sum games (assuming symmetric competition)
