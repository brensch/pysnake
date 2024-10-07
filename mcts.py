import numpy as np
import tensorflow as tf
import copy
import random
from typing import Tuple, List

from game_state import GameState, ACTIONS

NUM_ACTIONS = len(ACTIONS)

class MCTSNode:
    def __init__(self, state: GameState, parent: 'MCTSNode' = None, action: Tuple[int, ...] = None):
        self.state: GameState = state  # GameState object
        self.parent: 'MCTSNode' = parent
        self.action: Tuple[int, ...] = action  # The joint action that led to this node
        self.children: dict = {}  # key: joint action tuple, value: MCTSNode
        self.visit_count: int = 0
        self.total_value: np.ndarray = np.zeros(state.initial_num_snakes)  # Array per snake
        self.prior: float = 0.0  # For multi-agent, can use average of priors

    def is_expanded(self) -> bool:
        return len(self.children) > 0

def mcts_search(root: MCTSNode, model: tf.keras.Model, num_simulations: int, num_snakes: int) -> Tuple[Tuple[int, ...], float]:
    total_depth: int = 0  # To calculate average depth
    for _ in range(num_simulations):
        node = root
        search_path: List[MCTSNode] = [node]

        # Selection
        while node.is_expanded():
            joint_action, node = select_child(node)
            search_path.append(node)

        # Expansion
        expand_node(node, model, num_snakes)

        # Rollout
        value = rollout(node.state, num_snakes)

        # Backpropagation
        backpropagate(search_path, value)

        # Update total depth
        total_depth += len(search_path)

    # Calculate average depth
    avg_depth = total_depth / num_simulations if num_simulations > 0 else 0

    # After simulations, select the joint action with the highest visit count
    max_visits = -1
    best_joint_action = None
    for joint_action, child in root.children.items():
        if child.visit_count > max_visits:
            max_visits = child.visit_count
            best_joint_action = joint_action

    # Return the actions for each snake and average depth
    return best_joint_action, avg_depth

def select_child(node: MCTSNode) -> Tuple[Tuple[int, ...], MCTSNode]:
    """Selects the child with the highest UCB score."""
    total_visits = sum(child.visit_count for child in node.children.values())
    best_score = -float('inf')
    best_joint_action = None
    best_child = None

    for joint_action, child in node.children.items():
        # UCB formula adjusted for multi-agent scenario
        q_value = child.total_value / (child.visit_count + 1e-6)
        u_value = 2 * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
        ucb_score = np.sum(q_value) + u_value  # Sum over all snakes

        if ucb_score > best_score:
            best_score = ucb_score
            best_joint_action = joint_action
            best_child = child

    return best_joint_action, best_child

def expand_node(node: MCTSNode, model: tf.keras.Model, num_snakes: int) -> None:
    """Expands the node by adding all possible joint actions."""
    state_tensor = node.state.get_state_as_tensor()
    # Evaluate the state using the model
    outputs = model.predict(np.expand_dims(state_tensor, axis=0), verbose=0)
    policy_logits = outputs[:num_snakes]
    # We do not use the value from the model here since we are doing rollouts

    # Convert logits to probabilities
    policy_probs = [tf.nn.softmax(logits[0]).numpy() for logits in policy_logits]

    # Generate possible actions for alive snakes
    alive_snakes_indices = [i for i, alive in enumerate(node.state.alive_snakes) if alive]
    action_indices = []
    for i in range(num_snakes):
        if i in alive_snakes_indices:
            action_indices.append(list(range(NUM_ACTIONS)))
        else:
            # Dead snakes have no actions; use a placeholder action (e.g., action 0)
            action_indices.append([0])

    # Generate all possible joint actions (Cartesian product)
    joint_actions = list(np.array(np.meshgrid(*action_indices)).T.reshape(-1, num_snakes))

    for joint_action_indices in joint_actions:
        # Compute joint prior (average of individual priors)
        priors = []
        for i, action_idx in enumerate(joint_action_indices):
            if i in alive_snakes_indices:
                priors.append(policy_probs[i][action_idx])
            else:
                # Dead snakes have no prior
                priors.append(1.0)
        joint_prior = np.mean(priors)

        # Apply joint action to create new state
        new_state = copy.deepcopy(node.state)
        moves = []
        for i, action_idx in enumerate(joint_action_indices):
            if i in alive_snakes_indices:
                moves.append(ACTIONS[action_idx])
            else:
                moves.append(np.array([0, 0]))  # Dead snakes don't move
        new_state.apply_moves(np.array(moves))
        child_node = MCTSNode(new_state, parent=node, action=tuple(joint_action_indices))
        child_node.prior = joint_prior
        node.children[tuple(joint_action_indices)] = child_node

def rollout(state: GameState, num_snakes: int, max_rollout_depth: int = 10) -> np.ndarray:
    """Performs a random rollout from the given state."""
    current_state = copy.deepcopy(state)
    for _ in range(max_rollout_depth):
        if current_state.is_terminal():
            break
        moves = []
        for i in range(num_snakes):
            if current_state.alive_snakes[i]:
                safe_actions = current_state.get_safe_actions(i)
                if safe_actions:
                    action_idx = random.choice(safe_actions)
                else:
                    # No safe actions; choose randomly from all actions
                    action_idx = random.choice(current_state.get_valid_actions(i))
                moves.append(ACTIONS[action_idx])
            else:
                moves.append(np.array([0, 0]))  # Dead snake doesn't move
        current_state.apply_moves(np.array(moves))
    # Evaluate the terminal state from each snake's perspective
    return evaluate_state(current_state, num_snakes)

def evaluate_state(state: GameState, num_snakes: int) -> np.ndarray:
    """Returns an array of values for each snake: 1 for win, -1 for loss, 0 for tie."""
    values = np.zeros(num_snakes)
    alive_snakes_indices = [i for i, alive in enumerate(state.alive_snakes) if alive]

    if len(alive_snakes_indices) == 0:
        # All snakes are dead; it's a tie
        values[:] = 0
    elif len(alive_snakes_indices) == 1:
        # One snake remains; they win
        winner = alive_snakes_indices[0]
        values[:] = -1  # All others lose
        values[winner] = 1  # Winner gets +1
    else:
        # Game ended due to max depth; consider it a tie
        values[:] = 0
    return values

def backpropagate(search_path: List[MCTSNode], value: np.ndarray) -> None:
    """Propagates the evaluation value up the search path."""
    for node in reversed(search_path):
        node.visit_count += 1
        node.total_value += value  # value is an array per snake
        # For multi-agent games, we propagate the values as is.
