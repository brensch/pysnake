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
        self.prior: float = 0.0  # For multi-agent, can use product of priors

    def is_expanded(self) -> bool:
        return len(self.children) > 0

def mcts_search(root: MCTSNode, model: tf.keras.Model, num_simulations: int, num_snakes: int) -> Tuple[Tuple[int, ...], float]:
    total_depth: int = 0  # To calculate average depth
    for _ in range(num_simulations):
        node = root
        search_path: List[MCTSNode] = [node]

        # Selection
        while node.is_expanded() and not node.state.is_terminal():
            node = select_child(node, num_snakes)
            search_path.append(node)

        # Expansion and Evaluation
        value = expand_and_evaluate(node, model, num_snakes)

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

def select_child(node: MCTSNode, num_snakes: int) -> MCTSNode:
    """Selects the child node with the highest PUCT value."""
    total_visits = node.visit_count
    best_score = -float('inf')
    best_child = None

    c_puct = 1.0  # Exploration constant; adjust as needed

    for child in node.children.values():
        # Compute PUCT score for each snake individually and sum them up
        q_values = child.total_value / (child.visit_count + 1e-6)  # Avoid division by zero
        u_values = c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
        # For multi-agent, we can sum the individual PUCT values
        puct_values = q_values + u_values  # Element-wise addition
        total_puct = np.sum(puct_values)

        if total_puct > best_score:
            best_score = total_puct
            best_child = child

    return best_child

def expand_and_evaluate(node: MCTSNode, model: tf.keras.Model, num_snakes: int) -> np.ndarray:
    """Expands the node and evaluates it using the neural network or game outcome if terminal."""
    if node.state.is_terminal():
        # Terminal node: evaluate the state directly
        values = evaluate_state(node.state, num_snakes)
        return values
    else:
        state_tensor = node.state.get_state_as_tensor()
        outputs = model.predict(np.expand_dims(state_tensor, axis=0), verbose=0)

        # Extract the value estimates for each snake
        values = np.array([output[0][0] for output in outputs[num_snakes:]])  # Values per snake

        # Convert logits to probabilities
        policy_logits = outputs[:num_snakes]
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
            # Compute joint prior (product of individual priors)
            priors = []
            for i, action_idx in enumerate(joint_action_indices):
                if i in alive_snakes_indices:
                    priors.append(policy_probs[i][action_idx])
                else:
                    priors.append(1.0)  # Dead snakes don't contribute to prior
            joint_prior = np.prod(priors)  # Use product for joint prior

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

        return values  # Return the value estimates per snake

def rollout(state: GameState, num_snakes: int, max_rollout_depth: int = 20) -> np.ndarray:
    """Performs a random rollout from the given state."""
    current_state = copy.deepcopy(state)
    for _ in range(max_rollout_depth):
        if current_state.is_terminal():
            break
        moves = []
        for i in range(num_snakes):
            if current_state.alive_snakes[i]:
                # Get all possible actions
                actions = list(ACTIONS.keys())
                action_idx = random.choice(actions)
                moves.append(ACTIONS[action_idx])
            else:
                moves.append(np.array([0, 0]))  # Dead snake doesn't move
        current_state.apply_moves(np.array(moves))
    # Evaluate the terminal state from each snake's perspective
    return evaluate_state(current_state, num_snakes)

def evaluate_state(state: GameState, num_snakes: int) -> np.ndarray:
    """Returns an array of values for each snake based on the final state."""
    values = np.zeros(num_snakes)
    alive_snakes_indices = [i for i, alive in enumerate(state.alive_snakes) if alive]

    if len(alive_snakes_indices) == 0:
        # All snakes are dead; consider it a tie or zero reward
        values[:] = 0.0
    elif len(alive_snakes_indices) == 1:
        # One snake remains; they win
        winner = alive_snakes_indices[0]
        values[winner] = 1.0  # Winner gets +1
        for i in range(num_snakes):
            if i != winner:
                values[i] = -1.0  # Losers get -1
    else:
        # Game ended due to max depth or multiple snakes alive; assign zero reward
        values[:] = 0.0
    return values

def backpropagate(search_path: List[MCTSNode], value: np.ndarray) -> None:
    """Propagates the evaluation value up the search path."""
    for node in reversed(search_path):
        node.visit_count += 1
        node.total_value += value  # value is an array per snake
        # Do not invert the values, as we are not assuming a zero-sum game
        # Each snake's value is independent and should be propagated as is
