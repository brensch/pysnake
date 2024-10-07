import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import copy
from typing import List, Tuple, Dict

# Import classes and functions
from game_state import GameState
from mcts import MCTSNode, mcts_search, ACTIONS, NUM_ACTIONS

class ReplayBuffer:
    def __init__(self):
        self.buffer: deque = deque(maxlen=10000)

    def add(self, state_tensor: np.ndarray, target_policies: List[np.ndarray], target_values: np.ndarray) -> None:
        self.buffer.append((state_tensor, target_policies, target_values))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, List[List[np.ndarray]], np.ndarray]:
        actual_batch_size = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, actual_batch_size)
        state_tensors, target_policies, target_values = zip(*samples)
        return np.array(state_tensors), list(target_policies), np.array(target_values)

def create_model(board_height: int, board_width: int, num_snakes: int, num_actions: int) -> keras.Model:
    """
    Creates a neural network model that takes the game state as input
    and outputs policy logits for each snake and a value estimate per snake.
    """
    input_shape = (board_height, board_width, num_snakes * 2 + 1)
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)

    # Policy Heads for each snake
    policy_logits = []
    for _ in range(num_snakes):
        policy_logit = layers.Dense(num_actions)(x)
        policy_logits.append(policy_logit)

    # Value Heads for each snake
    values = []
    for _ in range(num_snakes):
        value = layers.Dense(1, activation='tanh')(x)
        values.append(value)

    model = keras.Model(inputs=inputs, outputs=policy_logits + values)
    return model

def self_play(model: keras.Model, num_games: int, num_simulations: int, num_snakes: int) -> Tuple[ReplayBuffer, List[Dict]]:
    """
    Generates training data through self-play.
    """
    # Initialize a replay buffer
    replay_buffer = ReplayBuffer()
    game_summaries: List[Dict] = []  # To store data for each game

    for game_index in range(num_games):
        # Play a single game and collect data
        game_data, game_summary = self_play_game(model, num_simulations, num_snakes, game_index)
        game_summaries.append(game_summary)

        # Add game data to the replay buffer
        for state_tensor, target_policies_per_state, value in game_data:
            replay_buffer.add(state_tensor, target_policies_per_state, value)

    return replay_buffer, game_summaries

def self_play_game(model: keras.Model, num_simulations: int, num_snakes: int, game_index: int) -> Tuple[List[Tuple[np.ndarray, List[np.ndarray], np.ndarray]], Dict]:
    """
    Plays a single game for self-play and prints statistics for each step.
    """
    print(f"Game {game_index} started.")

    # Initialize game-specific variables
    game_states: List[np.ndarray] = []
    game_policies: List[List[np.ndarray]] = []
    game_values: List[np.ndarray] = []
    game_moves: List[Tuple[int, ...]] = []  # To store moves for visualization
    game_state_objects: List[GameState] = []  # To store GameState objects for visualization
    mcts_depths: List[float] = []  # To store the depth of MCTS trees
    game_over: bool = False
    step_count: int = 0
    winning_snake: int = None

    # Initialize GameState
    board_size = (11, 11)
    snake_bodies: List[np.ndarray] = []
    food_positions: List[np.ndarray] = []

    # Generate starting positions and orientations
    positions, orientations = generate_snake_start_positions(num_snakes, board_size)

    # Create snake bodies with segments stacked at the starting position
    for pos, orientation in zip(positions, orientations):
        body_length = 3  # Initial length of the snake
        body = np.array([pos for _ in range(body_length)])
        snake_bodies.append(body)

        # Place food near each snake within two squares
        food_x = (pos[0] + 2 * orientation[0]) % board_size[0]
        food_y = (pos[1] + 2 * orientation[1]) % board_size[1]
        food_positions.append(np.array([food_x, food_y]))

    # Place additional food in the center
    center_food = np.array([board_size[0] // 2, board_size[1] // 2])
    food_positions.append(center_food)

    game_state = GameState(board_size, snake_bodies, food_positions)

    while not game_over:
        # Store the current game state
        game_state_objects.append(copy.deepcopy(game_state))

        # Run MCTS to get the best joint action and collect MCTS depth
        root = MCTSNode(copy.deepcopy(game_state))
        best_joint_action_indices, avg_mcts_depth = mcts_search(root, model, num_simulations, num_snakes)
        mcts_depths.append(avg_mcts_depth)

        # Print details about the current MCTS node statistics
        print(f"\nStep {step_count + 1}")
        print(f"  Total MCTS Simulations: {num_simulations}")
        print(f"  Total Nodes Visited: {sum(child.visit_count for child in root.children.values())}")
        print(f"  Average MCTS Depth: {avg_mcts_depth:.2f}")
        for joint_action, child in root.children.items():
            print(f"  Action {joint_action.tolist()}: Visits = {child.visit_count}, Value = {child.total_value / max(child.visit_count, 1):.4f}")

        # Apply the joint action
        moves = [ACTIONS[action_idx] for action_idx in best_joint_action_indices]
        game_state.apply_moves(np.array(moves))
        game_moves.append(best_joint_action_indices)

        # Record the state
        state_tensor = game_state.get_state_as_tensor()
        game_states.append(state_tensor)

        # Compute target policies based on MCTS visit counts
        visit_counts_per_snake = [np.zeros(NUM_ACTIONS) for _ in range(num_snakes)]
        total_visits = 0

        for child in root.children.values():
            joint_action = child.action  # Tuple of action indices per snake
            visit_count = child.visit_count
            total_visits += visit_count
            for i in range(num_snakes):
                visit_counts_per_snake[i][joint_action[i]] += visit_count

        # Determine alive snakes
        alive_snakes_indices = [i for i, alive in enumerate(game_state.alive_snakes) if alive]

        # Normalize the visit counts to get target policies
        target_policies = []
        for i in range(num_snakes):
            if i in alive_snakes_indices:
                if total_visits > 0:
                    target_policy = visit_counts_per_snake[i] / total_visits
                else:
                    target_policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
            else:
                # Snake is dead; set target policy to zeros
                target_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
            target_policy = target_policy.reshape(NUM_ACTIONS)
            target_policies.append(target_policy)
        game_policies.append(target_policies)

        # Get the value estimates from the model for training
        outputs = model.predict(np.expand_dims(state_tensor, axis=0), verbose=0)
        value_estimates = np.array([output[0][0] for output in outputs[num_snakes:]])  # Extract values per snake
        game_values.append(value_estimates)

        # Check if the game is over
        if game_state.is_terminal():
            game_over = True
            alive_snakes_indices = [i for i, alive in enumerate(game_state.alive_snakes) if alive]
            if len(alive_snakes_indices) == 1:
                winning_snake = alive_snakes_indices[0]
            else:
                winning_snake = None

            # Assign rewards
            rewards = np.array([-1.0] * num_snakes)
            if winning_snake is not None:
                rewards[winning_snake] = 1.0
            game_values = [rewards for _ in range(len(game_states))]  # List of rewards per state

        step_count += 1

    # Prepare game data to return
    game_data = []
    for state_tensor, target_policies_per_state, values in zip(game_states,
                                                               game_policies,
                                                               game_values):
        game_data.append((state_tensor, target_policies_per_state, values))

    # Calculate average MCTS depth
    avg_game_mcts_depth = sum(mcts_depths) / len(mcts_depths) if mcts_depths else 0

    # Create game summary
    game_summary = {
        'game_index': game_index,
        'step_count': step_count,
        'avg_mcts_depth': avg_game_mcts_depth,
        'moves': game_moves,
        'game_states': game_state_objects,
        'winner': winning_snake if game_over else None
    }

    print(f"Game {game_index} finished. Steps: {step_count}, Average MCTS Depth: {avg_game_mcts_depth:.2f}")

    return game_data, game_summary


def train_model(model: keras.Model, optimizer: keras.optimizers.Optimizer, replay_buffer: ReplayBuffer, batch_size: int, num_snakes: int) -> Tuple[float, float, float]:
    state_tensors, target_policies, target_values = replay_buffer.sample(batch_size)

    # Initialize lists for each snake
    target_policies_per_snake = [[] for _ in range(num_snakes)]
    target_values_per_snake = [[] for _ in range(num_snakes)]

    # Iterate over the batch
    for policy_list, value_array in zip(target_policies, target_values):
        for i in range(num_snakes):
            policy = policy_list[i]
            policy = policy.reshape(NUM_ACTIONS)
            target_policies_per_snake[i].append(policy)
            target_values_per_snake[i].append(value_array[i])

    # Stack policies and values for each snake
    target_policies_stacked = [np.stack(target_policies_per_snake[i], axis=0).astype(np.float32) for i in range(num_snakes)]
    target_values_stacked = [np.array(target_values_per_snake[i]).astype(np.float32) for i in range(num_snakes)]

    with tf.GradientTape() as tape:
        outputs = model(state_tensors, training=True)
        policy_logits_list = outputs[:num_snakes]
        value_preds_list = outputs[num_snakes:]

        # Compute policy loss for each snake
        policy_loss = 0.0
        for i in range(num_snakes):
            logits = policy_logits_list[i]
            labels = target_policies_stacked[i]

            policy_loss += tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            )

        # Compute value loss for each snake
        value_loss = 0.0
        for i in range(num_snakes):
            preds = tf.squeeze(value_preds_list[i], axis=-1)
            labels = target_values_stacked[i]
            value_loss += tf.reduce_mean(tf.square(labels - preds))

        total_loss = policy_loss + value_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss.numpy(), policy_loss.numpy(), value_loss.numpy()

def save_model(model: keras.Model, iteration: int, num_snakes: int, board_size: Tuple[int, int]) -> None:
    """
    Save the model with the specified parameters embedded in the filename.
    """
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Convert the board size to a string format like '11x11'
    board_size_str = f'{board_size[0]}x{board_size[1]}'

    # Generate a unique filename with iteration, num_snakes, board_size, and timestamp
    unique_filename = os.path.join(
        models_dir,
        f'model_iteration_{iteration}_snakes_{num_snakes}_board_{board_size_str}_'
        f'{int(time.time())}.keras'
    )

    # Save the model with the unique filename
    model.save(unique_filename)
    print(f"Model saved as {unique_filename}")

def load_latest_model(num_snakes: int, board_size: Tuple[int, int]) -> keras.Model:
    """
    Load the latest model that matches the specified num_snakes and board_size.
    """
    import glob

    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Convert the board size to a string format like '11x11'
    board_size_str = f'{board_size[0]}x{board_size[1]}'

    # Find all model files that match the specified parameters
    search_pattern = os.path.join(
        models_dir,
        f'model_iteration_*_snakes_{num_snakes}_board_{board_size_str}_*.keras'
    )
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        print(f"No saved model found with num_snakes={num_snakes} "
              f"and board_size={board_size}")
        return None

    # Sort the matching files by timestamp (descending order) and get the latest one
    latest_model_file = max(matching_files, key=os.path.getctime)

    # Load the latest matching model
    model = keras.models.load_model(latest_model_file)
    print(f"Loaded model from {latest_model_file}")

    return model

def generate_snake_start_positions(num_snakes: int, board_size: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    positions: List[Tuple[int, int]] = []
    orientations: List[np.ndarray] = []
    center = (board_size[0] // 2, board_size[1] // 2)
    radius_x = board_size[0] // 2 - 1
    radius_y = board_size[1] // 2 - 1

    for i in range(num_snakes):
        angle = 2 * np.pi * i / num_snakes
        x = int(center[0] + radius_x * np.cos(angle))
        y = int(center[1] + radius_y * np.sin(angle))
        positions.append((x, y))

        # Orientation towards the center
        orientation = np.array([np.sign(center[0] - x), np.sign(center[1] - y)])
        # If the snake is at the center, set a default orientation
        if orientation[0] == 0 and orientation[1] == 0:
            orientation = np.array([0, 1])  # Facing up
        orientations.append(orientation)
    return positions, orientations

def visualize_game(summary: Dict, num_snakes: int, board_size: Tuple[int, int]) -> None:
    """
    Visualizes the game moves using ASCII art.
    """
    import time

    game_states: List[GameState] = summary['game_states']
    game_index = summary['game_index']

    print(f"Visualizing Game {game_index}:")

    for step_num, game_state in enumerate(game_states):
        print(f"Step {step_num + 1}:")
        game_state.visualize_board_ascii()
        time.sleep(0.05)  # Pause briefly between steps
