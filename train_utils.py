import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import copy
from multiprocessing import Process, Manager, current_process

# Import classes and functions
from game_state import GameState
from mcts import MCTSNode, mcts_search, ACTIONS, NUM_ACTIONS

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=10000)

    def add(self, state_tensor, target_policy, target_value):
        self.buffer.append((state_tensor, target_policy, target_value))

    def sample(self, batch_size):
        actual_batch_size = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, actual_batch_size)
        state_tensors, target_policies, target_values = zip(*samples)
        return np.array(state_tensors), list(target_policies), np.array(target_values)

def create_model(board_height, board_width, num_snakes, num_actions):
    """
    Creates a neural network model that takes the game state as input
    and outputs policy logits for each snake and a value estimate.
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

    # Value Head
    value = layers.Dense(1, activation='tanh')(x)

    model = keras.Model(inputs=inputs, outputs=policy_logits + [value])
    return model

def inference_server(request_queue, response_queue, model):
    """
    Runs the inference server that listens for inference requests and returns predictions.
    """
    import time
    import sys

    print(f"Inference server started with PID: {os.getpid()}")

    # Variables to track inference rate
    inference_count = 0
    start_time = time.time()
    last_update_time = start_time

    while True:
        item = request_queue.get()
        if item is None:
            print("Inference server received shutdown signal.")
            break  # Exit the server loop
        idx, state_tensor = item
        # Perform inference
        outputs = model.predict(np.expand_dims(state_tensor, axis=0), verbose=0)
        response_queue.put((idx, outputs))

        # Update inference count
        inference_count += 1

        # Check if one second has passed since the last update
        current_time = time.time()
        if current_time - last_update_time >= 1.0:
            # Calculate inferences per second
            elapsed_time = current_time - start_time
            inferences_per_second = inference_count / elapsed_time

            # Print the rate, overwriting the previous line
            print(f"\rInferences per second: {inferences_per_second:.2f}", end='', flush=True)

            # Update last update time
            last_update_time = current_time

    print("\nInference server shutting down.")


def self_play_game(args):
    """
    Plays a single game for self-play. This function runs in a separate process.
    """
    request_queue, response_queue, num_simulations, num_snakes, game_index = args

    # Initialize game-specific variables
    game_states = []
    game_policies = []
    game_values = []
    game_over = False

    # Initialize GameState
    board_size = (11, 11)
    snake_bodies = []
    food_positions = []

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
        # Run MCTS to get the best joint action
        root = MCTSNode(copy.deepcopy(game_state))
        best_joint_action_indices = mcts_search(root, request_queue, response_queue,
                                                num_simulations, num_snakes)

        # Apply the joint action
        moves = [ACTIONS[action_idx] for action_idx in best_joint_action_indices]
        game_state.apply_moves(np.array(moves))

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

        # Send inference request to get the value estimate
        idx = current_process().pid  # Use process PID as unique identifier
        state_tensor = game_state.get_state_as_tensor()
        request_queue.put((idx, state_tensor))
        # Wait for the response
        while True:
            resp_idx, outputs = response_queue.get()
            if resp_idx == idx:
                break
            else:
                # Put back the response if it's not for this process
                response_queue.put((resp_idx, outputs))
        value = outputs[-1][0][0]  # Scalar value
        game_values.append(value)

        # Check if the game is over
        if not any(game_state.alive_snakes):
            game_over = True
            rewards = [-1] * num_snakes  # All snakes lost
        elif sum(game_state.alive_snakes) == 1:
            game_over = True
            winning_snake = np.argmax(game_state.alive_snakes)
            rewards = [-1] * num_snakes
            rewards[winning_snake] = 1  # Winning snake gets +1 reward
        else:
            rewards = [0] * num_snakes  # Game continues

        # Assign rewards if the game is over
        if game_over:
            # Assuming zero-sum game
            game_values = [rewards[i] for i in range(num_snakes)] * len(game_states)

    # Prepare game data to return
    game_data = []
    for state_tensor, target_policies_per_state, value in zip(game_states,
                                                              game_policies,
                                                              game_values):
        game_data.append((state_tensor, target_policies_per_state, value))

    return game_data

def self_play(model, num_games, num_simulations, num_snakes):
    """
    Generates training data through self-play using multiple processes.
    """
    # Create a Manager to manage shared objects
    with Manager() as manager:
        request_queue = manager.Queue()
        response_queue = manager.Queue()

        # Start the inference server in a separate process
        server_process = Process(target=inference_server,
                                 args=(request_queue, response_queue, model))
        server_process.start()

        # Prepare arguments for each game
        args_list = [(request_queue, response_queue, num_simulations, num_snakes, i)
                     for i in range(num_games)]

        # Use multiprocessing Pool for parallel execution
        from multiprocessing import Pool
        with Pool() as pool:
            results = pool.map(self_play_game, args_list)

        # Combine results into a single replay buffer
        replay_buffer = ReplayBuffer()
        for game_data in results:
            for state_tensor, target_policies_per_state, value in game_data:
                replay_buffer.add(state_tensor, target_policies_per_state, value)

        # Send shutdown signal to the inference server
        request_queue.put(None)
        server_process.join()

    return replay_buffer

def train_model(model, optimizer, replay_buffer, batch_size, num_snakes):
    state_tensors, target_policies, target_values = replay_buffer.sample(batch_size)

    # Initialize lists for each snake
    target_policies_per_snake = [[] for _ in range(num_snakes)]

    # Iterate over the batch
    for policy_list in target_policies:
        for i in range(num_snakes):
            policy = policy_list[i]
            policy = policy.reshape(NUM_ACTIONS)
            target_policies_per_snake[i].append(policy)

    # Stack policies for each snake
    target_policies_stacked = [np.stack(target_policies_per_snake[i], axis=0)
                               .astype(np.float32) for i in range(num_snakes)]

    with tf.GradientTape() as tape:
        outputs = model(state_tensors, training=True)
        policy_logits_list = outputs[:num_snakes]
        values = outputs[-1]

        # Compute policy loss for each snake
        policy_loss = 0
        for i in range(num_snakes):
            logits = policy_logits_list[i]
            labels = target_policies_stacked[i]

            policy_loss += tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            )

        # Compute value loss
        value_loss = tf.reduce_mean(tf.square(target_values - tf.squeeze(values)))

        total_loss = policy_loss + value_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss.numpy(), policy_loss.numpy(), value_loss.numpy()

def save_model(model, iteration, num_snakes, board_size):
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

def load_latest_model(num_snakes, board_size):
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

def generate_snake_start_positions(num_snakes, board_size):
    positions = []
    orientations = []
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
