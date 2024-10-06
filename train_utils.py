# train_utils.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import copy

import matplotlib.pyplot as plt

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
    input_shape = (board_height, board_width, num_snakes * 2 + 1)  # 2 channels per snake + 1 for the food layer
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

def self_play(model, num_games, num_simulations, num_snakes):
    replay_buffer = ReplayBuffer()
    for game in range(num_games):
        print(f"Starting game {game + 1}/{num_games}")
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
            best_joint_action_indices = mcts_search(root, model, num_simulations, num_snakes)

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
                        target_policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS  # Uniform distribution if no visits
                else:
                    # Snake is dead; set target policy to zeros
                    target_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
                # Ensure target_policy is an array of shape (NUM_ACTIONS,)
                target_policy = np.array(target_policy, dtype=np.float32).reshape(NUM_ACTIONS)
                target_policies.append(target_policy)
            game_policies.append(target_policies)

            # Get the value from the model for training (optional)
            outputs = model.predict(np.expand_dims(state_tensor, axis=0))
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

            # Backfill rewards if game over
            if game_over:
                # Assuming zero-sum game
                game_values = [rewards[i] for i in range(num_snakes)] * len(game_states)

        # Add game data to replay buffer
        for state_tensor, target_policies_per_state, value in zip(game_states, game_policies, game_values):
            replay_buffer.add(state_tensor, target_policies_per_state, value)

    return replay_buffer

def train_model(model, optimizer, replay_buffer, batch_size, num_snakes):
    state_tensors, target_policies, target_values = replay_buffer.sample(batch_size)

    # Initialize lists for each snake
    target_policies_per_snake = [[] for _ in range(num_snakes)]

    # Iterate over the batch
    for idx, policy_list in enumerate(target_policies):
        for i in range(num_snakes):
            policy = policy_list[i]
            # Ensure policy is a NumPy array of shape (NUM_ACTIONS,)
            policy = np.array(policy, dtype=np.float32).reshape(NUM_ACTIONS)
            target_policies_per_snake[i].append(policy)

    # Stack policies for each snake
    target_policies_stacked = [np.stack(target_policies_per_snake[i], axis=0).astype(np.float32) for i in range(num_snakes)]

    with tf.GradientTape() as tape:
        outputs = model(state_tensors, training=True)
        policy_logits_list = outputs[:num_snakes]
        values = outputs[-1]

        # Compute policy loss for each snake
        policy_loss = 0
        for i in range(num_snakes):
            # Ensure shapes match
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

# The main training loop can be included in your script or notebook
