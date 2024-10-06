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

def create_model(board_height, board_width, num_snakes, num_actions):
    """
    Creates a neural network model that takes the game state as input
    and outputs policy logits for each snake and a value estimate.
    """
    input_shape = (board_height, board_width, num_snakes + 1)  # +1 for the food layer
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

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=10000)

    def add(self, state_tensor, target_policy, target_value):
        self.buffer.append((state_tensor, target_policy, target_value))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        state_tensors, target_policies, target_values = zip(*samples)
        return np.array(state_tensors), np.array(target_policies), np.array(target_values)

def train_model(model, optimizer, replay_buffer, batch_size, num_snakes):
    state_tensors, target_policies, target_values = replay_buffer.sample(batch_size)

    # Convert target_policies to list of arrays for each snake
    target_policies_per_snake = [[] for _ in range(num_snakes)]
    for policy_list in target_policies:
        for i in range(num_snakes):
            target_policies_per_snake[i].append(policy_list[i])

    # Stack policies for each snake
    target_policies_stacked = [np.array(target_policies_per_snake[i]) for i in range(num_snakes)]

    with tf.GradientTape() as tape:
        outputs = model(state_tensors, training=True)
        policy_logits_list = outputs[:num_snakes]
        values = outputs[-1]

        # Compute policy loss for each snake
        policy_loss = 0
        for i in range(num_snakes):
            policy_loss += tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=target_policies_stacked[i], logits=policy_logits_list[i])
            )

        # Compute value loss
        value_loss = tf.reduce_mean(tf.square(target_values - tf.squeeze(values)))

        total_loss = policy_loss + value_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss.numpy(), policy_loss.numpy(), value_loss.numpy()


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
        for i in range(num_snakes):
            # Place snakes at different starting positions
            # For simplicity, place snakes at corners
            if i == 0:
                snake_bodies.append(np.array([[5, 5], [5, 4], [5, 3]]))  # Center
            elif i == 1:
                snake_bodies.append(np.array([[0, 0], [0, 1], [0, 2]]))  # Top-left corner
            elif i == 2:
                snake_bodies.append(np.array([[10, 10], [10, 9], [10, 8]]))  # Bottom-right corner
            elif i == 3:
                snake_bodies.append(np.array([[0, 10], [1, 10], [2, 10]]))  # Bottom-left corner

        # Place food in the middle and near the snakes
        food_positions = [
            np.array([5, 5]),  # Center
            np.array([2, 2]),  # Near top-left corner
            np.array([8, 8]),  # Near bottom-right corner
            np.array([2, 8])   # Near bottom-left corner
        ]

        game_state = GameState(board_size, snake_bodies, food_positions)

        while not game_over:
            # Run MCTS to get the best joint action
            root = MCTSNode(copy.deepcopy(game_state))
            best_joint_action_indices = mcts_search(root, model, num_simulations, num_snakes)

            # Extract individual policies and actions
            state_tensor = game_state.get_state_as_tensor()

            # Get policy logits and value from the model
            outputs = model.predict(np.expand_dims(state_tensor, axis=0))
            policy_logits = outputs[:num_snakes]
            value = outputs[-1][0][0]  # Scalar value

            # Convert logits to probabilities
            policy_probs = [tf.nn.softmax(logits[0]).numpy() for logits in policy_logits]

            # Record the state and the policy for each snake
            game_states.append(state_tensor)
            game_policies.append(policy_probs)
            game_values.append(value)

            # Apply the joint action
            moves = [ACTIONS[action_idx] for action_idx in best_joint_action_indices]
            game_state.apply_moves(np.array(moves))

            # Check if the game is over
            if game_state.num_snakes == 0 or all(len(body) == 0 for body in game_state.snake_bodies):
                game_over = True
                rewards = [-1] * num_snakes  # All snakes lost
            elif game_state.num_snakes == 1:
                game_over = True
                rewards = [1 if i == 0 else -1 for i in range(num_snakes)]  # Snake 0 wins
            else:
                rewards = [0] * num_snakes  # Game continues

            # Backfill rewards if game over
            if game_over:
                # Assuming zero-sum game
                game_values = [rewards[i] for i in range(num_snakes)] * len(game_states)

        # Add game data to replay buffer
        for state_tensor, policy_probs_list, value_list in zip(game_states, game_policies, game_values):
            # For each snake, add to replay buffer
            for i in range(num_snakes):
                replay_buffer.add(state_tensor, policy_probs_list[i], value_list)

    return replay_buffer