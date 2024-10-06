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
    and outputs policy logits and value estimate.
    """
    input_shape = (board_height, board_width, num_snakes + 1)  # +1 for the food layer
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)

    # Policy Head
    policy_logits = layers.Dense(num_actions)(x)

    # Value Head
    value = layers.Dense(1, activation='tanh')(x)

    model = keras.Model(inputs=inputs, outputs=[policy_logits, value])
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

def train_model(model, optimizer, replay_buffer, batch_size):
    state_tensors, target_policies, target_values = replay_buffer.sample(batch_size)

    with tf.GradientTape() as tape:
        policy_logits, values = model(state_tensors, training=True)

        # Compute losses
        policy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=target_policies, logits=policy_logits)
        )
        value_loss = tf.reduce_mean(tf.square(target_values - tf.squeeze(values)))

        total_loss = policy_loss + value_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss.numpy(), policy_loss.numpy(), value_loss.numpy()

def self_play(model, num_games, num_simulations):
    replay_buffer = ReplayBuffer()
    for game in range(num_games):
        print(f"Starting game {game + 1}/{num_games}")
        game_states = []
        game_policies = []
        game_values = []
        game_over = False

        # Initialize GameState
        board_size = (5, 5)
        snake_bodies = [np.array([[2, 2], [2, 1], [2, 0]])]  # Agent's snake
        food_positions = np.array([[random.randint(0, 4), random.randint(0, 4)]])
        game_state = GameState(board_size, snake_bodies, food_positions)

        while not game_over:
            # Run MCTS to get the best action
            root = MCTSNode(copy.deepcopy(game_state))
            best_action_idx = mcts_search(root, model, num_simulations)

            # Get the visit counts for each action to compute the target policy
            visit_counts = np.array([root.children[action].visit_count if action in root.children else 0
                                     for action in range(NUM_ACTIONS)])
            total_visits = np.sum(visit_counts)
            if total_visits > 0:
                target_policy = visit_counts / total_visits
            else:
                target_policy = np.ones(NUM_ACTIONS) / NUM_ACTIONS  # Equal probability if no visits

            # Record the state and the policy
            state_tensor = game_state.get_state_as_tensor()
            game_states.append(state_tensor)
            game_policies.append(target_policy)

            # Apply the selected action
            move = ACTIONS[best_action_idx]
            game_state.apply_moves(np.array([move]))

            # Check if the game is over (e.g., snake is dead)
            if game_state.num_snakes == 0 or len(game_state.snake_bodies[0]) == 0:
                game_over = True
                reward = -1  # Loss
            elif len(game_state.snake_bodies[0]) >= board_size[0] * board_size[1]:
                game_over = True
                reward = 1  # Win
            else:
                reward = 0  # Game continues

            game_values.append(reward)

        # After the game, backfill the rewards
        cumulative_reward = 0
        for i in reversed(range(len(game_values))):
            cumulative_reward = game_values[i] + 0.99 * cumulative_reward
            game_values[i] = cumulative_reward

        # Add game data to replay buffer
        for state_tensor, target_policy, target_value in zip(game_states, game_policies, game_values):
            replay_buffer.add(state_tensor, target_policy, target_value)

    return replay_buffer
