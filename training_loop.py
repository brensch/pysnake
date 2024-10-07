# training_loop.py

import tensorflow as tf
from tensorflow import keras
from train_utils import (
    self_play,
    train_model,
    save_model,
    load_latest_model,
    create_model,
    visualize_game,
)
from mcts import NUM_ACTIONS
import numpy as np
from game_state import GameState

# Initialize the model and optimizer
board_height, board_width = 11, 11
num_snakes = 2  # Set the number of snakes
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Load the latest model based on the parameters, or start from scratch
model = load_latest_model(num_snakes=num_snakes,
                          board_size=(board_height, board_width))

if model is None:
    model = create_model(board_height, board_width, num_snakes, NUM_ACTIONS)
    print("Starting with a new model.")
else:
    print("Model loaded successfully.")

# Training parameters
num_iterations = 3  # Number of training iterations
num_games_per_iteration = 2  # Games per iteration (set to 2 for testing)
num_simulations = 10  # MCTS simulations per move (set to 10 for testing)
batch_size = 5  # Batch size for training

training_losses = []  # To store training losses for visualization

# Training loop
for iteration in range(num_iterations):
    print(f"\nIteration {iteration + 1}/{num_iterations}")

    # Self-play to generate training data
    replay_buffer, game_summaries = self_play(model, num_games_per_iteration,
                                              num_simulations, num_snakes)

    # Summarize the games
    for summary in game_summaries:
        print(f"Game {summary['game_index']} summary:")
        print(f"  Steps: {summary['step_count']}")
        print(f"  Average MCTS Depth: {summary['avg_mcts_depth']:.2f}")
        print(f"  Winner: {'Snake ' + str(summary['winner']) if summary['winner'] is not None else 'Draw'}")
        # Visualize the game
        visualize_game(summary, num_snakes, (board_height, board_width))

    # Train the model
    num_samples = len(replay_buffer.buffer)
    num_batches = num_samples // batch_size
    print(f"Got {num_samples} samples.")

    if num_batches == 0:
        print("Not enough samples to form a batch.")
        continue

    losses = []
    for _ in range(num_batches):
        loss, policy_loss, value_loss = train_model(model, optimizer,
                                                    replay_buffer, batch_size,
                                                    num_snakes)
        losses.append(loss)

    avg_loss = np.mean(losses)
    training_losses.append(avg_loss)
    print(f"Average training loss: {avg_loss:.4f}")

    # Save the model after each iteration with training parameters
    save_model(model, iteration + 1, num_snakes=num_snakes,
               board_size=(board_height, board_width))

print("Training completed.")
