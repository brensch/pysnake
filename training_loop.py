# training_loop.py

import tensorflow as tf
from tensorflow import keras
from train_utils import (
    evaluate_model,
    get_model_path,
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

# Training parameters
board_height, board_width = 11, 11
num_snakes = 2  # Number of snakes
num_iterations = 30  # Number of training iterations
num_games_per_iteration = 10  # Games per iteration
num_simulations = 10  # MCTS simulations per move
batch_size = 20  # Batch size for training
evaluation_interval = 5  # Evaluate every 5 iterations
num_evaluation_games = 10  # Number of games for evaluation
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Load the latest model and optimizer
model = load_latest_model(num_snakes=num_snakes,
                        board_size=(board_height, board_width))

if model is None:
    model = create_model(board_height, board_width, num_snakes, NUM_ACTIONS)
    print("Starting with a new model.")
else:
    print("Model and optimizer loaded successfully.")

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
        # Visualize the game (optional)
        # visualize_game(summary, num_snakes, (board_height, board_width))

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

    # Save the model and optimizer after each iteration
    save_model(model, iteration + 1, num_snakes=num_snakes,
               board_size=(board_height, board_width))

    # Evaluate the model every iteration
    if iteration > 0:
        print(f"Evaluating model at iteration {iteration + 1}")
        previous_model_file = get_model_path(iteration, num_snakes, (board_height, board_width))
        print(f"playing against {previous_model_file}")
        previous_model = keras.models.load_model(previous_model_file)
        win_rate = evaluate_model(model, previous_model, num_evaluation_games, num_simulations, num_snakes)
        print(f"Win rate against previous model: {win_rate:.2%}")

print("Training completed.")
