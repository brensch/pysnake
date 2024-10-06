import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import copy
import random
from collections import deque
import matplotlib.pyplot as plt

ACTIONS = {
    0: np.array([0, 1]),   # Up
    1: np.array([0, -1]),  # Down
    2: np.array([-1, 0]),  # Left
    3: np.array([1, 0])    # Right
}
NUM_ACTIONS = len(ACTIONS)


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

def mcts_search(root, model, num_simulations):
    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # Selection
        while node.is_expanded():
            action, node = select_child(node)
            search_path.append(node)

        # Expansion
        value = expand_and_evaluate(node, model)

        # Backpropagation
        backpropagate(search_path, value)

    # After simulations, select the action with the highest visit count
    max_visits = -1
    best_action = None
    for action, child in root.children.items():
        if child.visit_count > max_visits:
            max_visits = child.visit_count
            best_action = action
    return best_action

def select_child(node):
    """
    Selects the child with the highest UCB score.
    """
    total_visits = sum(child.visit_count for child in node.children.values())
    best_score = -float('inf')
    best_action = None
    best_child = None

    for action, child in node.children.items():
        ucb_score = (child.total_value / (child.visit_count + 1e-6)) + \
                    2 * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
        if ucb_score > best_score:
            best_score = ucb_score
            best_action = action
            best_child = child

    return best_action, best_child

def expand_and_evaluate(node, model):
    """
    Expands the node by adding all possible actions and evaluates the state using the neural network.
    """
    state_tensor = node.state.get_state_as_tensor()
    state_tensor = np.expand_dims(state_tensor, axis=0)  # Add batch dimension

    # Get policy and value from the model
    policy_logits, value = model.predict(state_tensor)
    policy_probs = tf.nn.softmax(policy_logits[0]).numpy()
    value = value[0][0]

    # Expand the node with new children
    for action_idx, action_prob in enumerate(policy_probs):
        # Apply action to create new state
        new_state = copy.deepcopy(node.state)
        move = np.array([ACTIONS[action_idx]])
        # For simplicity, assume only one snake (the agent)
        new_state.apply_moves(move)
        child_node = MCTSNode(new_state, parent=node)
        child_node.prior = action_prob
        node.children[action_idx] = child_node

    return value

def backpropagate(search_path, value):
    """
    Propagates the evaluation value up the search path.
    """
    for node in reversed(search_path):
        node.visit_count += 1
        node.total_value += value
        value = -value  # Assume zero-sum game


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # GameState object
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior = 0  # From neural network policy head

    def is_expanded(self):
        return len(self.children) > 0

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
            visit_counts = np.array([child.visit_count if action in root.children else 0
                                     for action in range(NUM_ACTIONS)])
            target_policy = visit_counts / np.sum(visit_counts)

            # Record the state and the policy
            state_tensor = game_state.get_state_as_tensor()
            game_states.append(state_tensor)
            game_policies.append(target_policy)

            # Apply the selected action
            move = np.array([ACTIONS[best_action_idx]])
            game_state.apply_moves(move)

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


# init model and optimiser
board_height, board_width = 11, 11
num_snakes = 1  # Only the agent
model = create_model(board_height, board_width, num_snakes, NUM_ACTIONS)
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# training loop
num_iterations = 10
num_games_per_iteration = 5
num_simulations = 50
batch_size = 32

training_losses = []

for iteration in range(num_iterations):
    print(f"\nIteration {iteration + 1}/{num_iterations}")
    # Self-play to generate training data
    replay_buffer = self_play(model, num_games_per_iteration, num_simulations)

    # Train the model
    num_batches = len(replay_buffer.buffer) // batch_size
    losses = []
    for _ in range(num_batches):
        loss, policy_loss, value_loss = train_model(model, optimizer, replay_buffer, batch_size)
        losses.append(loss)

    avg_loss = np.mean(losses)
    training_losses.append(avg_loss)
    print(f"Average training loss: {avg_loss:.4f}")

plt.plot(training_losses)
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Training Loss over Iterations')
plt.show()
