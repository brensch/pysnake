import numpy as np
from typing import List, Tuple
import random

ACTIONS = {
    0: np.array([0, 1]),   # Up
    1: np.array([0, -1]),  # Down
    2: np.array([-1, 0]),  # Left
    3: np.array([1, 0])    # Right
}

class GameState:
    def __init__(self, board_size: Tuple[int, int], snake_bodies: List[np.ndarray], food_positions: List[np.ndarray]):
        self.board_size = board_size
        self.initial_num_snakes = len(snake_bodies)
        self.turn = 0  # Keep track of the turn number

        # Initialize snake bodies
        self.snake_bodies = [body.copy() for body in snake_bodies]

        # Initialize food layer
        self.food_layer = np.zeros(board_size, dtype=int)
        for food in food_positions:
            self.food_layer[food[1], food[0]] = 1  # Place food on the board

        self.snake_health = np.ones(self.initial_num_snakes, dtype=np.int32) * 100
        self.alive_snakes = np.ones(self.initial_num_snakes, dtype=bool)

        self.initialize_snakes()

    def initialize_snakes(self):
        """Initialize snake layers."""
        self.snake_layers = np.zeros((self.initial_num_snakes, self.board_size[1], self.board_size[0]), dtype=np.int32)
        for i, body in enumerate(self.snake_bodies):
            if not self.alive_snakes[i] or len(body) == 0:
                continue
            length = len(body)
            for j, segment in enumerate(body):
                self.snake_layers[i, segment[1], segment[0]] = length - j  # Countdown from head to tail

    def apply_moves(self, snake_moves: np.ndarray):
        """Apply moves using NumPy operations."""
        num_snakes = self.initial_num_snakes

        # Decrease health by 1 for alive snakes
        self.snake_health[self.alive_snakes] -= 1

        # Update alive snakes based on health and clear body if health is 0
        for i in range(num_snakes):
            if self.snake_health[i] <= 0:
                self.alive_snakes[i] = False
                self.snake_bodies[i] = np.array([], dtype=int).reshape(0, 2)  # Clear the body for dead snake
                self.snake_health[i] = 0  # Ensure health is exactly 0

        # Prepare arrays for old heads, new heads, and new bodies
        old_heads = np.full((num_snakes, 2), -1, dtype=int)
        new_heads = np.full((num_snakes, 2), -1, dtype=int)
        new_bodies = [body.copy() for body in self.snake_bodies]

        # Process moves for alive snakes
        for i in range(num_snakes):
            if not self.alive_snakes[i]:
                continue

            # Get the current head and move
            old_head = self.snake_bodies[i][0]
            old_heads[i] = old_head
            move = snake_moves[i]
            new_head = old_head + move

            # Check if the new head is within bounds
            if not (0 <= new_head[0] < self.board_size[0]) or not (0 <= new_head[1] < self.board_size[1]):
                self.alive_snakes[i] = False  # Snake is out of bounds
                new_bodies[i] = np.array([], dtype=int).reshape(0, 2)  # Clear the body for dead snake
                self.snake_health[i] = 0  # Set health to 0
                continue

            new_heads[i] = new_head

            # Update the snake body by moving the head and removing the tail
            new_bodies[i] = np.vstack(([new_head], self.snake_bodies[i][:-1]))

            # Check for self-collision (if the head hits any part of the body)
            if np.any(np.all(new_bodies[i][1:] == new_head, axis=1)):
                self.alive_snakes[i] = False  # Self-collision
                new_bodies[i] = np.array([], dtype=int).reshape(0, 2)  # Clear the body for dead snake
                self.snake_health[i] = 0  # Set health to 0

        # Detect head-to-head collisions first
        alive_indices = np.where(self.alive_snakes)[0]
        dead_snakes_due_to_h2h = set()

        # Build a mapping of head positions to snakes
        head_positions = {}
        for i in alive_indices:
            pos = tuple(new_heads[i])
            head_positions.setdefault(pos, []).append(i)

        # Check for head-to-head collisions at the same position
        for pos, snakes in head_positions.items():
            if len(snakes) > 1:
                # Collision at the same position
                max_length = max(len(new_bodies[i]) for i in snakes)
                longest_snakes = [i for i in snakes if len(new_bodies[i]) == max_length]
                for i in snakes:
                    if i not in longest_snakes:
                        dead_snakes_due_to_h2h.add(i)
                if len(longest_snakes) > 1:
                    # All snakes die if lengths are equal
                    dead_snakes_due_to_h2h.update(longest_snakes)

        # Check for head-on collisions where snakes swap positions
        for idx_i in range(len(alive_indices)):
            i = alive_indices[idx_i]
            if i in dead_snakes_due_to_h2h:
                continue
            for idx_j in range(idx_i + 1, len(alive_indices)):
                j = alive_indices[idx_j]
                if j in dead_snakes_due_to_h2h:
                    continue
                if np.array_equal(new_heads[i], old_heads[j]) and np.array_equal(new_heads[j], old_heads[i]):
                    # Snakes have swapped positions
                    length_i = len(new_bodies[i])
                    length_j = len(new_bodies[j])
                    if length_i > length_j:
                        dead_snakes_due_to_h2h.add(j)
                    elif length_j > length_i:
                        dead_snakes_due_to_h2h.add(i)
                    else:
                        dead_snakes_due_to_h2h.add(i)
                        dead_snakes_due_to_h2h.add(j)

        # Update snakes that died due to head-to-head collisions
        for i in dead_snakes_due_to_h2h:
            self.alive_snakes[i] = False
            new_bodies[i] = np.array([], dtype=int).reshape(0, 2)  # Clear the body
            self.snake_health[i] = 0  # Set health to 0

        # Now, rebuild the list of alive snakes after head-to-head collisions
        alive_indices = np.where(self.alive_snakes)[0]

        # Build body occupancy grid excluding heads of alive snakes
        # Exclude bodies of snakes that died in head-to-head collisions
        body_occupancy = np.full(self.board_size, -1, dtype=int)
        for i in alive_indices:
            body = new_bodies[i][1:]  # Exclude the new head
            if len(body) > 0:
                body_occupancy[body[:, 1], body[:, 0]] = i

        # Detect head-to-body collisions
        for i in alive_indices.copy():
            head = new_heads[i]
            occupant = body_occupancy[head[1], head[0]]
            if occupant != -1:
                # Collision detected
                self.alive_snakes[i] = False
                new_bodies[i] = np.array([], dtype=int).reshape(0, 2)  # Clear the body
                self.snake_health[i] = 0  # Set health to 0

        # Handle food consumption and update health
        for i in range(num_snakes):
            if not self.alive_snakes[i]:
                continue
            head = new_heads[i]
            if self.food_layer[head[1], head[0]] == 1:
                # Snake eats food, grow the body
                tail = new_bodies[i][-1]
                new_bodies[i] = np.vstack((new_bodies[i], [tail]))  # Extend the body
                self.food_layer[head[1], head[0]] = 0  # Remove the food
                self.snake_health[i] = 100  # Reset health to 100

        # After handling snake movements and collisions, add food with a random chance
        food_add_probability = 0.1

        if random.random() < food_add_probability:
            # Find unoccupied positions
            occupied_positions = set()
            for snake in self.snake_bodies:
                for segment in snake:
                    occupied_positions.add((segment[0], segment[1]))
            for food in self.food_positions:
                occupied_positions.add((food[0], food[1]))

            # Generate a list of all positions on the board
            all_positions = set((x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1]))

            # Determine unoccupied positions
            unoccupied_positions = list(all_positions - occupied_positions)

            if unoccupied_positions:
                # Choose a random unoccupied position to place food
                new_food_position = random.choice(unoccupied_positions)
                self.food_positions.append(np.array(new_food_position))

        # Update snake bodies
        self.snake_bodies = new_bodies

        # Re-initialize snake layers with updated bodies and health
        self.initialize_snakes()

        self.turn += 1  # Increment the turn counter

    def is_terminal(self) -> bool:
        """Check if the game is over."""
        return not any(self.alive_snakes) or sum(self.alive_snakes) == 1

    def get_safe_actions(self, snake_index: int) -> List[int]:
        """Get a list of safe actions for the given snake, only checking for out-of-bounds moves."""
        safe_actions = []
        if not self.alive_snakes[snake_index]:
            return safe_actions

        head = self.snake_bodies[snake_index][0]
        for action_idx, move in ACTIONS.items():
            new_head = head + move
            # Check bounds only
            if 0 <= new_head[0] < self.board_size[0] and 0 <= new_head[1] < self.board_size[1]:
                safe_actions.append(action_idx)
        return safe_actions

    def get_valid_actions(self, snake_index: int) -> List[int]:
        """Get a list of valid actions (including potentially unsafe ones)."""
        if not self.alive_snakes[snake_index]:
            return []
        return list(ACTIONS.keys())

    def get_state_as_tensor(self) -> np.ndarray:
        """Returns the game state as a stacked NumPy array suitable for input into TensorFlow."""
        layers = []
        for i in range(self.initial_num_snakes):
            layers.append(self.snake_layers[i])  # Snake body layer
            # Normalize health to [0,1] and create a health layer
            health_layer = np.full(self.board_size, self.snake_health[i] / 100.0)
            health_layer[self.snake_layers[i] == 0] = 0  # Zero out health where there's no snake
            layers.append(health_layer)
        layers.append(self.food_layer)  # Food layer
        state_tensor = np.stack(layers, axis=-1).astype(np.float32)
        return state_tensor

    def visualize_board_ascii(self) -> None:
        """Visualizes the current board state by combining snake layers and food into a single ASCII grid."""
        combined_grid = np.full(self.board_size, '.', dtype=str)

        for i in range(self.initial_num_snakes):
            if not self.alive_snakes[i]:
                continue
            snake_layer = self.snake_layers[i]
            for y in range(self.board_size[1]):
                for x in range(self.board_size[0]):
                    if snake_layer[y, x] > 0:
                        combined_grid[y, x] = chr(ord('A') + i).lower() if snake_layer[y, x] < len(self.snake_bodies[i]) else chr(ord('A') + i).upper()

        for y in range(self.board_size[1]):
            for x in range(self.board_size[0]):
                if self.food_layer[y, x] == 1:
                    combined_grid[y, x] = 'F'

        for row in reversed(combined_grid):
            print(' '.join(row))

    @classmethod
    def from_state_tensor(cls, state_tensor: np.ndarray, num_snakes: int) -> 'GameState':
        board_size = (state_tensor.shape[1], state_tensor.shape[0])

        # Extract the snake body and health layers from the state tensor
        snake_bodies = []
        for i in range(num_snakes):
            snake_body_layer = state_tensor[..., i * 2]
            snake_body = np.argwhere(snake_body_layer > 0)
            snake_bodies.append(snake_body)

        food_positions = np.argwhere(state_tensor[..., -1] == 1)

        # Initialize GameState with the extracted information
        return cls(board_size, snake_bodies, food_positions)
