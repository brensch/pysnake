# game_state.py

import numpy as np
from typing import List, Tuple

class GameState:
    def __init__(self, board_size: Tuple[int, int], snake_bodies: List[np.ndarray], food_positions: List[np.ndarray]):
        self.board_size = board_size
        self.num_snakes = len(snake_bodies)
        self.snake_layers = np.zeros((self.num_snakes, board_size[1], board_size[0]), dtype=int)  # One layer per snake
        self.food_layer = np.zeros(board_size, dtype=int)  # Separate food layer
        self.snake_bodies = snake_bodies  # List of arrays, each array is the body of a snake (including head and tail)
        self.snake_health = np.full(self.num_snakes, 100, dtype=int)  # Health for each snake

        # Initialize snakes
        self.initialize_snakes()

        # Place food
        for food in food_positions:
            self.food_layer[food[1], food[0]] = 1  # Food is marked as 1

    def initialize_snakes(self):
        """ Initialize the layers with snakes represented as countdowns from head to tail. """
        for i, body in enumerate(self.snake_bodies):
            length = len(body)
            for j, segment in enumerate(body):
                self.snake_layers[i, segment[1], segment[0]] = length - j  # Countdown from head (max) to tail (1)

    def apply_moves(self, snake_moves: np.ndarray):
        """ Apply moves to all snakes, update health, and handle collisions. """
        # Decrease health by 1
        self.snake_health -= 1

        # Mark snakes that are alive (health > 0)
        alive_snakes = self.snake_health > 0

        # Decrease all positive values on the snake layers (snake body countdown)
        self.snake_layers[self.snake_layers > 0] -= 1

        new_heads = []
        old_heads = []
        new_bodies = []

        for i in range(self.num_snakes):
            if not alive_snakes[i]:
                new_heads.append(None)
                old_heads.append(None)
                new_bodies.append(self.snake_bodies[i])
                continue

            old_head = self.snake_bodies[i][0]
            move = snake_moves[i]
            new_head = old_head + move

            # Ensure the new head is within the board boundaries
            new_head = np.mod(new_head, self.board_size)

            new_heads.append(new_head)
            old_heads.append(old_head)

            # Update the snake body by removing the last element and adding the new head
            new_body = np.vstack(([new_head], self.snake_bodies[i][:-1]))
            new_bodies.append(new_body)

        # Build body occupancy grid (excluding heads)
        body_occupancy = np.full(self.board_size, -1, dtype=int)
        for i in range(self.num_snakes):
            if not alive_snakes[i]:
                continue
            body = self.snake_bodies[i]
            for segment in body[1:]:
                body_occupancy[segment[1], segment[0]] = i

        # Check for head-to-body collisions
        for i in range(self.num_snakes):
            if not alive_snakes[i]:
                continue
            new_head = new_heads[i]
            if body_occupancy[new_head[1], new_head[0]] != -1:
                # Collision detected
                alive_snakes[i] = False
                print(f"Snake {i} died by colliding into a body.")

        # Check for head-to-head collisions (same position)
        positions = {}
        for i in range(self.num_snakes):
            if not alive_snakes[i]:
                continue
            new_head = tuple(new_heads[i])
            positions.setdefault(new_head, []).append(i)

        for pos, snakes_at_pos in positions.items():
            if len(snakes_at_pos) > 1:
                # Head-to-head collision at pos
                lengths = [len(new_bodies[i]) for i in snakes_at_pos]
                max_length = max(lengths)
                snakes_with_max_length = [snakes_at_pos[j] for j, l in enumerate(lengths) if l == max_length]

                if len(snakes_with_max_length) == len(snakes_at_pos):
                    # All snakes have equal length, all die
                    for i in snakes_at_pos:
                        alive_snakes[i] = False
                        print(f"Snake {i} died in a head-to-head collision at {pos} (equal length).")
                else:
                    # Snakes with shorter length die
                    for j, i in enumerate(snakes_at_pos):
                        if lengths[j] < max_length:
                            alive_snakes[i] = False
                            print(f"Snake {i} died in a head-to-head collision at {pos} (shorter length).")

        # Check for head-on collisions (passing through each other)
        for i in range(self.num_snakes):
            if not alive_snakes[i]:
                continue
            for j in range(i + 1, self.num_snakes):
                if not alive_snakes[j]:
                    continue
                if np.array_equal(new_heads[i], old_heads[j]) and np.array_equal(new_heads[j], old_heads[i]):
                    # Head-on collision detected
                    length_i = len(new_bodies[i])
                    length_j = len(new_bodies[j])

                    if length_i > length_j:
                        alive_snakes[j] = False
                        print(f"Snake {j} died in a head-on collision with snake {i} (snake {i} is longer).")
                    elif length_i < length_j:
                        alive_snakes[i] = False
                        print(f"Snake {i} died in a head-on collision with snake {j} (snake {j} is longer).")
                    else:
                        alive_snakes[i] = False
                        alive_snakes[j] = False
                        print(f"Snake {i} and snake {j} died in a head-on collision (equal length).")

        # Handle food consumption and update health
        for i in range(self.num_snakes):
            if not alive_snakes[i]:
                continue
            new_head = new_heads[i]
            if self.food_layer[new_head[1], new_head[0]] == 1:
                # Snake eats food, grow the body
                tail = new_bodies[i][-1]
                new_bodies[i] = np.vstack((new_bodies[i], [tail]))  # Extend the body
                self.food_layer[new_head[1], new_head[0]] = 0  # Remove the food
                self.snake_health[i] = 100

        # Remove dead snakes
        alive_indices = [i for i in range(self.num_snakes) if alive_snakes[i]]
        self.snake_bodies = [new_bodies[i] for i in alive_indices]
        self.snake_health = self.snake_health[alive_indices]
        self.num_snakes = len(self.snake_bodies)
        self.snake_layers = np.zeros((self.num_snakes, self.board_size[1], self.board_size[0]), dtype=int)

        # Re-initialize snake layers with updated bodies
        self.initialize_snakes()

    def get_state_as_tensor(self) -> np.ndarray:
        """ Returns the game state as a stacked NumPy array suitable for input into TensorFlow. """
        # Stack all snake layers along with the food layer to create a multi-channel tensor
        state_tensor = np.stack([*self.snake_layers, self.food_layer], axis=-1)
        return state_tensor


    def visualize_board_ascii(self):
        """ 
        Visualizes the current board state by combining snake layers and food into a single ASCII grid.
        """
        combined_grid = np.zeros(self.board_size, dtype=str)

        # Fill with empty spaces initially
        combined_grid[:, :] = '.'

        # Combine snakes and food layers
        for i in range(self.num_snakes):
            snake_layer = self.snake_layers[i]
            for y in range(self.board_size[1]):
                for x in range(self.board_size[0]):
                    if snake_layer[y, x] > 0:
                        combined_grid[y, x] = chr(ord('A') + i).lower() if snake_layer[y, x] < len(self.snake_bodies[i]) else chr(ord('A') + i).upper()

        # Place food
        for y in range(self.board_size[1]):
            for x in range(self.board_size[0]):
                if self.food_layer[y, x] == 1:
                    combined_grid[y, x] = 'F'

        # Flip the board on the y-axis and print
        for row in reversed(combined_grid):
            print(' '.join(row))
        print()
