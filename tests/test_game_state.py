import unittest
import numpy as np
from game_state import GameState

class TestGameState(unittest.TestCase):

    def assertGameStateEqual(self, actual: GameState, expected: GameState):
        """
        Asserts that the actual GameState matches the expected GameState.
        Provides detailed feedback including visualized boards if there is a failure.
        """


        def print_difference(array_name, actual_array, expected_array):
            print(f"\n--- Difference in {array_name} ---")
            differences = np.argwhere(actual_array != expected_array)
            
            # Create a copy of both arrays for displaying with differences highlighted
            actual_display = actual_array.astype(str)
            expected_display = expected_array.astype(str)
            
            if differences.size > 0:
                for diff in differences:
                    # Mark the mismatched elements with parentheses
                    actual_display[tuple(diff)] = f"({actual_array[tuple(diff)]})"
                    expected_display[tuple(diff)] = f"({expected_array[tuple(diff)]})"

                # Print both arrays in full with highlighted differences
                print(f"\nExpected {array_name}:")
                print(expected_display)
                
                print(f"\nActual {array_name}:")
                print(actual_display)
            else:
                print(f"No differences found in {array_name}.")

        print("\n--- Expected Board ---")
        expected.visualize_board_ascii()
        print("\n--- Actual Board ---")
        actual.visualize_board_ascii()

        try:
            # Check snake layers (one layer per snake)
            if not np.array_equal(actual.snake_layers, expected.snake_layers):
                print_difference('snake_layers', actual.snake_layers, expected.snake_layers)
            np.testing.assert_array_equal(actual.snake_layers, expected.snake_layers)
        except AssertionError as e:
            raise AssertionError("Mismatch in snake_layers") from e

        try:
            # Check snake bodies
            for i, (actual_body, expected_body) in enumerate(zip(actual.snake_bodies, expected.snake_bodies)):
                if not np.array_equal(actual_body, expected_body):
                    print_difference(f'snake_body[{i}]', actual_body, expected_body)
                np.testing.assert_array_equal(actual_body, expected_body)
        except AssertionError as e:
            raise AssertionError(f"Mismatch in snake_body[{i}]") from e

        try:
            # Check food layer
            if not np.array_equal(actual.food_layer, expected.food_layer):
                print_difference('food_layer', actual.food_layer, expected.food_layer)
            np.testing.assert_array_equal(actual.food_layer, expected.food_layer)
        except AssertionError as e:
            raise AssertionError("Mismatch in food_layer") from e

        try:
            # Check snake health
            if not np.array_equal(actual.snake_health, expected.snake_health):
                print_difference('snake_health', actual.snake_health, expected.snake_health)
            np.testing.assert_array_equal(actual.snake_health, expected.snake_health)
        except AssertionError as e:
            raise AssertionError("Mismatch in snake_health") from e

    def run_game_test(self, initial_state, moves, expected_state, test_name):
        """
        Runs a test where it compares the actual state after applying moves with the expected state.
        """
        with self.subTest(test_name):
            print(f"\n--- Test: {test_name} ---")

            # Initialize game state using snake bodies
            game_state = GameState(
                initial_state['board_size'],
                initial_state['snake_bodies'],
                initial_state['food_positions']
            )
            game_state.snake_health = initial_state.get('snake_health', np.full(len(initial_state['snake_bodies']), 100))
            print("\n--- Before Board ---")
            game_state.visualize_board_ascii()
            # Apply the moves
            game_state.apply_moves(moves)

            # Create the expected game state for comparison
            expected_game_state = GameState(
                expected_state['board_size'],
                expected_state['snake_bodies'],
                expected_state['food_positions']
            )
            expected_game_state.snake_health = expected_state.get('snake_health', np.full(len(expected_state['snake_bodies']), 100))

            # Compare the actual and expected game states
            self.assertGameStateEqual(game_state, expected_game_state)

    def test_snake_moves(self):
        """
        Runs multiple test cases with different starting conditions, moves, and expected resulting boards.
        """

        # Define test cases: each test contains an initial state, moves, expected state, and test name
        test_cases = [
            {
                'name': 'Simple move without collision or eating',
                'initial_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [
                        np.array([[2, 2], [1, 2], [0, 2]]),  # Snake 1 body
                        np.array([[4, 4], [3, 4], [2, 4]])   # Snake 2 body
                    ],
                    'food_positions': np.array([[0, 0]]),
                    'snake_health': np.array([100, 100])
                },
                'moves': np.array([[1, 0], [0, -1]]),  # Snake 1 moves right, Snake 2 moves down
                'expected_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [
                        np.array([[3, 2], [2, 2], [1, 2]]),  # Snake 1 body after the move
                        np.array([[4, 3], [4, 4], [3, 4]])   # Snake 2 body after the move
                    ],
                    'food_positions': np.array([[0, 0]]),
                    'snake_health': np.array([99, 99])
                }
            },
            {
                'name': 'Collision with food',
                'initial_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [
                        np.array([[2, 2], [1, 2], [0, 2]])
                    ],
                    'food_positions': np.array([[3, 2]]),
                    'snake_health': np.array([100])
                },
                'moves': np.array([[1, 0]]),  # Snake moves right to eat food
                'expected_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [
                        np.array([[3, 2], [2, 2], [1, 2], [1, 2]])  # Snake has grown
                    ],
                    'food_positions': np.array([]),
                    'snake_health': np.array([100])  # Health restored to 100
                }
            },
            {
                'name': 'Head-to-body collision',
                'initial_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [
                        np.array([[2, 2], [2, 3], [2, 4]]),  # Snake 1
                        np.array([[3, 2], [3, 1], [3, 0]])   # Snake 2
                    ],
                    'food_positions': np.array([]),
                    'snake_health': np.array([100, 100])
                },
                'moves': np.array([[0, -1], [-1, 0]]),  # Snake 1 moves up into Snake 2's body
                'expected_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [
                        np.array([[2, 1], [2, 2], [2, 3]]),  # Snake 1
                        np.array([[2, 2], [3, 2], [3, 1]])   # Snake 2
                    ],
                    'food_positions': np.array([]),
                    'snake_health': np.array([99, 99])
                }
            },
            {
                'name': 'Head-to-head collision, equal length',
                'initial_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [
                        np.array([[2, 2], [2, 3]]),  # Snake 1
                        np.array([[2, 4], [2, 3]])   # Snake 2
                    ],
                    'food_positions': np.array([]),
                    'snake_health': np.array([100, 100])
                },
                'moves': np.array([[0, 1], [0, -1]]),  # Both snakes move into [2, 3]
                'expected_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [],
                    'food_positions': np.array([]),
                    'snake_health': np.array([])
                }
            },
            {
                'name': 'Head-on collision, unequal lengths',
                'initial_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [
                        np.array([[2, 2], [2, 3], [2, 4]]),  # Snake 1 (length 3)
                        np.array([[2, 1], [2, 0]])           # Snake 2 (length 2)
                    ],
                    'food_positions': np.array([]),
                    'snake_health': np.array([100, 100])
                },
                'moves': np.array([[0, -1], [0, 1]]),  # Snakes move into each other's old head positions
                'expected_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [
                        np.array([[2, 1], [2, 2], [2, 3]])  # Snake 1 survives
                    ],
                    'food_positions': np.array([]),
                    'snake_health': np.array([99])
                }
            },
            {
                'name': 'Snake dies due to health depletion',
                'initial_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [
                        np.array([[2, 2], [2, 3], [2, 4]])  # Snake 1
                    ],
                    'food_positions': np.array([]),
                    'snake_health': np.array([1])
                },
                'moves': np.array([[0, 1]]),  # Snake moves, health should drop to 0
                'expected_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [],
                    'food_positions': np.array([]),
                    'snake_health': np.array([])
                }
            },
            {
                'name': 'Snakes passing through each other (no collision)',
                'initial_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [
                        np.array([[1, 2], [0, 2]]),  # Snake 1
                        np.array([[2, 2], [3, 2]])   # Snake 2
                    ],
                    'food_positions': np.array([]),
                    'snake_health': np.array([100, 100])
                },
                'moves': np.array([[1, 0], [-1, 0]]),  # Snakes move into each other's bodies (but not heads)
                'expected_state': {
                    'board_size': (5, 5),
                    'snake_bodies': [],  # Both snakes die due to head-on collision
                    'food_positions': np.array([]),
                    'snake_health': np.array([])
                }
            },
            {
                'name': 'Two snakes collide on the same food (same length, both die)',
                'initial_state': {
                    'board_size': (7, 7),
                    'snake_bodies': [
                        np.array([[2, 2], [2, 1], [2, 0]]),  # Snake 1 (length 3)
                        np.array([[2,4],[2,5],[2,6],])   # Snake 2 (length 3)
                    ],
                    'food_positions': np.array([[2, 3]]),   # Food placed at [2, 1]
                    'snake_health': np.array([100, 100])
                },
                'moves': np.array([[0, 1], [0, -1]]),  # Both snakes move towards [2, 1] where the food is located
                'expected_state': {
                    'board_size': (7, 7),
                    'snake_bodies': [],  # Both snakes die, so no snakes are left
                    'food_positions': np.array([[2,3]]),  # Food is consumed by the dead snakes
                    'snake_health': np.array([])  # Both snakes are dead
                }
            }
        ]

        # Run each test case
        for i, test_case in enumerate(test_cases):
            self.run_game_test(
                test_case['initial_state'], 
                test_case['moves'], 
                test_case['expected_state'], 
                test_case['name']
            )


# Run the test
if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
