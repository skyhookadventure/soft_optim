"""Tic Tac Toe Game implementation"""
import re
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np


class Player(Enum):
    """Tic Tac Toe Player"""

    X = "x"
    O = "o"


class Square(Player):
    """Tic Tac Toe Square"""

    # Note this extends player, so it already has X and O (and thus just needs
    # the third option of BLANK).
    X = "x"
    O = "o"
    EMPTY = "-"


class Board:
    """Tic Tac Toe Board"""

    contains_illegal_move: bool = False
    """Flag that the board contains an illegal move"""

    number_columns: int = 3
    """Number of columns"""

    number_rows: int = 3
    """Number of rows"""

    number_joined_to_win: int = 3
    """Number in a row/column/diagonal to win"""

    @property
    def board_squares(self) -> List[List[Square]]:
        """Board squares

        Nested list of squares, in the format [rows x columns].

        >>> Board().board_squares
        [[Square.BLANK, Square.BLANK, Square.BLANK],
        [Square.BLANK, Square.BLANK, Square.BLANK],
        [Square.BLANK, Square.BLANK, Square.BLANK]]
        """
        return self._board_squares

    @board_squares.setter
    def board_squares(self, value: List[List[Square]]) -> None:
        """Board squares setter

        Checks that the size of the board is correct when setting all squares at
        once.

        Params:
            value: Nested list in the format rows x columns. For example, the
            middle square is `self.board[1][1]`.
        """
        # Check number of rows
        assert len(value) == self.number_rows, \
            ValueError("Invalid number of rows")

        # Check number of columns
        for row in value:
            assert len(row) == self.number_columns, \
                ValueError("Invalid number of columns")

        # Set the board
        self._board_squares = value

        # Check the number of moves makes sense (note this must be done after
        # the board is set)
        x_moves = self.number_squares_played(Player.X)
        o_moves = self.number_squares_played(Player.O)
        assert x_moves == o_moves or x_moves == o_moves + 1, \
            ValueError("Invalid number of moves")

        # TODO: Check there isn't more than one winner

    def __init__(
        self,
        board_string: Optional[str] = None,
        allow_illegal_moves: Optional[bool] = False
    ):
        """Initialise the board

        Args:
            board_string: Board string to parse (of the format "- x o" on three
            lines).
        """
        # Settings (must be applied first)
        self.allow_illegal_moves = allow_illegal_moves

        # Parse the board string if provided
        if board_string:
            self.board_squares = self._parse_board_string(board_string)

        # Default to an empty board
        else:
            self.board_squares = \
                [[Square.EMPTY] * self.number_columns] * self.number_rows

    @property
    def valid_moves(self) -> List[Tuple[int, int]]:
        """Valid Moves

        Returns:
            List of empty squares

        >>> Board().valid_moves
        [[0, 0], [0, 1], [0, 2],
        [1, 0], [1, 1], [1, 2],
        [2, 0], [2, 1], [2, 2]]
        """
        empty_squares: List[Tuple[int, int]] = []

        for row_idx, row in enumerate(self.board_squares):
            for col_idx, square in enumerate(row):
                if square == Square.EMPTY:
                    empty_squares.append((row_idx, col_idx))

        return empty_squares

    def play_move(self, row: int, col: int, player: Player):
        """Play a move

        Args:
            row: Row
            col: Column
            player: Player
        """
        # Check the square exists
        assert self.board_squares[row], \
            ValueError("Invalid move - row doesn't exist")
        assert self.board_squares[row][col], \
            ValueError("Invalid move - column doesn't exist")

        # Check for illegal moves, if they're not allowed
        if not self.allow_illegal_moves:
            # Check the square is empty
            assert self.board_squares[row][col] == Square.EMPTY, \
                ValueError("Invalid move - square is not empty")

            # Check it is the player's turn
            assert self.current_turn_player == player, \
                ValueError("Invalid move - not this player's turn")

        # TODO: Check there isn't a winner already

        # Modify the board
        self._board_squares[row][col] = player

    def get_col(self, col_idx: int) -> List[Square]:
        column: List[Square] = []

        for row_idx in len(self.board_squares):
            square = self.board_squares[row_idx][col_idx]
            column.append(square)

        return column

    @property
    def winner(self) -> Optional[Player]:
        # TODO: Allow for winning streak to be less than the full length

        # Check rows
        for row in self.board_squares:
            if len(set(row)) == 1 and row[0] != Square.EMPTY:
                return row[0]

        # Check columns
        for col_idx in self.board_squares:
            col = self.get_col(col_idx)
            if len(set(col)) == 1 and col[0] != Square.EMPTY:
                return col[0]

        # Check diagonals TODO

        # Otherwise return None
        return None

    def __str__(self) -> str:
        """String representation of the board"""
        view: str = ""

        for row in self.board_squares:
            for square in row:
                square_view: str = str(square) + " "
                view += square_view
            view += "\n"

        return view

    def number_squares_played(self, player: Player) -> int:
        """Number of squares played by a specific player

        Args:
            player: Player

        Returns:
            int: Number of squares the player has played
        """
        count: int = 0

        for row in self.board_squares:
            for square in row:
                if square == player:
                    count += 1

        return count

    @property
    def current_turn_player(self) -> Optional[Player]:
        """The player whose turn it is currently"""
        # Count the number of squares played
        x_squares = self.number_squares_played(Player.X)
        o_squares = self.number_squares_played(Player.O)

        # Allow for a full board
        total_squares = self.number_rows * self.number_columns
        if x_squares + o_squares == total_squares:
            return None

        # Otherwise look at who has played most squares
        if o_squares < x_squares:
            return Player.O
        else:
            return Player.X

    # def validate_state(self):
    #     """ Check that the board state is valid, including that the number of
    #     x's and o's is correct.
    #     """
    #     blank_count, x_count, o_count = self.get_counts()
    #     assert blank_count + x_count + o_count == 9
    #     assert (x_count == o_count) or (x_count == o_count + 1)

    def _parse_board_string(self, board_string: str):
        # Trim proceeding and trailing spaces & new lines
        trimmed: str = board_string.strip()

        # Split the string into lines
        lines: List[str] = trimmed.split("\n")

        # Initialise the board squares
        board_squares: List[List[Square]] = []

        for line in lines:
            row: List[Square] = []

            trimmed_line: str = line.strip()

            for char in trimmed_line.split(" "):
                # Check the character is one of the Square ENUM values
                assert Square(char), ValueError(
                    "Invalid character in board string")

                row.append(Square(char))

            board_squares.append(row)

        # Check the number of squares

        return board_squares

        # create a dict that does the opposite of self.map
        rev_map = {v: k for k, v in self.map.items()}

        # iterate over it and convert to state
        for i, line in enumerate(lines):
            l = line.strip(" ").split(" ")
            for j, char in enumerate(l):
                self.board_squares[i, j] = rev_map[char]


class TicTacToeGame:
    """ Class to represent a game of Tic Tac Toe at multiple points in time.
    """

    def __init__(self,
                 init_board: str = None,
                 game_string: str = None,
                 check_valid_move: bool = True,
                 check_valid_state: bool = True):

        self.board = Board(init_board)
        self.history: List[Board] = [self.board]

        # Choose which validity checks to perform
        #  Check that the board state has only 'x', 'o', and '-' characters
        self.check_valid_string = True

        #  Check that on state change, only one blank piece has changed
        self.check_valid_move = check_valid_move

        # Check that the board state is valid, such that #x == #o or #x == #o + 1
        self.check_valid_state = check_valid_state

    def reset(self):
        self.board = Board()
        self.history = [self.board]
        return self

    def validate_move(self, old_state: Board, new_state: Board):
        if self.check_valid_state:
            assert old_state.validate_state()
            assert new_state.validate_state()

        if self.check_valid_move:
            # Check that the new state has one more piece than the old state
            assert np.sum(new_state.board_squares !=
                          old_state.board_squares) == 1
            assert np.sum(new_state.board_squares ==
                          old_state.board_squares) == 8

            old_blank_count, _old_x_count, _old_o_count = old_state.get_counts()
            new_blank_count, _new_x_count, _new_o_count = new_state.get_counts()
            assert new_blank_count == old_blank_count - 1

    def add_state(self, board_string: str) -> Tuple[int, bool]:
        """ Add a new state to the game, check for validity.

        Returns:
            outcome (int): The outcome of the game, 0 if the game is not over
            valid (bool): Whether the state was valid
        """

        # Perform validity checks
        try:
            # Load the new state and save it to the history
            self.board = Board(board_string)
            self.history.append(self.board)

            if self.check_valid_state:
                assert self.board.validate_state()

            if self.check_valid_move and len(self.history) > 1:
                self.validate_move(self.history[-2], self.board)

        except ValueError as _err:
            return 0, False

        except AssertionError as _err:
            return 0, False

        except AssertionError as e:
            print(e)
            return 0, False

        #  If valid, perform win checks
        outcome = self.board.check_win()
        return outcome, True

    def validate_game_string(self, game_string: str) -> Tuple[int, bool]:
        self.board = Board()
        self.history = [self.board]

        # split game string into board states
        board_strings = game_string.split("\n\n")[1:]

        final_outcome = 0

        for board_string in board_strings:
            outcome, valid = self.add_state(board_string)

            # If the game is won, return score based on winner
            if outcome and not final_outcome:
                final_outcome = outcome

            if not valid:
                return 0, False

        return final_outcome, True

    def evaluate_game_string(self,
                             game_string: str,
                             ) -> int:
        outcome, valid = self.validate_game_string(game_string)

        # If the game is not valid, return -1
        if not valid:
            return 0.0

        # If the game is won, return score based on winner
        if outcome == self.board.x:
            return 1.0
        if outcome == self.board.o:
            return 0.0

        return 0.0


def generate_random_game():
    b = Board()
    game_state_history = [str(b)]
    for t in range(9):
        # Get the player
        # Make a random valid move
        valid_moves = b.get_valid_moves()
        move = np.random.choice(len(valid_moves))
        b.play_move(*valid_moves[move])
        game_state_history.append(str(b))

    return "Let's play Tic Tac Toe:\n" + "\n".join(game_state_history) + "<|endoftext|>"


def generate_dataset(number_games: int) -> List[str]:
    """Generate a list of games

    Args:
        number_games (int): Number of games

    Returns:
        List: List of games (strings with a full game)
    """
    return [generate_random_game() for _ in range(number_games)]


if __name__ == "__main__":
    # TEST
    # Generate a game
    game = generate_random_game()
    print(game)

    # Evaluate the game
    print(evaluate_game_string(game))
