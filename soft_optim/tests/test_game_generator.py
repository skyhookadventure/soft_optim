import numpy as np
import pytest

from soft_optim.game import TicTacToeBoard


class TestTicTacToeBoardValidStates:
    """Test valid states for TicTacToeBoard"""
    
    mock_empty_board_str: str = \
        """- - -
        - - -
        - - -"""
    
    mock_x_won_board_str: str = \
        """x x x
        o - o
        - o -"""
    
    def test_initializes_empty_board(self):
        board = TicTacToeBoard()
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(
            board.board_state,
            expected)
    
    def test_parses_empty_board(self):
        board = TicTacToeBoard(self.mock_empty_board_str)
        expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(
            board.board_state,
            expected)
    
    def test_parses_x_won_board(self):
        board = TicTacToeBoard(self.mock_x_won_board_str)
        expected = np.array([[1, 1, 1], [2, 0, 2], [0, 2, 0]])
        np.testing.assert_array_equal(
            board.board_state,
            expected)

class TestTicTacToeBoardInvalidStates:
    """Test invalid states for TicTacToeBoard"""
    
    def test_parser_errors_too_many_lines(self):
        with pytest.raises(ValueError, match='Invalid'):
            invalid_str: str = \
                """x x x
                o - o
                - o -
                x x x"""
            TicTacToeBoard(invalid_str)
    
    def test_parser_errors_too_many_columns(self):
        with pytest.raises(ValueError, match='Invalid'):
            invalid_str: str = \
                """x x x x
                o - o
                - o -"""
            TicTacToeBoard(invalid_str)
    
    def test_parser_errors_invalid_row_character(self):
        with pytest.raises(ValueError, match='Invalid'):
            invalid_str: str = \
                """x x y
                o - o
                - o -"""
            TicTacToeBoard(invalid_str)