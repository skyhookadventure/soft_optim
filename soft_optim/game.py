"""Tic Tac Toe Game implementation"""
import re
from enum import IntEnum
from typing import List

import numpy as np


class TicTacToeBoard:
    """Tic Tac Toe Board
    
    Contains the board state at a single point in time, i.e. 9 squares with 3
    possible values [-,x,o].
    """
    
    board_state: np.ndarray
    """Board state as a numpy array of shape (3,3)
    
    Note we use integers to represent the state of each square, rather than
    strings, which are defined below."""
    
    blank: int = 0
    """Integer representation of blank on the board (numpy array)"""
    
    x: int = 1
    """Integer representation of x on the board (numpy array)"""
    
    o: int = 2
    """Integer representation of o on the board (numpy array)"""
    
    def __init__(self, string=None):
        # Initialise an empty board
        self.board_state = np.full((3,3), self.blank, int)
        
        # Setup the mapping of strings to square values
        self.map = {self.x:'x', self.o:'o', self.blank: '-'}

        # Parse a string representation of the board state, if given
        if string is not None:
            self.parse_str(string)

    def get_valid_moves(self):
        ''' return a list of valid (i,j,player) moves '''
        # work out whose turn it is
        num_x = np.sum(self.board_state == self.x)
        num_o = np.sum(self.board_state == self.o)
        if num_x == num_o:
            turn = self.x
        elif num_x == num_o + 1:
            turn = self.o
        else:
            print("Invalid board state")

        # make list
        l = []
        for i in range(3):
            for j in range(3):
                if self.board_state[i,j] == self.blank:
                    l.append((i,j,turn))
        return l


    def make_move(self, i, j, player):
        # check if legal
        if i >= 3 or i < 0 or j >= 3 or j < 0:
            print("Index out of bounds")
        elif self.board_state[i,j] != self.blank:
            print("Not a blank square")
        elif player != self.x and player != self.o:
            print("Invalid player")

        # modify board
        self.board_state[i,j] = player


    def check_win(self):
        for player in [self.x, self.o]:
            won = False
            # check columns
            if np.any(np.all(self.board_state == player, axis=0)):
                won = True
            # check rows
            if np.any(np.all(self.board_state == player, axis=1)):
                won = True

            # check diagonals
            elif np.all(np.diag(self.board_state) == player) \
                    or np.all(np.diag(np.fliplr(self.board_state))== player):
                won = True

            if won:
                return player

        return False

    def __str__(self):
        b = self.board_state
        out = ''
        # convert state to string
        for i in range(3):
            for j in range(3):
                out += f" {self.map[b[i,j]]}"
            out += "\n"
        return out

    def validate_str(self, string):
        lines = string.strip("\n").split("\n")

        # Check there are 3 lines
        if len(lines) != 3:
            raise ValueError("Invalid game string - incorrect number of rows")

        # iterate over it and convert to state
        for line in lines:
            if re.match(r"^[ ]*[-xo][ ][-xo][ ][-xo][ ]*$", line) is None:
                raise ValueError("Invalid game string - invalid row format")

        return True

    def parse_str(self, string: str):
        # Ensure the state string is of the correct format
        self.validate_str(string)

        # Split the string into lines
        lines = string.strip("\n").split("\n")

        # create a dict that does the opposite of self.map
        rev_map = {v:k for k,v in self.map.items()}

        # iterate over it and convert to state
        for i, line in enumerate(lines):
            l = line.strip(" ").split(" ")
            for j, char in enumerate(l):
                self.board_state[i,j] = rev_map[char]


def evaluate_game_string(game_string) -> int:
    # split game string into board states
    game_states = game_string.split("\n\n")[1:]
    for state in game_states:
        b = TicTacToeBoard(state)
        outcome = b.check_win()
        if outcome == b.x:
            return 1
        if outcome == b.o:
            return -1
    return 0


def generate_random_game():
    b = TicTacToeBoard()
    game_state_history = [ str(b) ]
    for t in range(9):
        valid_moves = b.get_valid_moves()
        move = np.random.choice(len(valid_moves))
        b.make_move(*valid_moves[move])
        game_state_history.append( str(b) )

    return "Let's play Tic Tac Toe:\n" + "\n".join(game_state_history) + "<|endoftext|>"


def generate_dataset(number_games: int) -> List[str]:
    """Generate a list of games

    Args:
        number_games (int): Number of games

    Returns:
        List: List of games (strings with a full game)
    """
    return [ generate_random_game() for _ in range(number_games) ]


if __name__ == "__main__":
    # TEST
    # Generate a game
    game = generate_random_game()
    print(game)

    # Evaluate the game
    print(evaluate_game_string(game))
