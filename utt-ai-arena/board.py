from enum import IntEnum
from typing import Callable, Union


class Piece(IntEnum):
    EMPTY = 0
    O = -1
    X = 1


class BoardState(IntEnum):
    NOT_FINISHED = 1
    DRAW = 1
    O_WON = 2
    X_WON = 3


class Board:
    """Class to represent any board, inner boards or the main board (composed of 9 inner boards)."""

    def __init__(
        self, piece_factory: Callable[[], Union[Piece, "Board"]] = lambda: Piece.EMPTY
    ) -> None:
        """
        Initializes a board with a 3x3 grid of pieces or inner boards.

        Args:
            piece_factory (Callable[[], Union[Piece, "Board"]]): A factory function that
                generates a new piece or inner board. Defaults to a function that returns Piece.EMPTY.
        """

        # Create a 3x3 grid with independent values generated from the factory
        self.board = [[piece_factory() for _ in range(3)] for _ in range(3)]
        self.board_state = BoardState.NOT_FINISHED

    def __getitem__(self, idx: int) -> Union[Piece, "Board"]:
        return self.board[idx]

    def __setitem__(self, idx: int, value) -> None:
        self.board[idx] = value

    @property
    def value(self):
        """(⚆ᗝ⚆)
        Super cool function that lets the main board treat inner boards like regular pieces.

        If this board isn't finished yet, it returns Piece.EMPTY — just like an empty cell.
        This makes it easier to compute the main board's state with the same function as inner boards without special cases.

        Returns:
            Piece: The current state of this board, interpreted as a Piece.
        """
        match self.board_state:
            case BoardState.X_WON:
                return Piece.X
            case BoardState.O_WON:
                return Piece.O
            case _:
                return Piece.EMPTY

    def place_piece(self, l: int, c: int, p: Piece) -> bool:
        """
        Places a piece on the board at the specified location.

        Args:
            l (int): The row index (0-2).
            c (int): The column index (0-2).
            p (Piece): The piece to place (X or O).

        Returns:
            bool: True if the piece was placed successfully, False if the move is invalid.
        """

        # Valid move
        if (
            self.board[l][c] == Piece.EMPTY
            and self.board_state == BoardState.NOT_FINISHED
        ):
            self.board[l][c] = p
            self.board_state = self.get_game_state()
            return True
        return False

    def get_game_state(self) -> BoardState:
        """
        Checks the current state of the board to determine if the game is finished, and if so,
        whether X or O has won, or if it's a draw.

        Returns:
            BoardState: The current state of the board.
        """

        x_won = 3
        o_won = -3
        tot_sum = 0  # if it reaches 9 board is full

        for i in range(3):
            line_sum = sum(self.board[i][j].value for j in range(3))
            col_sum = sum(self.board[j][i].value for j in range(3))
            tot_sum += abs(line_sum)

            if line_sum == x_won or col_sum == x_won:
                return BoardState.X_WON
            elif line_sum == o_won or col_sum == o_won:
                return BoardState.O_WON

        diag1 = self.board[0][0].value + self.board[1][1].value + self.board[2][2].value
        diag2 = self.board[2][0].value + self.board[1][1].value + self.board[0][2].value

        if diag1 == x_won or diag2 == x_won:
            return BoardState.X_WON
        elif diag1 == o_won or diag2 == o_won:
            return BoardState.O_WON
        elif tot_sum == 9:  # all cells filled
            return BoardState.DRAW

        return BoardState.NOT_FINISHED


def get_board() -> Board:
    """Returns a main board composed of 9 fresh inner boards."""
    return Board(piece_factory=lambda: Board())
