from __future__ import annotations
from enum import IntEnum
from typing import Callable, Union, Any
from dataclasses import dataclass
from typing import Tuple


class Piece(IntEnum):
    EMPTY = 0
    O = -1
    X = 1


@dataclass
class Move:
    piece: Piece
    outer: Tuple[int, int]  # outer board position select
    inner: Tuple[int, int]  # inner board position select


class BoardState(IntEnum):
    NOT_FINISHED = 0
    DRAW = 1
    O_WON = 2
    X_WON = 3


class Board:
    """Class to represent any board, inner boards or the main board (composed of 9 inner boards)."""

    def __init__(
        self, piece_factory: Callable[[], Union[Piece, "Board"]] = lambda: Piece.EMPTY
    ) -> None:
        """Initializes a board with a 3x3 grid of pieces or inner boards."""

        # Create a 3x3 grid with independent values generated from the factory
        self.board = [[piece_factory() for _ in range(3)] for _ in range(3)]
        self.board_state = BoardState.NOT_FINISHED

    def __getitem__(self, idx: int) -> Any:
        return self.board[idx]

    def __setitem__(self, idx: int, value) -> None:
        self.board[idx] = value

    @property
    def value(self):
        """(⚆ᗝ⚆)
        Super cool function that lets the main board treat inner boards like regular pieces.

        If this board isn't finished yet, it returns Piece.EMPTY — just like an empty cell.
        This makes it easier to compute the main board's state with the same function as inner boards without special cases.
        """
        match self.board_state:
            case BoardState.X_WON:
                return Piece.X
            case BoardState.O_WON:
                return Piece.O
            case _:
                return Piece.EMPTY

    def place_piece(self, l: int, c: int, p: Piece) -> bool:
        """Places a piece on the board at the specified location."""

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
        x_won = 3
        o_won = -3

        for i in range(3):
            line_sum = sum(self.board[i][j].value for j in range(3))
            col_sum = sum(self.board[j][i].value for j in range(3))
            if line_sum == x_won or col_sum == x_won:
                return BoardState.X_WON
            if line_sum == o_won or col_sum == o_won:
                return BoardState.O_WON

        # diagonals
        diag1 = self.board[0][0].value + self.board[1][1].value + self.board[2][2].value
        diag2 = self.board[2][0].value + self.board[1][1].value + self.board[0][2].value
        if diag1 == x_won or diag2 == x_won:
            return BoardState.X_WON
        if diag1 == o_won or diag2 == o_won:
            return BoardState.O_WON

        # empty?
        def is_empty(cell) -> bool:
            if isinstance(cell, Piece):
                return cell == Piece.EMPTY
            return cell.board_state == BoardState.NOT_FINISHED

        any_empty = any(is_empty(self.board[r][c]) for r in range(3) for c in range(3))
        return BoardState.NOT_FINISHED if any_empty else BoardState.DRAW

    def clone(self) -> Board:
        """Returns a deep copy of this board, including its inner boards or pieces."""

        new_board = Board(piece_factory=lambda: Piece.EMPTY)
        new_board.board_state = self.board_state

        for i in range(3):
            for j in range(3):
                cell = self.board[i][j]
                if isinstance(cell, Board):
                    new_board.board[i][j] = cell.clone()
                else:
                    new_board.board[i][j] = cell

        return new_board


def get_board() -> Board:
    """Returns a main board composed of 9 inner boards."""
    return Board(piece_factory=lambda: Board())


def legal_moves(board, piece, restriction) -> list[Move]:
    moves = []
    if restriction is None:
        outers = [
            (R, C)
            for R in range(3)
            for C in range(3)
            if board[R][C].get_game_state() == BoardState.NOT_FINISHED
        ]
    else:
        outers = (
            [restriction]
            if board[restriction[0]][restriction[1]].get_game_state()
            == BoardState.NOT_FINISHED
            else [
                (R, C)
                for R in range(3)
                for C in range(3)
                if board[R][C].get_game_state() == BoardState.NOT_FINISHED
            ]
        )

    for R, C in outers:
        inner = board[R][C]
        for r in range(3):
            for c in range(3):
                if inner[r][c] == Piece.EMPTY:
                    moves.append(Move(piece, (R, C), (r, c)))
    return moves


def swap_piece(p: Piece) -> Piece:
    return Piece.X if p == Piece.O else Piece.O


def get_restriction(board: Board, move: Move) -> Tuple[int, int] | None:
    target = board[move.inner[0]][move.inner[1]]
    return move.inner if target.get_game_state() == BoardState.NOT_FINISHED else None
