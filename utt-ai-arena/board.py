from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass
from typing import Callable, Union, Any, Tuple, List


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
        self.board: List[List[Union[Piece, "Board"]]] = [
            [piece_factory() for _ in range(3)] for _ in range(3)
        ]
        self.board_state: BoardState = BoardState.NOT_FINISHED
        self.restriction: Tuple[int, int] | None = None  # next required outer (or None)

    def __getitem__(self, idx: int) -> Any:
        return self.board[idx]

    def __setitem__(self, idx: int, value) -> None:
        self.board[idx] = value

    def _update_restriction(self, move: Move | None) -> None:
        """Updates the next restriction after a move."""
        if move is None:
            return
        target = self.board[move.inner[0]][move.inner[1]]
        if isinstance(target, Piece):
            self.restriction = None
            return
        self.restriction = (
            move.inner if target.get_game_state() == BoardState.NOT_FINISHED else None
        )

    @property
    def value(self) -> Piece:
        """(⚆ᗝ⚆) Supper cool hack to treat inner boards like pieces: X if X won, O if O won, EMPTY otherwise."""
        match self.board_state:
            case BoardState.X_WON:
                return Piece.X
            case BoardState.O_WON:
                return Piece.O
            case _:
                return Piece.EMPTY

    def place_piece(self, l: int, c: int, p: Piece) -> bool:
        """Places a piece on the board at the specified location."""
        if (
            self.board[l][c] == Piece.EMPTY
            and self.board_state == BoardState.NOT_FINISHED
        ):
            self.board[l][c] = p
            self.board_state = self.get_game_state()
            return True
        return False

    def make_move(self, move: Move) -> bool:
        """Applies a move. Enforces restriction and inner state. Updates next restriction."""
        out_rc, in_rc = move.outer, move.inner

        # restriction (None means free choice)
        if self.restriction is not None and out_rc != self.restriction:
            return False

        # inner must be playable
        inner = self.board[out_rc[0]][out_rc[1]]
        if inner.get_game_state() != BoardState.NOT_FINISHED:
            return False

        # place
        ok = inner.place_piece(in_rc[0], in_rc[1], move.piece)
        if ok:
            self._update_restriction(move)
            # refresh main board state (inner .value may have changed)
            self.board_state = self.get_game_state()
        return ok

    def get_game_state(self) -> BoardState:
        """Computes the state of this board (win/draw/playing)."""
        x_won = 3
        o_won = -3

        # lines + columns
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
        def is_empty(cell: Union[Piece, "Board"]) -> bool:
            if isinstance(cell, Piece):
                return cell == Piece.EMPTY
            return cell.board_state == BoardState.NOT_FINISHED

        any_empty = any(is_empty(self.board[r][c]) for r in range(3) for c in range(3))
        return BoardState.NOT_FINISHED if any_empty else BoardState.DRAW

    def clone(self) -> Board:
        """Deep copy (keeps inner boards and restriction)."""
        new_board = Board(piece_factory=lambda: Piece.EMPTY)
        new_board.board_state = self.board_state
        new_board.restriction = self.restriction
        for i in range(3):
            for j in range(3):
                cell = self.board[i][j]
                new_board.board[i][j] = (
                    cell.clone() if isinstance(cell, Board) else cell
                )
        return new_board


def get_board() -> Board:
    """Returns a main board composed of 9 inner boards."""
    return Board(piece_factory=lambda: Board())


def legal_moves(
    board: Board, piece: Piece, restriction: Tuple[int, int] | None = None
) -> list[Move]:
    """All valid moves given a restriction (defaults to board.restriction)."""
    moves: list[Move] = []
    use_restr = board.restriction if restriction is None else restriction

    if use_restr is None:
        outers = [
            (R, C)
            for R in range(3)
            for C in range(3)
            if board[R][C].get_game_state() == BoardState.NOT_FINISHED
        ]
    else:
        if (
            board[use_restr[0]][use_restr[1]].get_game_state()
            == BoardState.NOT_FINISHED
        ):
            outers = [use_restr]
        else:
            outers = [
                (R, C)
                for R in range(3)
                for C in range(3)
                if board[R][C].get_game_state() == BoardState.NOT_FINISHED
            ]

    for R, C in outers:
        inner = board[R][C]
        for r in range(3):
            for c in range(3):
                if inner[r][c] == Piece.EMPTY:
                    moves.append(Move(piece, (R, C), (r, c)))
    return moves


def swap_piece(p: Piece) -> Piece:
    """Swap X/O."""
    return Piece.X if p == Piece.O else Piece.O
