from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass
from typing import Callable, Union, Any, Tuple, List, Set


class Piece(IntEnum):
    EMPTY = 0
    O = -1
    X = 1


@dataclass(slots=True)
class Move:
    piece: Piece
    outer: Tuple[int, int]  # outer board position select
    inner: Tuple[int, int]  # inner board position select


@dataclass(slots=True)
class UndoToken:
    """Minimal info to undo a move (for bot exploration)."""

    outer: Tuple[int, int]
    inner: Tuple[int, int]
    prev_inner_state: "BoardState"
    prev_main_state: "BoardState"
    prev_restriction: Tuple[int, int] | None


class BoardState(IntEnum):
    NOT_FINISHED = 0
    DRAW = 1
    O_WON = 2
    X_WON = 3


def board_state_to_piece(state: BoardState) -> Piece:
    """Map a finished board state to a piece; unfinished/draw => EMPTY."""
    if state == BoardState.X_WON:
        return Piece.X
    if state == BoardState.O_WON:
        return Piece.O
    return Piece.EMPTY


def board_state_to_value(state: BoardState) -> int:
    """Map a finished board state to +1/-1; unfinished/draw => 0."""
    if state == BoardState.X_WON:
        return 1
    if state == BoardState.O_WON:
        return -1
    return 0


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

        # Detect type: this instance is an inner board if cells are Piece
        self.is_inner: bool = isinstance(self.board[0][0], Piece)

        # For main board:
        self.restriction: Tuple[int, int] | None = None  # next required outer (or None)
        self.playable_outers_list: List[Tuple[int, int]] = []
        self.playable_outers_set: Set[Tuple[int, int]] = set()

        # For inner boards: track empty cells to speed legal move gen
        self.empty_cells: Set[Tuple[int, int]] = set()

        if self.is_inner:
            self.empty_cells = {(r, c) for r in range(3) for c in range(3)}  # all empty
        else:
            # main board starts with all 9 outers playable
            self._refresh_playable_outers()

    def __getitem__(self, idx: int) -> Any:
        return self.board[idx]

    def __setitem__(self, idx: int, value) -> None:
        self.board[idx] = value

    def _refresh_playable_outers(self) -> None:
        """Recomputes playable outer boards (NOT_FINISHED)."""
        self.playable_outers_list = []
        self.playable_outers_set.clear()
        for r in range(3):
            for c in range(3):
                cell = self.board[r][c]
                if (
                    isinstance(cell, Board)
                    and cell.board_state == BoardState.NOT_FINISHED
                ):
                    self.playable_outers_list.append((r, c))
                    self.playable_outers_set.add((r, c))

    def _remove_outer_if_finished(self, rc: Tuple[int, int]) -> None:
        """Removes outer from playable sets if it just finished."""
        if rc in self.playable_outers_set:
            self.playable_outers_set.remove(rc)
            self.playable_outers_list = [
                x for x in self.playable_outers_list if x != rc
            ]

    def _add_outer_if_playable(self, rc: Tuple[int, int]) -> None:
        """Adds outer to playable sets if it is NOT_FINISHED."""
        if rc not in self.playable_outers_set:
            self.playable_outers_set.add(rc)
            self.playable_outers_list.append(rc)

    def _update_restriction(self, move: Move | None) -> None:
        """Updates the next restriction after a move."""
        if move is None or self.is_inner:
            return
        target = self.board[move.inner[0]][move.inner[1]]
        self.restriction = (
            move.inner if target.board_state == BoardState.NOT_FINISHED else None
        )

    def place_piece(self, l: int, c: int, p: Piece) -> bool:
        """Places a piece on the board at the specified location."""
        if (
            self.board[l][c] == Piece.EMPTY
            and self.board_state == BoardState.NOT_FINISHED
        ):
            self.board[l][c] = p
            if self.is_inner:
                # keep empty-cells in sync
                self.empty_cells.discard((l, c))
            self.board_state = self.get_game_state()
            return True
        return False

    def make_move(self, move: Move) -> UndoToken | None:
        """Applies a move. Enforces restriction and inner state. Updates next restriction. Returns UndoToken."""
        out_rc, in_rc = move.outer, move.inner

        # restriction (None means free choice)
        if (
            not self.is_inner
            and self.restriction is not None
            and out_rc != self.restriction
        ):
            return None

        # inner must be playable + cell must be empty
        inner = self.board[out_rc[0]][out_rc[1]]
        if not isinstance(inner, Board):
            return None
        if inner.board_state != BoardState.NOT_FINISHED:
            return None
        if inner[in_rc[0]][in_rc[1]] != Piece.EMPTY:
            return None

        token = UndoToken(
            outer=out_rc,
            inner=in_rc,
            prev_inner_state=inner.board_state,
            prev_main_state=self.board_state,
            prev_restriction=self.restriction,
        )

        ok = inner.place_piece(in_rc[0], in_rc[1], move.piece)
        if not ok:
            return None

        # If inner flipped to finished, remove from playable outers
        if not self.is_inner:
            if (
                inner.board_state != BoardState.NOT_FINISHED
                and token.prev_inner_state == BoardState.NOT_FINISHED
            ):
                self._remove_outer_if_finished(out_rc)

        # Update next restriction and main state
        self._update_restriction(move)
        self.board_state = self.get_game_state()
        return token

    def undo_move(self, token: UndoToken) -> None:
        """Undo a move previously done with make_move()."""
        out_rc, in_rc = token.outer, token.inner
        inner = self.board[out_rc[0]][out_rc[1]]
        if not isinstance(inner, Board):
            return

        # restore piece to EMPTY
        inner[in_rc[0]][in_rc[1]] = Piece.EMPTY
        if inner.is_inner:
            inner.empty_cells.add(in_rc)

        # restore states + restriction
        inner.board_state = token.prev_inner_state
        self.board_state = token.prev_main_state
        self.restriction = token.prev_restriction

        # If inner became NOT_FINISHED again, re-add to playable outers
        if not self.is_inner:
            if (
                inner.board_state == BoardState.NOT_FINISHED
                and out_rc not in self.playable_outers_set
            ):
                self._add_outer_if_playable(out_rc)

    def get_game_state(self) -> BoardState:
        """Computes the state of this board (win/draw/playing)."""
        x_won = 3
        o_won = -3
        b = self.board

        if self.is_inner:
            # lines + columns
            for i in range(3):
                line_sum = b[i][0].value + b[i][1].value + b[i][2].value
                col_sum = b[0][i].value + b[1][i].value + b[2][i].value
                if line_sum == x_won or col_sum == x_won:
                    return BoardState.X_WON
                if line_sum == o_won or col_sum == o_won:
                    return BoardState.O_WON

            # diagonals
            diag1 = b[0][0].value + b[1][1].value + b[2][2].value
            diag2 = b[2][0].value + b[1][1].value + b[0][2].value
            if diag1 == x_won or diag2 == x_won:
                return BoardState.X_WON
            if diag1 == o_won or diag2 == o_won:
                return BoardState.O_WON

            return BoardState.NOT_FINISHED if self.empty_cells else BoardState.DRAW

        # lines + columns
        for i in range(3):
            line_sum = (
                board_state_to_value(b[i][0].board_state)
                + board_state_to_value(b[i][1].board_state)
                + board_state_to_value(b[i][2].board_state)
            )
            col_sum = (
                board_state_to_value(b[0][i].board_state)
                + board_state_to_value(b[1][i].board_state)
                + board_state_to_value(b[2][i].board_state)
            )
            if line_sum == x_won or col_sum == x_won:
                return BoardState.X_WON
            if line_sum == o_won or col_sum == o_won:
                return BoardState.O_WON

        # diagonals
        diag1 = (
            board_state_to_value(b[0][0].board_state)
            + board_state_to_value(b[1][1].board_state)
            + board_state_to_value(b[2][2].board_state)
        )
        diag2 = (
            board_state_to_value(b[2][0].board_state)
            + board_state_to_value(b[1][1].board_state)
            + board_state_to_value(b[0][2].board_state)
        )
        if diag1 == x_won or diag2 == x_won:
            return BoardState.X_WON
        if diag1 == o_won or diag2 == o_won:
            return BoardState.O_WON

        any_empty = any(
            b[r][c].board_state == BoardState.NOT_FINISHED
            for r in range(3)
            for c in range(3)
        )
        return BoardState.NOT_FINISHED if any_empty else BoardState.DRAW

    def clone(self) -> Board:
        """Deep copy (keeps inner boards, restriction, and playable outers/empties)."""
        new_board = Board(piece_factory=lambda: Piece.EMPTY)
        new_board.board_state = self.board_state
        new_board.is_inner = self.is_inner
        new_board.restriction = self.restriction

        for i in range(3):
            for j in range(3):
                cell = self.board[i][j]
                if isinstance(cell, Board):
                    new_board.board[i][j] = cell.clone()
                else:
                    new_board.board[i][j] = cell

        if self.is_inner:
            new_board.empty_cells = set(self.empty_cells)
        else:
            new_board._refresh_playable_outers()
        return new_board


def get_board() -> Board:
    """Returns a main board composed of 9 inner boards."""
    return Board(piece_factory=lambda: Board())


def legal_moves(
    board: Board, piece: Piece, restriction: Tuple[int, int] | None = None
) -> list[Move]:
    """All valid moves given a restriction (defaults to board.restriction)."""
    moves: list[Move] = []

    # choose outers
    use_restr = board.restriction if restriction is None else restriction
    if use_restr is not None and use_restr in board.playable_outers_set:
        outers = [use_restr]
    else:
        outers = board.playable_outers_list

    append = moves.append
    for R, C in outers:
        inner = board[R][C]
        if inner.board_state != BoardState.NOT_FINISHED:
            continue
        for r, c in inner.empty_cells:
            append(Move(piece, (R, C), (r, c)))
    return moves


def swap_piece(p: Piece) -> Piece:
    """Swap X/O."""
    return Piece.X if p == Piece.O else Piece.O
