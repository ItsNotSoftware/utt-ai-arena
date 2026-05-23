from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass
from typing import Callable, Union, Any, Tuple, List, Set
import random


class Piece(IntEnum):
    EMPTY = 0
    O = -1
    X = 1


@dataclass(slots=True, frozen=True)
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
    removed_outer_index: int | None = None


class BoardState(IntEnum):
    NOT_FINISHED = 0
    DRAW = 1
    O_WON = 2
    X_WON = 3


WIN_LINES = (
    ((0, 0), (0, 1), (0, 2)),
    ((1, 0), (1, 1), (1, 2)),
    ((2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 1), (2, 1)),
    ((0, 2), (1, 2), (2, 2)),
    ((0, 0), (1, 1), (2, 2)),
    ((2, 0), (1, 1), (0, 2)),
)


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

    __slots__ = (
        "board",
        "board_state",
        "is_inner",
        "restriction",
        "playable_outers_list",
        "playable_outers_set",
        "empty_cells",
    )

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
        if inner.board[in_rc[0]][in_rc[1]] != Piece.EMPTY:
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
                if out_rc in self.playable_outers_set:
                    token.removed_outer_index = self.playable_outers_list.index(out_rc)
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
        inner.board[in_rc[0]][in_rc[1]] = Piece.EMPTY
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
                self.playable_outers_set.add(out_rc)
                if (
                    token.removed_outer_index is None
                    or token.removed_outer_index >= len(self.playable_outers_list)
                ):
                    self.playable_outers_list.append(out_rc)
                else:
                    self.playable_outers_list.insert(token.removed_outer_index, out_rc)

    def get_game_state(self) -> BoardState:
        """Computes the state of this board (win/draw/playing)."""
        b = self.board

        if self.is_inner:
            for (a, b1, c) in WIN_LINES:
                p = b[a[0]][a[1]]
                if p != Piece.EMPTY and p == b[b1[0]][b1[1]] == b[c[0]][c[1]]:
                    return BoardState.X_WON if p == Piece.X else BoardState.O_WON

            return BoardState.NOT_FINISHED if self.empty_cells else BoardState.DRAW

        for (a, b1, c) in WIN_LINES:
            st = b[a[0]][a[1]].board_state
            if (
                (st == BoardState.X_WON or st == BoardState.O_WON)
                and st == b[b1[0]][b1[1]].board_state
                and st == b[c[0]][c[1]].board_state
            ):
                return st

        any_empty = any(
            b[r][c].board_state == BoardState.NOT_FINISHED
            for r in range(3)
            for c in range(3)
        )
        return BoardState.NOT_FINISHED if any_empty else BoardState.DRAW

    def clone(self) -> Board:
        """Deep copy (keeps inner boards, restriction, and playable outers/empties)."""
        new_board = Board.__new__(Board)
        new_board.board_state = self.board_state
        new_board.is_inner = self.is_inner
        new_board.restriction = self.restriction

        if self.is_inner:
            # Shallow copy is fine: cells are immutable Piece enums
            new_board.board = [row[:] for row in self.board]
            new_board.empty_cells = set(self.empty_cells)
            new_board.playable_outers_list = []
            new_board.playable_outers_set = set()
        else:
            new_board.board = [
                [self.board[i][j].clone() for j in range(3)] for i in range(3)
            ]
            new_board.playable_outers_list = list(self.playable_outers_list)
            new_board.playable_outers_set = set(self.playable_outers_set)
            new_board.empty_cells = set()
        return new_board

    def legal_moves(
        self, piece: Piece, restriction: Tuple[int, int] | None = None
    ) -> list[Move]:
        """All valid moves given a restriction (defaults to self.restriction)."""
        if self.board_state != BoardState.NOT_FINISHED:
            return []

        moves: list[Move] = []

        # choose outers
        use_restr = self.restriction if restriction is None else restriction
        if use_restr is not None and use_restr in self.playable_outers_set:
            outers = [use_restr]
        else:
            outers = self.playable_outers_list

        append = moves.append
        for R, C in outers:
            inner = self.board[R][C]
            if inner.board_state != BoardState.NOT_FINISHED:
                continue
            for r, c in inner.empty_cells:
                append(Move(piece, (R, C), (r, c)))
        return moves

    def random_legal_move(
        self, piece: Piece, restriction: Tuple[int, int] | None = None
    ) -> Move | None:
        """Choose a uniformly random valid move without building the full move list."""
        if self.board_state != BoardState.NOT_FINISHED:
            return None

        use_restr = self.restriction if restriction is None else restriction
        if use_restr is not None and use_restr in self.playable_outers_set:
            outers = (use_restr,)
        else:
            outers = self.playable_outers_list

        total = 0
        board = self.board
        for R, C in outers:
            inner = board[R][C]
            if inner.board_state == BoardState.NOT_FINISHED:
                total += len(inner.empty_cells)
        if total == 0:
            return None

        idx = random.randrange(total)
        for R, C in outers:
            inner = board[R][C]
            if inner.board_state != BoardState.NOT_FINISHED:
                continue
            n = len(inner.empty_cells)
            if idx >= n:
                idx -= n
                continue
            for r, c in inner.empty_cells:
                if idx == 0:
                    return Move(piece, (R, C), (r, c))
                idx -= 1
        return None

    def packed_key(self, turn: Piece | None = None) -> int:
        """Compact integer key for fast transposition-table lookups."""
        val = 0
        if self.is_inner:
            for r in range(3):
                row = self.board[r]
                for c in range(3):
                    val = val * 3 + row[c].value + 1
        else:
            for R in range(3):
                for C in range(3):
                    inner = self.board[R][C]
                    for r in range(3):
                        row = inner.board[r]
                        for c in range(3):
                            val = val * 3 + row[c].value + 1
            restr = (
                0
                if self.restriction is None
                else self.restriction[0] * 3 + self.restriction[1] + 1
            )
            val = val * 10 + restr

        if turn is not None:
            val = val * 3 + turn.value + 1
        return val

    def key(self, turn: Piece | None = None) -> tuple:
        """Immutable key for hashing/caching. Optionally include turn."""
        if self.is_inner:
            # 9 cells as piece values
            cells = tuple(self.board[r][c].value for r in range(3) for c in range(3))
            base = ("inner",) + cells
        else:
            # main board: restriction + all inner boards
            restriction = self.restriction if self.restriction is not None else (-1, -1)
            inners = tuple(
                self.board[r][c].board_key() for r in range(3) for c in range(3)
            )
            base = ("main", restriction) + inners

        if turn is None:
            return base
        return ("turn", turn.value) + base

    def board_key(self) -> tuple:
        """Alias for key() to keep hash usage consistent."""
        return self.key()

    @staticmethod
    def from_key(key: tuple) -> tuple[Board, Piece | None]:
        """Rebuild a board from a board_key (recomputes board_state). Returns (board, turn)."""
        if not key:
            raise ValueError("Empty board key")

        tag = key[0]
        if tag == "turn":
            if len(key) < 3:
                raise ValueError("Invalid turn-prefixed key length")
            turn = Piece(key[1])
            board, _ = Board.from_key(key[2:])
            return board, turn
        if tag == "inner":
            if len(key) != 10:
                raise ValueError("Invalid inner board key length")
            b = Board(piece_factory=lambda: Piece.EMPTY)
            b.is_inner = True
            idx = 1
            for r in range(3):
                for c in range(3):
                    b.board[r][c] = Piece(key[idx])
                    idx += 1
            b.empty_cells = {
                (r, c)
                for r in range(3)
                for c in range(3)
                if b.board[r][c] == Piece.EMPTY
            }
            b.board_state = b.get_game_state()
            return b, None

        if tag == "main":
            if len(key) != 11:
                raise ValueError("Invalid main board key length")
            b = Board(piece_factory=lambda: Board())
            b.is_inner = False
            restriction = key[1]
            b.restriction = None if restriction == (-1, -1) else restriction
            idx = 2
            for r in range(3):
                for c in range(3):
                    inner, _ = Board.from_key(key[idx])
                    b.board[r][c] = inner
                    idx += 1
            b._refresh_playable_outers()
            b.board_state = b.get_game_state()
            return b, None

        raise ValueError("Unknown board key tag")


def get_board() -> Board:
    """Returns a main board composed of 9 inner boards."""
    return Board(piece_factory=lambda: Board())


def swap_piece(p: Piece) -> Piece:
    """Swap X/O."""
    return Piece.X if p == Piece.O else Piece.O
