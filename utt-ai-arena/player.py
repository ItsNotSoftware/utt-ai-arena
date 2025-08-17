from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import math
import random

from board import (
    Board,
    BoardState,
    Move,
    Piece,
    legal_moves,
    swap_piece,
)

# Layout (set from main)
_screen_w = 0
_screen_h = 0
_board_size = 0
_board_left = 0
_board_top = 0


def set_layout(
    screen_w: int, screen_h: int, board_size: int, board_left: int, board_top: int
) -> None:
    """Sets layout info for input mapping."""
    global _screen_w, _screen_h, _board_size, _board_left, _board_top
    _screen_w, _screen_h = screen_w, screen_h
    _board_size, _board_left, _board_top = board_size, board_left, board_top


class Player(ABC):
    """Abstract player."""

    def __init__(self, piece: Piece, depth_limit: int = 0) -> None:
        self.piece = piece
        self.depth_limit = depth_limit
        self.name = "Player"

    def get_name(self) -> str:
        return f"{'X' if self.piece == Piece.X else 'O'} – {self.name}"

    @abstractmethod
    def get_move(self, board: Board) -> Move | None: ...


class HumanPlayer(Player):
    """Mouse-based human."""

    def __init__(self, piece: Piece, depth_limit: int = 0) -> None:
        super().__init__(piece, depth_limit)
        self._prev_down = False
        self.name = "HumanPlayer"

    def get_move(self, board: Board) -> Move | None:
        from pygame import mouse

        # single click
        down = mouse.get_pressed()[0]
        if not down:
            self._prev_down = False
            return None
        if self._prev_down:
            return None
        self._prev_down = True

        x, y = mouse.get_pos()

        # Must be inside the board square
        if not (
            _board_left <= x < _board_left + _board_size
            and _board_top <= y < _board_top + _board_size
        ):
            return None

        # local coords
        lx = x - _board_left
        ly = y - _board_top

        big = _board_size / 3
        small = _board_size / 9

        out_l = int(ly // big)
        out_c = int(lx // big)
        in_l = int(ly // small) % 3
        in_c = int(lx // small) % 3

        return Move(self.piece, (out_l, out_c), (in_l, in_c))


class MinmaxPlayer(Player):
    """Pure minimax (no pruning, no heuristic) using apply/undo."""

    def __init__(self, piece: Piece, depth_limit: int = 0) -> None:
        super().__init__(piece, depth_limit=depth_limit)
        self.name = "Minmax"

    def _minmax(
        self,
        piece: Piece,
        board: Board,
        depth: int,
        depth_limit: int,
    ) -> float:
        # Terminal?
        if board.board_state == BoardState.DRAW:
            return 0.0
        elif board.board_state == BoardState.X_WON:
            return math.inf
        elif board.board_state == BoardState.O_WON:
            return -math.inf

        # Depth limit
        if depth >= depth_limit:
            return 0.0

        # Children
        moves = legal_moves(board, piece, board.restriction)
        if not moves:
            return 0.0

        maximizing = piece == Piece.X
        best = -math.inf if maximizing else math.inf

        for m in moves:
            token = board.make_move(m)
            if token is None:
                continue  # should not happen with legal_moves
            score = self._minmax(swap_piece(piece), board, depth + 1, depth_limit)
            board.undo_move(token)

            if maximizing:
                if score > best:
                    best = score
            else:
                if score < best:
                    best = score

        return best

    def get_move(self, board: Board) -> Move | None:
        # get legal moves
        moves = legal_moves(board, self.piece, board.restriction)

        # allow for random move selection when multiple have the same eval
        random.shuffle(moves)

        if not moves:
            return None

        maximizing = self.piece == Piece.X
        best_score = -math.inf if maximizing else math.inf
        best_move = moves[0]

        # Evaluate candidate moves sequentially
        for m in moves:
            token = board.make_move(m)
            if token is None:
                # Should not happen with legal_moves;
                continue
            score = self._minmax(swap_piece(self.piece), board, 1, self.depth_limit)
            board.undo_move(token)

            if maximizing:
                if score > best_score:
                    best_score, best_move = score, m
            else:
                if score < best_score:
                    best_score, best_move = score, m

        return best_move
