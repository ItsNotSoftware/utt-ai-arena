from __future__ import annotations
from abc import ABC, abstractmethod
from time import perf_counter
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
        self._move_count = 0
        self._move_time_total = 0.0

    def get_name(self) -> str:
        return f"{'X' if self.piece == Piece.X else 'O'} – {self.name}"

    def get_move(self, board: Board) -> Move | None:
        start = perf_counter()
        move = self._select_move(board)
        duration = perf_counter() - start
        self._move_count += 1
        self._move_time_total += duration
        avg = self._move_time_total / self._move_count
        print(
            f"{self.get_name()} move time: {duration:.3f}s "
            f"(avg {avg:.3f}s move{'s' if self._move_count != 1 else ''})"
        )
        return move

    @abstractmethod
    def _select_move(self, board: Board) -> Move | None: ...


class HumanPlayer(Player):
    """Mouse-based human."""

    def __init__(self, piece: Piece, depth_limit: int = 0) -> None:
        super().__init__(piece, depth_limit)
        self._prev_down = False
        self.name = "HumanPlayer"

    def _select_move(self, board: Board) -> Move | None:
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

    # Heuristic scores
    heuristics = {
        "win": 1_000,
        "draw": 0,
        "two_in_row_outer": 100,
        "inner_win": 50,
        "two_in_row_inner": 10,
        "center_corner": 3,
    }

    def __init__(
        self,
        piece: Piece,
        depth_limit: int = 0,
        use_heuristic_eval=True,
        use_pruning=True,
    ) -> None:
        super().__init__(piece, depth_limit=depth_limit)
        self.name = "Minimax"
        features = []
        if use_heuristic_eval:
            features.append("heuristic")
        if use_pruning:
            features.append("pruning")
        if features:
            self.name += " (" + ", ".join(features) + ")"

        self.use_heuristic_eval = use_heuristic_eval

    def _evaluate_board(self, board: Board) -> float:
        boards = [board[i][j] for i in range(3) for j in range(3)]
        heur = self.heuristics
        score = 0.0

        # Score utilities
        def inner_line_score(values: list[Piece]) -> float:
            """Two-in-a-row patterns inside an inner board."""
            x_count = values.count(Piece.X)
            o_count = values.count(Piece.O)
            empty_count = values.count(Piece.EMPTY)

            if x_count == 2 and o_count == 0 and empty_count == 1:
                return heur["two_in_row_inner"]
            if o_count == 2 and x_count == 0 and empty_count == 1:
                return -heur["two_in_row_inner"]
            return 0.0

        def outer_line_score(values: list[Piece]) -> float:
            """Two-in-a-row patterns across inner board results."""
            x_count = values.count(Piece.X)
            o_count = values.count(Piece.O)
            empty_count = values.count(Piece.EMPTY)

            if x_count == 2 and o_count == 0 and empty_count == 1:
                return heur["two_in_row_outer"]
            if o_count == 2 and x_count == 0 and empty_count == 1:
                return -heur["two_in_row_outer"]
            return 0.0

        important_positions = {(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)}

        # Outer board heuristic: based on inner board outcomes
        outer_values = [
            [
                board[r][c].value if isinstance(board[r][c], Board) else Piece.EMPTY
                for c in range(3)
            ]
            for r in range(3)
        ]

        for r in range(3):
            score += outer_line_score([outer_values[r][c] for c in range(3)])
            score += outer_line_score([outer_values[c][r] for c in range(3)])
        score += outer_line_score([outer_values[i][i] for i in range(3)])
        score += outer_line_score([outer_values[2 - i][i] for i in range(3)])

        # Reward controlling critical inner boards (outer center/corners)
        for r in range(3):
            for c in range(3):
                inner_board = board[r][c]
                if not isinstance(inner_board, Board):
                    continue
                value = inner_board.value
                if (r, c) in important_positions:
                    if value == Piece.X:
                        score += heur["center_corner"]
                    elif value == Piece.O:
                        score -= heur["center_corner"]

        # Evaluate each inner board
        for inner in boards:
            if not isinstance(inner, Board):
                continue

            if inner.board_state == BoardState.X_WON:
                score += heur["inner_win"]
            elif inner.board_state == BoardState.O_WON:
                score -= heur["inner_win"]

            # Reward center/corner occupancy inside inner boards
            for r in range(3):
                for c in range(3):
                    piece = inner[r][c]
                    if piece == Piece.EMPTY:
                        continue
                    if (r, c) in important_positions:
                        if piece == Piece.X:
                            score += heur["center_corner"]
                        elif piece == Piece.O:
                            score -= heur["center_corner"]

            if inner.board_state != BoardState.NOT_FINISHED:
                continue

            # Two-in-a-row patterns inside an unfinished inner board
            for r in range(3):
                row_values = [inner[r][c] for c in range(3)]
                col_values = [inner[c][r] for c in range(3)]
                score += inner_line_score(row_values)
                score += inner_line_score(col_values)

            score += inner_line_score([inner[i][i] for i in range(3)])
            score += inner_line_score([inner[2 - i][i] for i in range(3)])

        return score

    def _minmax(
        self,
        piece: Piece,
        board: Board,
        depth: int,
        depth_limit: int,
    ) -> float:
        # Terminal?
        if board.board_state == BoardState.DRAW:
            return self.heuristics["draw"]
        elif board.board_state == BoardState.X_WON:
            return self.heuristics["win"]
        elif board.board_state == BoardState.O_WON:
            return -self.heuristics["win"]

        # Depth limit
        if depth >= depth_limit:
            return self._evaluate_board(board) if self.use_heuristic_eval else 0.0

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

    def _select_move(self, board: Board) -> Move | None:
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
