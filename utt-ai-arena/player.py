from abc import ABC, abstractmethod
from board import Board, Piece, Move, legal_moves, BoardState, swap_piece
from pygame import mouse
import math

# Layout (set from main)
_screen_w = 0
_screen_h = 0
_board_size = 0
_board_left = 0
_board_top = 0


def set_layout(
    screen_w: int, screen_h: int, board_size: int, board_left: int, board_top: int
) -> None:
    global _screen_w, _screen_h, _board_size, _board_left, _board_top
    _screen_w, _screen_h = screen_w, screen_h
    _board_size, _board_left, _board_top = board_size, board_left, board_top


class Player(ABC):
    def __init__(self, piece: Piece, depth_limit: int = 0) -> None:
        self.piece = piece
        self.depth_limit = depth_limit
        self.name = "Player"

    def get_name(self) -> str:
        return f"{'X' if self.piece == Piece.X else 'O'} – {self.name}"

    @abstractmethod
    def get_move(self, board: Board) -> Move | None: ...


class HumanPlayer(Player):
    def __init__(self, piece: Piece, depth_limit: int = 0) -> None:
        super().__init__(piece)
        self._prev_down = False
        self.name = "HumanPlayer"

    def get_move(self, board: Board) -> Move | None:
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
    def __init__(
        self, piece: Piece, depth_limit: int = 0, use_heuristics: bool = False
    ) -> None:
        super().__init__(piece)
        self._prev_down = False
        self.name = "Minmax" if not use_heuristics else "HeuristicMinmax"

    @staticmethod
    def minmax(piece: Piece, board: Board, depth: int, depth_limit: int) -> float:
        if board.board_state == BoardState.DRAW:
            return 0.0
        elif board.board_state == BoardState.X_WON:
            return math.inf
        elif board.board_state == BoardState.O_WON:
            return -math.inf

        piece = swap_piece(piece)
        moves = legal_moves()

    def get_move(self, board: Board) -> Move | None:

        pass
