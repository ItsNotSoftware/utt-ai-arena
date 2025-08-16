from abc import ABC, abstractmethod
from board import Board, Piece, Move
from pygame import mouse

screen_size = 0


def set_screen_size(sz: int) -> None:
    global screen_size
    screen_size = sz


class Player(ABC):
    """Abstract player class."""

    def __init__(self, piece: Piece) -> None:
        self.piece = piece

    @abstractmethod
    def get_move(self, board: Board) -> Move | None:
        pass


class HumanPlayer(Player):
    def get_move(self, board: Board) -> Move | None:
        down = mouse.get_pressed()[0]
        if not down:
            self._prev_down = False
            return None
        if self._prev_down:
            return None  # still held
        self._prev_down = True

        x, y = mouse.get_pos()
        big_square_sz = screen_size / 3
        small_square_sz = screen_size / 9
        out_l = int(y // big_square_sz)
        out_c = int(x // big_square_sz)
        in_l = int(y // small_square_sz) % 3
        in_c = int(x // small_square_sz) % 3
        return Move(self.piece, (out_l, out_c), (in_l, in_c))
