import random
import pygame
from board import get_board, Piece, BoardState, Board, Piece
from player import Move, Player, HumanPlayer, set_screen_size
import time

# --- Constants ---
SCREEN_SIZE = 1280
MARGIN = 20

# --- Setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
clock = pygame.time.Clock()


def render_inner_board(board: Board, size: int) -> pygame.Surface:
    """Renders a single inner board of the Tic Tac Toe game."""
    surface = pygame.Surface((size, size))
    surface.fill("white")

    cell_size = size // 3
    line_color = pygame.Color("black")
    x_color = pygame.Color("red")
    o_color = pygame.Color("blue")

    # Draw the 2 vertical and 2 horizontal grid lines
    for i in range(1, 3):
        x = i * cell_size
        pygame.draw.line(surface, line_color, (x, MARGIN), (x, size - MARGIN), width=3)
        pygame.draw.line(surface, line_color, (MARGIN, x), (size - MARGIN, x), width=3)

    # Draw each piece in its cell
    for i in range(3):
        for j in range(3):
            piece = board[i][j]
            if piece == Piece.EMPTY:
                continue

            # Center of this cell
            cx = j * cell_size + cell_size // 2
            cy = i * cell_size + cell_size // 2
            # How far from the center the strokes/radius go
            pad = cell_size // 2 - MARGIN

            if piece == Piece.X:
                # draw two crossing lines
                pygame.draw.line(
                    surface,
                    x_color,
                    (cx - pad, cy - pad),
                    (cx + pad, cy + pad),
                    width=6,
                )
                pygame.draw.line(
                    surface,
                    x_color,
                    (cx - pad, cy + pad),
                    (cx + pad, cy - pad),
                    width=6,
                )
            else:  # Piece.O
                # draw circle
                pygame.draw.circle(
                    surface,
                    o_color,
                    (cx, cy),
                    pad,
                    width=6,
                )

    return surface


def draw_main_board(screen: pygame.Surface, board: Board) -> None:
    """Draws the main board composed of 9 inner boards."""

    inner_size = SCREEN_SIZE // 3
    line_color = pygame.Color("black")
    x_color = pygame.Color("red")
    o_color = pygame.Color("blue")

    # Draw each inner board
    for row in range(3):
        for col in range(3):
            x = col * inner_size
            y = row * inner_size
            inner_surface = render_inner_board(board[row][col], inner_size)
            screen.blit(inner_surface, (x, y))

    # Draw a X or O over any inner boards that's already been won
    for row in range(3):
        for col in range(3):
            inner = board[row][col]
            val = inner.value
            if val == Piece.EMPTY:
                continue

            # compute center of this big cell
            x0 = col * inner_size
            y0 = row * inner_size
            cx = x0 + inner_size // 2
            cy = y0 + inner_size // 2
            pad = inner_size // 2 - MARGIN * 2

            if val == Piece.X:
                pygame.draw.line(
                    screen,
                    x_color,
                    (cx - pad, cy - pad),
                    (cx + pad, cy + pad),
                    width=12,
                )
                pygame.draw.line(
                    screen,
                    x_color,
                    (cx - pad, cy + pad),
                    (cx + pad, cy - pad),
                    width=12,
                )
            else:  # Piece.O
                pygame.draw.circle(
                    screen,
                    o_color,
                    (cx, cy),
                    pad,
                    width=12,
                )

    #  Draw main-grid lines
    for i in range(1, 3):
        # horizontal
        pygame.draw.line(
            screen,
            line_color,
            (MARGIN, i * inner_size),
            (SCREEN_SIZE - MARGIN, i * inner_size),
            width=6,
        )
        # vertical
        pygame.draw.line(
            screen,
            line_color,
            (i * inner_size, MARGIN),
            (i * inner_size, SCREEN_SIZE - MARGIN),
            width=6,
        )


def make_move(board: Board, move: Move) -> bool:
    piece = move.piece
    out_m, in_m = move.outer, move.inner

    l, c = out_m
    inner_b = board[l][c]

    if inner_b.get_game_state() != BoardState.NOT_FINISHED:
        print(inner_b.get_game_state())
        return False

    l, c = in_m
    status = inner_b.place_piece(l, c, piece)
    return status


def game_loop() -> None:
    """Main game loop that handles events and renders the game state."""

    board = get_board()

    # TODO player selection
    p1 = HumanPlayer(Piece.X)
    p2 = HumanPlayer(Piece.O)
    player = p1

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        screen.fill("white")
        draw_main_board(screen, board)

        move = player.get_move(board)

        if move:
            # repeat loop if an invalid move is given
            if not make_move(board, move):
                print("Invalid move!")
                continue

            player = p1 if player == p2 else p2  # change player
            time.sleep(0.2)  # Give time for player to unclick mouse

        pygame.display.flip()
        clock.tick(60)


def main() -> None:
    set_screen_size(SCREEN_SIZE)
    game_loop()
    pygame.quit()


if __name__ == "__main__":
    main()
