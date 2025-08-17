from __future__ import annotations
import time
from typing import Tuple

import pygame

from board import Board, BoardState, Piece, get_board
from player import HumanPlayer, MinmaxPlayer, Player, set_layout

# --- constants ---
SCREEN_SIZE = 1280
STATUS_BAR_H = 110
HEADER_H = 28  # small top header for column numbers
SIDEBAR_W = 28  # small left sidebar for row letters
BOARD_SIZE = SCREEN_SIZE - STATUS_BAR_H - HEADER_H
BOARD_LEFT = (SCREEN_SIZE - BOARD_SIZE) // 2
BOARD_TOP = HEADER_H  # push board down to make room for header

# --- colors ---
BG = pygame.Color(250, 251, 255)
BOARD_BG = pygame.Color(255, 255, 255)
BAR_BG = pygame.Color(243, 245, 250)
BORDER = pygame.Color(30, 30, 30)
LINE_COLOR = pygame.Color(50, 50, 55)
X_COLOR = pygame.Color(210, 30, 30)
O_COLOR = pygame.Color(25, 80, 220)
LBL_COLOR = pygame.Color(90, 95, 110)
WARN_COLOR = pygame.Color(200, 30, 30)

# --- grid thickness ---
GRID_THIN = 2
DIV_W = 6
INNER_MARGIN = 16  # margin inside each inner board

# --- fonts ---
FONT = None
FONT_BOLD = None

# --- setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
clock = pygame.time.Clock()
pygame.display.set_caption("Ultimate Tic-Tac-Toe")

FONT = pygame.font.SysFont(None, 32)
FONT_BOLD = pygame.font.SysFont(None, 40)


def draw_endgame_banner(screen: pygame.Surface, state: BoardState) -> None:
    """Draws a message across the top header when the game is finished."""
    # cover the header band
    header_rect = pygame.Rect(BOARD_LEFT, 0, BOARD_SIZE, HEADER_H)
    pygame.draw.rect(screen, BAR_BG, header_rect)
    pygame.draw.line(
        screen,
        BORDER,
        (BOARD_LEFT, HEADER_H - 1),
        (BOARD_LEFT + BOARD_SIZE, HEADER_H - 1),
        1,
    )

    if state == BoardState.X_WON:
        msg = "X wins!"
        col = X_COLOR
    elif state == BoardState.O_WON:
        msg = "O wins!"
        col = O_COLOR
    else:
        msg = "Draw"
        col = LBL_COLOR

    text = FONT_BOLD.render(msg, True, col)
    x = BOARD_LEFT + (BOARD_SIZE - text.get_width()) // 2
    y = (HEADER_H - text.get_height()) // 2
    screen.blit(text, (x, y))


def idx_to_label(rc: Tuple[int, int]) -> str:
    """(row, col) -> A1..C3."""
    r, c = rc
    return f"{chr(65 + r)}{c + 1}"


def render_inner_board(board: Board, size: int) -> pygame.Surface:
    """Renders a single inner board with margins."""
    surface = pygame.Surface((size, size))
    surface.fill(BOARD_BG)

    cell = size // 3
    m = INNER_MARGIN
    end = size - INNER_MARGIN

    # thin grid
    for i in range(1, 3):
        x = i * cell
        y = i * cell
        pygame.draw.line(
            surface, LINE_COLOR, (x, m), (x, end), width=GRID_THIN
        )  # vertical
        pygame.draw.line(
            surface, LINE_COLOR, (m, y), (end, y), width=GRID_THIN
        )  # horizontal

    # pieces
    for i in range(3):
        for j in range(3):
            piece = board[i][j]
            if piece == Piece.EMPTY:
                continue
            cx = j * cell + cell // 2
            cy = i * cell + cell // 2
            pad = cell // 2 - max(10, m - 2)
            if piece == Piece.X:
                pygame.draw.line(
                    surface,
                    X_COLOR,
                    (cx - pad, cy - pad),
                    (cx + pad, cy + pad),
                    width=6,
                )
                pygame.draw.line(
                    surface,
                    X_COLOR,
                    (cx - pad, cy + pad),
                    (cx + pad, cy - pad),
                    width=6,
                )
            else:
                pygame.draw.circle(surface, O_COLOR, (cx, cy), pad, width=6)

    return surface


def draw_labels(screen: pygame.Surface, big_cell: int) -> None:
    """Row/col labels with matching header + sidebar."""
    # top header band
    header_rect = pygame.Rect(BOARD_LEFT, 0, BOARD_SIZE, HEADER_H)
    pygame.draw.rect(screen, BAR_BG, header_rect)
    # bottom edge
    pygame.draw.line(
        screen,
        BORDER,
        (BOARD_LEFT, HEADER_H - 1),
        (BOARD_LEFT + BOARD_SIZE, HEADER_H - 1),
        1,
    )

    # left sidebar band
    sidebar_rect = pygame.Rect(BOARD_LEFT - SIDEBAR_W, BOARD_TOP, SIDEBAR_W, BOARD_SIZE)
    pygame.draw.rect(screen, BAR_BG, sidebar_rect)
    # right edge
    pygame.draw.line(
        screen,
        BORDER,
        (BOARD_LEFT - 1, BOARD_TOP),
        (BOARD_LEFT - 1, BOARD_TOP + BOARD_SIZE),
        1,
    )

    # column numbers
    for c in range(3):
        text = FONT.render(str(c + 1), True, LBL_COLOR)
        x = BOARD_LEFT + c * big_cell + big_cell // 2 - text.get_width() // 2
        y = HEADER_H // 2 - text.get_height() // 2
        screen.blit(text, (x, y))

    # row letters
    for r in range(3):
        text = FONT.render(chr(65 + r), True, LBL_COLOR)
        x = BOARD_LEFT - SIDEBAR_W // 2 - text.get_width() // 2
        y = BOARD_TOP + r * big_cell + big_cell // 2 - text.get_height() // 2
        screen.blit(text, (x, y))


def draw_main_board(
    screen: pygame.Surface, board: Board, restriction: Tuple[int, int] | None
) -> int:
    """Main board with tint, dividers, overlays and labels."""
    inner_size = BOARD_SIZE // 3
    board_rect = pygame.Rect(BOARD_LEFT, BOARD_TOP, BOARD_SIZE, BOARD_SIZE)

    # background + border
    pygame.draw.rect(screen, BOARD_BG, board_rect)
    pygame.draw.rect(screen, BORDER, board_rect, width=1)

    # inner boards
    for r in range(3):
        for c in range(3):
            x = BOARD_LEFT + c * inner_size
            y = BOARD_TOP + r * inner_size

            # restriction tint
            if restriction == (r, c):
                tint = pygame.Surface((inner_size, inner_size), pygame.SRCALPHA)
                tint.fill((255, 230, 120, 70))
                screen.blit(tint, (x, y))

            surface = render_inner_board(board[r][c], inner_size)
            screen.blit(surface, (x, y))

    # big dividers
    for i in range(1, 3):
        yy = BOARD_TOP + i * inner_size
        xx = BOARD_LEFT + i * inner_size
        pygame.draw.rect(
            screen,
            LINE_COLOR,
            pygame.Rect(BOARD_LEFT, yy - DIV_W // 2, BOARD_SIZE, DIV_W),
        )
        pygame.draw.rect(
            screen,
            LINE_COLOR,
            pygame.Rect(xx - DIV_W // 2, BOARD_TOP, DIV_W, BOARD_SIZE),
        )

    # overlays for won inner boards
    pad_big = inner_size // 2 - 18
    for r in range(3):
        for c in range(3):
            val = board[r][c].value
            if val == Piece.EMPTY:
                continue
            cx = BOARD_LEFT + c * inner_size + inner_size // 2
            cy = BOARD_TOP + r * inner_size + inner_size // 2
            if val == Piece.X:
                pygame.draw.line(
                    screen,
                    X_COLOR,
                    (cx - pad_big, cy - pad_big),
                    (cx + pad_big, cy + pad_big),
                    width=12,
                )
                pygame.draw.line(
                    screen,
                    X_COLOR,
                    (cx - pad_big, cy + pad_big),
                    (cx + pad_big, cy - pad_big),
                    width=12,
                )
            else:
                pygame.draw.circle(screen, O_COLOR, (cx, cy), pad_big, width=12)

    # labels last
    draw_labels(screen, inner_size)
    return inner_size


def draw_status_bar(
    screen: pygame.Surface,
    p1: Player,
    p2: Player,
    current: Player,
    restriction: Tuple[int, int] | None,
    last_invalid_until: float,
) -> None:
    """Bottom status bar."""
    y0 = SCREEN_SIZE - STATUS_BAR_H
    pygame.draw.rect(screen, BAR_BG, pygame.Rect(0, y0, SCREEN_SIZE, STATUS_BAR_H))

    # player names
    p1_col = X_COLOR if p1.piece == Piece.X else O_COLOR
    p2_col = X_COLOR if p2.piece == Piece.X else O_COLOR
    p1_label = FONT_BOLD.render(p1.get_name(), True, p1_col)
    p2_label = FONT_BOLD.render(p2.get_name(), True, p2_col)

    # current turn
    turn_col = X_COLOR if current.piece == Piece.X else O_COLOR
    turn_label = FONT_BOLD.render(
        f"Turn: {'X' if current.piece == Piece.X else 'O'}", True, turn_col
    )

    # restriction
    rest_text = "Any" if restriction is None else idx_to_label(restriction)
    rest_label = FONT_BOLD.render(f"Restriction: {rest_text}", True, LBL_COLOR)

    pad = 14
    screen.blit(p1_label, (pad, y0 + 12))
    screen.blit(p2_label, (pad, y0 + 12 + p1_label.get_height() + 4))
    screen.blit(turn_label, (SCREEN_SIZE // 3, y0 + 16))
    screen.blit(rest_label, (SCREEN_SIZE // 3, y0 + 16 + turn_label.get_height() + 6))

    # warning
    if time.time() < last_invalid_until:
        warn = FONT_BOLD.render("Invalid move!", True, WARN_COLOR)
        screen.blit(warn, (SCREEN_SIZE - warn.get_width() - pad, y0 + 16))


def game_loop() -> bool:
    """Main game loop."""
    board = get_board()

    p1 = HumanPlayer(Piece.X)
    p2 = MinmaxPlayer(Piece.O, depth_limit=5)

    current = p1
    last_invalid_until = 0.0

    draw_main_board(screen, board, board.restriction)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        screen.fill(BG)

        # handle input/ai
        move = current.get_move(board)
        if move:
            token = board.make_move(move)
            if token is None:
                last_invalid_until = time.time() + 1.5
            else:
                current = p1 if current is p2 else p2

        draw_main_board(screen, board, board.restriction)
        draw_status_bar(screen, p1, p2, current, board.restriction, last_invalid_until)

        # Game finished
        if board.board_state != BoardState.NOT_FINISHED:
            draw_endgame_banner(screen, board.board_state)
            pygame.display.flip()  # <-- show the banner
            pygame.event.pump()  # <-- keep window responsive
            pygame.time.wait(4000)  # <-- pause for 4s without freezing OS events
            return True

        pygame.display.flip()
        clock.tick(60)


def main() -> None:
    set_layout(
        screen_w=SCREEN_SIZE,
        screen_h=SCREEN_SIZE,
        board_size=BOARD_SIZE,
        board_left=BOARD_LEFT,
        board_top=BOARD_TOP,
    )

    play_again = True
    while play_again:
        play_again = game_loop()

    pygame.quit()


if __name__ == "__main__":
    main()
