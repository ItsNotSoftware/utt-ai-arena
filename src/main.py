from __future__ import annotations
import time
import multiprocessing
import queue
from typing import Tuple

import pygame

from board import Board, BoardState, Piece, board_state_to_piece, get_board, Move
from player import HumanPlayer, MinimaxPlayer, Player, set_layout, MonteCarloPlayer

# --- constants ---
SCREEN_SIZE = 1280
STATUS_BAR_H = 110
HEADER_H = 28  # small top header for column numbers
SIDEBAR_W = 28  # small left sidebar for row letters
BOARD_SIZE = SCREEN_SIZE - STATUS_BAR_H - HEADER_H
BOARD_LEFT = (SCREEN_SIZE - BOARD_SIZE) // 2
BOARD_TOP = HEADER_H  # push board down to make room for header

# --- colors ---
BG = pygame.Color(246, 247, 252)
BOARD_BG = pygame.Color(252, 253, 255)
BAR_BG = pygame.Color(236, 239, 246)
BORDER = pygame.Color(24, 24, 28)
LINE_COLOR = pygame.Color(60, 63, 72)
X_COLOR = pygame.Color(200, 40, 35)
O_COLOR = pygame.Color(35, 90, 200)
LBL_COLOR = pygame.Color(82, 88, 105)
WARN_COLOR = pygame.Color(200, 45, 45)

# --- grid thickness ---
GRID_THIN = 2
DIV_W = 6
INNER_MARGIN = 16  # margin inside each inner board

# --- fonts ---
FONT = None
FONT_BOLD = None


def _compute_ai_move(
    player: Player, board_snapshot: Board, out_q: "queue.Queue"
) -> None:
    """Worker process: compute AI move on a snapshot and return it via queue."""
    out_q.put(player.get_move(board_snapshot))


# --- setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
clock = pygame.time.Clock()
pygame.display.set_caption("Ultimate Tic-Tac-Toe")

FONT = pygame.font.SysFont("Georgia", 28)
FONT_BOLD = pygame.font.SysFont("Georgia", 36, bold=True)


def draw_endgame_overlay(screen: pygame.Surface, state: BoardState) -> None:
    """Big 'X wins! / O wins! / Draw' centered over the main board."""
    big_font = pygame.font.SysFont(None, 110)

    if state == BoardState.X_WON:
        msg, col = "X wins!", X_COLOR
    elif state == BoardState.O_WON:
        msg, col = "O wins!", O_COLOR
    else:
        msg, col = "Draw", LBL_COLOR

    text = big_font.render(msg, True, col)
    outline = big_font.render(msg, True, BORDER)

    # center inside board
    cx = BOARD_LEFT + BOARD_SIZE // 2
    cy = BOARD_TOP + BOARD_SIZE // 2
    tx = cx - text.get_width() // 2
    ty = cy - text.get_height() // 2

    # optional subtle translucent wash for readability
    wash = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
    wash.fill((255, 255, 255, 90))
    screen.blit(wash, (BOARD_LEFT, BOARD_TOP))

    screen.blit(outline, (tx + 2, ty + 2))
    screen.blit(text, (tx, ty))


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

            surface = render_inner_board(board[r][c], inner_size)
            screen.blit(surface, (x, y))

            if restriction is not None and restriction != (r, c):
                dim = pygame.Surface((inner_size, inner_size), pygame.SRCALPHA)
                dim.fill((20, 20, 20, 18))
                screen.blit(dim, (x, y))
            elif restriction == (r, c):
                pass

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
            val = board_state_to_piece(board[r][c].board_state)
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
    thinking: bool = False,
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
    left_w = max(p1_label.get_width(), p2_label.get_width())
    mid_x = pad + left_w + 36

    screen.blit(p1_label, (pad, y0 + 12))
    screen.blit(p2_label, (pad, y0 + 12 + p1_label.get_height() + 4))
    screen.blit(turn_label, (mid_x, y0 + 16))
    screen.blit(rest_label, (mid_x, y0 + 16 + turn_label.get_height() + 6))

    # warning
    if time.time() < last_invalid_until:
        warn = FONT_BOLD.render("Invalid move!", True, WARN_COLOR)
        screen.blit(warn, (SCREEN_SIZE - warn.get_width() - pad, y0 + 16))

    if thinking:
        dots = "." * (int(time.time() * 3) % 4) + " " * (3 - (int(time.time() * 3) % 4))
        msg = FONT_BOLD.render(f"Thinking{dots}", True, LBL_COLOR)
        screen.blit(msg, (SCREEN_SIZE - msg.get_width() - pad, y0 + 16))


def game_loop() -> bool:
    board = get_board()

    p1 = MinimaxPlayer(Piece.X)
    p2 = MonteCarloPlayer(piece=Piece.O, iter_nr=1000, use_heuristics=True)

    current = p1 if time.time() % 2 < 1 else p2
    last_invalid_until = 0.0
    pending_move = None
    ai_process: multiprocessing.Process | None = None
    mp_ctx = (
        multiprocessing.get_context("fork")
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing.get_context()
    )
    ai_result: "queue.Queue[Move | None]" = mp_ctx.Queue()

    # --- draw and flip once before loop ---
    screen.fill(BG)
    draw_main_board(screen, board, board.restriction)
    draw_status_bar(screen, p1, p2, current, board.restriction, last_invalid_until)
    pygame.display.flip()

    # --- now enter loop ---
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if ai_process is not None and ai_process.is_alive():
                    ai_process.terminate()
                    ai_process.join()
                return False

        # handle input/ai
        if pending_move is None:
            if isinstance(current, HumanPlayer):
                pending_move = current.get_move(board)
            else:
                if ai_process is None:
                    board_snapshot = board.clone()
                    ai_process = mp_ctx.Process(
                        target=_compute_ai_move,
                        args=(current, board_snapshot, ai_result),
                        daemon=True,
                    )
                    ai_process.start()
                try:
                    pending_move = ai_result.get_nowait()
                    ai_process.join()
                    ai_process = None
                except queue.Empty:
                    if ai_process is not None and not ai_process.is_alive():
                        ai_process.join()
                        ai_process = None
                    pass
        move = pending_move
        if move:
            token = board.make_move(move)
            if token is None:
                last_invalid_until = time.time() + 1.5
            else:
                current = p1 if current is p2 else p2
            pending_move = None

        screen.fill(BG)
        draw_main_board(screen, board, board.restriction)
        draw_status_bar(
            screen,
            p1,
            p2,
            current,
            board.restriction,
            last_invalid_until,
            thinking=not isinstance(current, HumanPlayer) and pending_move is None,
        )

        if board.board_state != BoardState.NOT_FINISHED:
            draw_endgame_overlay(screen, board.board_state)
            pygame.display.flip()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type in (pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN):
                        waiting = False
                        break
                clock.tick(60)
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
