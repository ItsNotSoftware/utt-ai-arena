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

# --- menu defaults ---
MC_DEFAULT_ITERS = 1000
MC_DEFAULT_HEURISTICS = True
MINIMAX_DEFAULT_DEPTH = 6
MINIMAX_DEFAULT_HEURISTICS = True
MINIMAX_DEFAULT_PRUNING = True


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


def _make_player(choice: str, piece: Piece, params: dict | None = None) -> Player:
    params = params or {}
    if choice == "human":
        return HumanPlayer(piece)
    if choice == "minimax":
        return MinimaxPlayer(
            piece,
            depth_limit=int(params.get("depth", MINIMAX_DEFAULT_DEPTH)),
            use_heuristic_eval=bool(
                params.get("heuristics", MINIMAX_DEFAULT_HEURISTICS)
            ),
            use_pruning=bool(params.get("pruning", MINIMAX_DEFAULT_PRUNING)),
        )
    if choice == "mcts":
        return MonteCarloPlayer(
            piece=piece,
            iter_nr=int(params.get("iters", MC_DEFAULT_ITERS)),
            use_heuristics=MC_DEFAULT_HEURISTICS,
        )
    raise ValueError(f"Unknown player choice: {choice}")


def _draw_menu_option(
    screen: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    selected: bool,
) -> None:
    if selected:
        pygame.draw.rect(screen, LINE_COLOR, rect, border_radius=10)
        inner = rect.inflate(-6, -6)
        pygame.draw.rect(screen, BOARD_BG, inner, border_radius=8)
    else:
        pygame.draw.rect(screen, BAR_BG, rect, border_radius=10)
        pygame.draw.rect(screen, BORDER, rect, width=1, border_radius=10)

    text = FONT.render(label, True, LBL_COLOR)
    screen.blit(
        text,
        (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2),
    )


def menu() -> tuple[tuple[str, dict], tuple[str, dict]] | None:
    options = [
        {"label": "Human", "key": "human", "params": {}},
        {
            "label": "Minimax",
            "key": "minimax",
            "params": {
                "depth": MINIMAX_DEFAULT_DEPTH,
                "heuristics": MINIMAX_DEFAULT_HEURISTICS,
                "pruning": MINIMAX_DEFAULT_PRUNING,
            },
        },
        {
            "label": "MonteCarlo",
            "key": "mcts",
            "params": {"iters": MC_DEFAULT_ITERS},
        },
    ]

    param_specs = {
        "minimax": [
            {"name": "depth", "label": "Depth*", "type": "int", "step": 1, "min": 0},
            {"name": "heuristics", "label": "Heuristic", "type": "bool"},
            {"name": "pruning", "label": "Pruning*", "type": "bool"},
        ],
        "mcts": [
            {
                "name": "iters",
                "label": "Nr of sims*",
                "type": "int",
                "step": 100,
                "min": 100,
            }
        ],
    }

    keymap = {
        0: {
            "dec": pygame.K_q,
            "inc": pygame.K_w,
            "toggle1": pygame.K_e,
            "toggle2": pygame.K_r,
        },
        1: {
            "dec": pygame.K_u,
            "inc": pygame.K_i,
            "toggle1": pygame.K_o,
            "toggle2": pygame.K_p,
        },
    }

    selected = [0, 0]  # p1, p2
    params = [
        {opt["key"]: dict(opt["params"]) for opt in options},
        {opt["key"]: dict(opt["params"]) for opt in options},
    ]

    title_font = pygame.font.SysFont("Georgia", 44, bold=True)
    sub_font = pygame.font.SysFont("Georgia", 26, bold=True)
    tiny_font = pygame.font.SysFont("Georgia", 22, bold=False)
    param_font = pygame.font.SysFont("Georgia", 24, bold=False)

    col_w = 340
    gap = 80
    top = 200
    opt_h = 56
    opt_gap = 16

    start_rect = pygame.Rect(SCREEN_SIZE // 2 - 140, SCREEN_SIZE - 160, 280, 64)

    while True:
        left_x = SCREEN_SIZE // 2 - col_w - gap // 2
        right_x = SCREEN_SIZE // 2 + gap // 2
        params_y = top + len(options) * (opt_h + opt_gap) + 16

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_RETURN:
                    left_choice = options[selected[0]][1]
                    right_choice = options[selected[1]][1]
                    left_params = params[0][left_choice]
                    right_params = params[1][right_choice]
                    return (left_choice, dict(left_params)), (
                        right_choice,
                        dict(right_params),
                    )
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # left column
                for idx in range(len(options)):
                    y = top + idx * (opt_h + opt_gap)
                    left_rect = pygame.Rect(left_x, y, col_w, opt_h)
                    right_rect = pygame.Rect(right_x, y, col_w, opt_h)
                    if left_rect.collidepoint(mx, my):
                        selected[0] = idx
                    if right_rect.collidepoint(mx, my):
                        selected[1] = idx
                if start_rect.collidepoint(mx, my):
                    left_choice = options[selected[0]][1]
                    right_choice = options[selected[1]][1]
                    left_params = params[0][left_choice]
                    right_params = params[1][right_choice]
                    return (left_choice, dict(left_params)), (
                        right_choice,
                        dict(right_params),
                    )
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_a, pygame.K_d):
                    selected[0] = (selected[0] - 1) % len(options)
                if event.key in (pygame.K_s, pygame.K_f):
                    selected[0] = (selected[0] + 1) % len(options)
                if event.key in (pygame.K_j, pygame.K_l):
                    selected[1] = (selected[1] - 1) % len(options)
                if event.key in (pygame.K_k, pygame.K_SEMICOLON):
                    selected[1] = (selected[1] + 1) % len(options)

                def _apply_param_keys(player_idx: int) -> None:
                    choice = options[selected[player_idx]]["key"]
                    specs = param_specs.get(choice, [])
                    player_keys = keymap[player_idx]
                    p = params[player_idx][choice]
                    for spec_idx, spec in enumerate(specs):
                        if spec["type"] == "int" and spec_idx == 0:
                            if event.key == player_keys["dec"]:
                                step = spec.get("step", 1)
                                min_val = spec.get("min", 0)
                                p[spec["name"]] = max(min_val, p[spec["name"]] - step)
                            if event.key == player_keys["inc"]:
                                step = spec.get("step", 1)
                                p[spec["name"]] = p[spec["name"]] + step
                        if spec["type"] == "bool" and spec_idx == 1:
                            if event.key == player_keys["toggle1"]:
                                p[spec["name"]] = not p[spec["name"]]
                        if spec["type"] == "bool" and spec_idx == 2:
                            if event.key == player_keys["toggle2"]:
                                p[spec["name"]] = not p[spec["name"]]

                _apply_param_keys(0)
                _apply_param_keys(1)

        screen.fill(BG)
        title = title_font.render("Select Players", True, BORDER)
        screen.blit(
            title,
            (SCREEN_SIZE // 2 - title.get_width() // 2, 32),
        )

        p1_label = sub_font.render("Player X", True, X_COLOR)
        p2_label = sub_font.render("Player O", True, O_COLOR)
        screen.blit(p1_label, (left_x, top - 40))
        screen.blit(p2_label, (right_x, top - 40))

        for idx, opt in enumerate(options):
            label = opt["label"]
            y = top + idx * (opt_h + opt_gap)
            left_rect = pygame.Rect(left_x, y, col_w, opt_h)
            right_rect = pygame.Rect(right_x, y, col_w, opt_h)
            _draw_menu_option(screen, left_rect, label, selected[0] == idx)
            _draw_menu_option(screen, right_rect, label, selected[1] == idx)

        # params panes
        left_choice = options[selected[0]]["key"]
        right_choice = options[selected[1]]["key"]

        def _render_params(
            x: int,
            choice: str,
            p: dict,
            keys: dict[str, int],
        ) -> int:
            row_h = 26
            y = params_y
            specs = param_specs.get(choice, [])
            if not specs:
                label = param_font.render("No parameters", True, LBL_COLOR)
                screen.blit(label, (x, y))
                return y + row_h
            for idx, spec in enumerate(specs):
                name = spec["name"]
                label = spec["label"]
                if spec["type"] == "int":
                    text = (
                        f"{label}: {p[name]}  "
                        f"({pygame.key.name(keys['dec'])}/{pygame.key.name(keys['inc'])})"
                    )
                    line = param_font.render(text, True, LBL_COLOR)
                    screen.blit(line, (x, y))
                    y += row_h
                    continue
                else:
                    toggle_key = keys["toggle1"] if idx == 1 else keys["toggle2"]
                    text = (
                        f"{label}: {'on' if p[name] else 'off'}  "
                        f"({pygame.key.name(toggle_key)})"
                    )
                    line = param_font.render(text, True, LBL_COLOR)
                    screen.blit(line, (x, y))
                    y += row_h
            return y

        left_params = params[0][left_choice]
        right_params = params[1][right_choice]
        y_after_left = _render_params(left_x, left_choice, left_params, keymap[0])
        y_after_right = _render_params(right_x, right_choice, right_params, keymap[1])

        note_y = max(y_after_left, y_after_right) + 10
        note = "* affects compute time"
        note_text = tiny_font.render(note, True, LBL_COLOR)
        screen.blit(
            note_text,
            (SCREEN_SIZE // 2 - note_text.get_width() // 2, note_y),
        )

        # start button
        pygame.draw.rect(screen, LINE_COLOR, start_rect, border_radius=12)
        inner = start_rect.inflate(-6, -6)
        pygame.draw.rect(screen, BOARD_BG, inner, border_radius=10)
        start_text = FONT_BOLD.render("Start Game", True, BORDER)
        screen.blit(
            start_text,
            (
                start_rect.centerx - start_text.get_width() // 2,
                start_rect.centery - start_text.get_height() // 2,
            ),
        )

        pygame.display.flip()
        clock.tick(60)


def game_loop(p1_choice: str, p1_params: dict, p2_choice: str, p2_params: dict) -> bool:
    board = get_board()

    p1 = _make_player(p1_choice, Piece.X, p1_params)
    p2 = _make_player(p2_choice, Piece.O, p2_params)

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
        choices = menu()
        if choices is None:
            break
        (p1_choice, p1_params), (p2_choice, p2_params) = choices
        play_again = game_loop(p1_choice, p1_params, p2_choice, p2_params)

    pygame.quit()


if __name__ == "__main__":
    main()
