from __future__ import annotations
import os
import time
import multiprocessing
import queue
from typing import Tuple

import pygame

from board import Board, BoardState, Piece, board_state_to_piece, get_board, Move
from player import HumanPlayer, MinimaxPlayer, Player, set_layout, MonteCarloPlayer, QLearningPlayer, DQNPlayer

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
MC_DEFAULT_ITERS = 10000
MC_DEFAULT_HEURISTICS = False
MINIMAX_DEFAULT_DEPTH = 6
MINIMAX_DEFAULT_HEURISTICS = True
MINIMAX_DEFAULT_PRUNING = True
MODELS_DIR = "models/qlearning"
DQN_MODELS_DIR = "models/dqn"
MAX_GAMES = 9999


def _format_model_label(name: str) -> str:
    """Strip q_table_ / dqn_ prefix and display episode count, e.g. '100k ep'."""
    for prefix in ("q_table_", "dqn_"):
        if name.startswith(prefix):
            ep = name[len(prefix):]
            return f"{ep} ep"
    return name


def _parse_ep_count(name: str) -> int:
    """Extract episode count from a model filename for sorting."""
    for prefix in ("q_table_", "dqn_"):
        if name.startswith(prefix):
            ep = name[len(prefix):]
            break
    else:
        ep = name
    try:
        if ep.endswith("M"):
            return int(float(ep[:-1]) * 1_000_000)
        if ep.endswith("k"):
            return int(float(ep[:-1]) * 1_000)
        return int(ep)
    except ValueError:
        return 0


def _list_models(directory: str = MODELS_DIR, ext: str = ".pkl") -> list[str]:
    """Return model names sorted by episode count ascending."""
    if not os.path.isdir(directory):
        return []
    names = [f[: -len(ext)] for f in os.listdir(directory) if f.endswith(ext)]
    return sorted(names, key=_parse_ep_count)


def _compute_ai_move(
    player: Player, board_snapshot: Board, out_q: "queue.Queue"
) -> None:
    """Worker process: compute AI move on a snapshot and return it via queue."""
    from time import perf_counter

    start = perf_counter()
    move = player.get_move(board_snapshot)
    elapsed = perf_counter() - start
    out_q.put((move, elapsed))


# --- setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
clock = pygame.time.Clock()
pygame.display.set_caption("Ultimate Tic-Tac-Toe")

FONT = pygame.font.SysFont("Georgia", 28)
FONT_BOLD = pygame.font.SysFont("Georgia", 36, bold=True)


def draw_endgame_overlay(
    screen: pygame.Surface,
    state: BoardState,
    score: tuple[int, int, int] | None = None,
    game_num: int = 1,
    total_games: int = 1,
) -> None:
    """Big 'X wins! / O wins! / Draw' centered over the main board."""
    big_font = pygame.font.SysFont(None, 110)
    note_font = pygame.font.SysFont(None, 32)
    score_font = pygame.font.SysFont(None, 42)

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

    below_y = ty + text.get_height() + 10

    # series score line
    if score is not None and total_games > 1:
        x_wins, o_wins, draws = score
        score_msg = f"Series — X: {x_wins}  O: {o_wins}  Draw: {draws}"
        score_surf = score_font.render(score_msg, True, LBL_COLOR)
        screen.blit(score_surf, (cx - score_surf.get_width() // 2, below_y))
        below_y += score_surf.get_height() + 8

    if total_games > 1 and game_num < total_games:
        note = f"(press any key — game {game_num + 1} of {total_games} next)"
    else:
        note = "(press any key)"
    note_text = note_font.render(note, True, LBL_COLOR)
    screen.blit(note_text, (cx - note_text.get_width() // 2, below_y))


def draw_series_result(
    screen: pygame.Surface,
    score: tuple[int, int, int],
    p1_name: str,
    p2_name: str,
) -> None:
    """Full-screen series result overlay after all games are played."""
    title_font = pygame.font.SysFont(None, 80)
    big_font = pygame.font.SysFont(None, 110)
    score_font = pygame.font.SysFont(None, 52)
    note_font = pygame.font.SysFont(None, 32)

    x_wins, o_wins, draws = score

    wash = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
    wash.fill((255, 255, 255, 200))
    screen.blit(wash, (0, 0))

    cx = SCREEN_SIZE // 2
    cy = SCREEN_SIZE // 2

    # title
    title_surf = title_font.render("Series Complete!", True, BORDER)
    screen.blit(title_surf, (cx - title_surf.get_width() // 2, cy - 220))

    # winner
    if x_wins > o_wins:
        winner_msg = f"X wins the series!"
        winner_col = X_COLOR
    elif o_wins > x_wins:
        winner_msg = f"O wins the series!"
        winner_col = O_COLOR
    else:
        winner_msg = "It's a tie!"
        winner_col = LBL_COLOR

    win_surf = big_font.render(winner_msg, True, winner_col)
    screen.blit(win_surf, (cx - win_surf.get_width() // 2, cy - 130))

    # score breakdown
    breakdown = f"X: {x_wins}   O: {o_wins}   Draw: {draws}"
    score_surf = score_font.render(breakdown, True, LBL_COLOR)
    screen.blit(score_surf, (cx - score_surf.get_width() // 2, cy + 30))

    # score bar visualisation
    total = x_wins + o_wins + draws
    if total > 0:
        bar_w = 600
        bar_h = 28
        bar_x = cx - bar_w // 2
        bar_y = cy + 100
        # background
        pygame.draw.rect(screen, BAR_BG, pygame.Rect(bar_x, bar_y, bar_w, bar_h), border_radius=6)
        drawn = 0
        for wins, color in ((x_wins, X_COLOR), (o_wins, O_COLOR), (draws, LBL_COLOR)):
            seg_w = int(bar_w * wins / total)
            if seg_w > 0:
                pygame.draw.rect(screen, color, pygame.Rect(bar_x + drawn, bar_y, seg_w, bar_h))
                drawn += seg_w
        pygame.draw.rect(screen, BORDER, pygame.Rect(bar_x, bar_y, bar_w, bar_h), width=2, border_radius=6)

    note_surf = note_font.render("(press any key to return to menu)", True, LBL_COLOR)
    screen.blit(note_surf, (cx - note_surf.get_width() // 2, cy + 160))


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
    score: tuple[int, int, int] | None = None,
    game_num: int = 1,
    total_games: int = 1,
) -> None:
    """Bottom status bar."""
    score_font = pygame.font.SysFont("Georgia", 22, bold=True)

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

    # series score (right side)
    if score is not None and total_games > 1:
        x_wins, o_wins, draws = score
        game_line = score_font.render(f"Game {game_num} / {total_games}", True, LBL_COLOR)
        score_line = score_font.render(
            f"X: {x_wins}   O: {o_wins}   Draw: {draws}", True, LBL_COLOR
        )
        right_x = SCREEN_SIZE - max(game_line.get_width(), score_line.get_width()) - pad
        screen.blit(game_line, (right_x, y0 + 12))
        screen.blit(score_line, (right_x, y0 + 12 + game_line.get_height() + 4))
    else:
        # warning / thinking on right side
        if time.time() < last_invalid_until:
            warn = FONT_BOLD.render("Invalid move!", True, WARN_COLOR)
            screen.blit(warn, (SCREEN_SIZE - warn.get_width() - pad, y0 + 16))

        if thinking:
            dots = "." * (int(time.time() * 3) % 4) + " " * (3 - (int(time.time() * 3) % 4))
            msg = FONT_BOLD.render(f"Thinking{dots}", True, LBL_COLOR)
            screen.blit(msg, (SCREEN_SIZE - msg.get_width() - pad, y0 + 16))

    # warning / thinking always visible even in series mode, but below score
    if total_games > 1:
        if time.time() < last_invalid_until:
            warn = FONT_BOLD.render("Invalid move!", True, WARN_COLOR)
            screen.blit(warn, (SCREEN_SIZE // 2 - warn.get_width() // 2, y0 + 16))
        if thinking:
            dots = "." * (int(time.time() * 3) % 4) + " " * (3 - (int(time.time() * 3) % 4))
            msg = FONT_BOLD.render(f"Thinking{dots}", True, LBL_COLOR)
            screen.blit(msg, (SCREEN_SIZE // 2 - msg.get_width() // 2, y0 + 16))


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
    if choice == "qlearning":
        model_name = params.get("model")
        if model_name:
            path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
            if os.path.exists(path):
                p = QLearningPlayer.load(path, piece=piece, epsilon=0.0)
                p.name = f"Q-Learning ({_format_model_label(model_name)})"
                return p
        return QLearningPlayer(piece=piece, epsilon=0.0)
    if choice == "dqn":
        model_name = params.get("model")
        if model_name:
            path = os.path.join(DQN_MODELS_DIR, f"{model_name}.pt")
            if os.path.exists(path):
                p = DQNPlayer.load(path, piece=piece, epsilon=0.0)
                p.name = f"DQN ({_format_model_label(model_name)})"
                return p
        return DQNPlayer(piece=piece, epsilon=0.0)
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


def menu() -> tuple[tuple[str, dict], tuple[str, dict], int, bool] | None:
    ql_models = _list_models(MODELS_DIR, ".pkl")
    dqn_models = _list_models(DQN_MODELS_DIR, ".pt")
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
        {
            "label": "Q-Learning",
            "key": "qlearning",
            "params": {"model": ql_models[-1] if ql_models else None},
        },
        {
            "label": "DQN",
            "key": "dqn",
            "params": {"model": dqn_models[-1] if dqn_models else None},
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
        "qlearning": (
            [{"name": "model", "label": "Model", "type": "cycle", "choices": ql_models}]
            if ql_models
            else []
        ),
        "dqn": (
            [{"name": "model", "label": "Model", "type": "cycle", "choices": dqn_models}]
            if dqn_models
            else []
        ),
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
    num_games = 1
    auto_skip = False

    title_font = pygame.font.SysFont("Georgia", 44, bold=True)
    sub_font = pygame.font.SysFont("Georgia", 26, bold=True)
    tiny_font = pygame.font.SysFont("Georgia", 22, bold=False)
    param_font = pygame.font.SysFont("Georgia", 24, bold=False)
    ng_font = pygame.font.SysFont("Georgia", 26, bold=True)

    col_w = 340
    gap = 80
    top = 200
    opt_h = 56
    opt_gap = 16

    start_rect = pygame.Rect(SCREEN_SIZE // 2 - 140, SCREEN_SIZE - 114, 280, 64)

    # num_games + auto_skip control geometry (centered, above start button)
    ng_btn_size = 40
    ng_panel_y = SCREEN_SIZE - 272  # top of the panel
    # buttons sit on the second row of the panel (offset 38 from panel top)
    # layout: [--] [-]  number  [+] [++]
    _cx = SCREEN_SIZE // 2
    ng_dec_rect  = pygame.Rect(_cx - 88,  ng_panel_y + 38, ng_btn_size, ng_btn_size)
    ng_dec5_rect = pygame.Rect(_cx - 138, ng_panel_y + 38, ng_btn_size, ng_btn_size)
    ng_inc_rect  = pygame.Rect(_cx + 48,  ng_panel_y + 38, ng_btn_size, ng_btn_size)
    ng_inc5_rect = pygame.Rect(_cx + 98,  ng_panel_y + 38, ng_btn_size, ng_btn_size)
    # auto-skip toggle checkbox (third row, offset 90 from panel top)
    as_toggle_rect = pygame.Rect(SCREEN_SIZE // 2 - 110, ng_panel_y + 90, 24, 24)

    def _get_choices() -> tuple[str, dict, str, dict]:
        left_choice = options[selected[0]]["key"]
        right_choice = options[selected[1]]["key"]
        return left_choice, params[0][left_choice], right_choice, params[1][right_choice]

    def _return_result():
        lc, lp, rc, rp = _get_choices()
        return (lc, dict(lp)), (rc, dict(rp)), num_games, auto_skip

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
                    return _return_result()
                # num_games: - and = (plus)
                if event.key == pygame.K_MINUS:
                    num_games = max(1, num_games - 1)
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    num_games = min(MAX_GAMES, num_games + 1)
                # auto-skip: Tab
                if event.key == pygame.K_TAB:
                    auto_skip = not auto_skip

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
                if ng_dec5_rect.collidepoint(mx, my):
                    num_games = max(1, num_games - 5)
                if ng_dec_rect.collidepoint(mx, my):
                    num_games = max(1, num_games - 1)
                if ng_inc_rect.collidepoint(mx, my):
                    num_games = min(MAX_GAMES, num_games + 1)
                if ng_inc5_rect.collidepoint(mx, my):
                    num_games = min(MAX_GAMES, num_games + 5)
                if as_toggle_rect.collidepoint(mx, my):
                    auto_skip = not auto_skip
                if start_rect.collidepoint(mx, my):
                    return _return_result()

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
                        if spec["type"] == "cycle" and spec_idx == 0:
                            choices = spec["choices"]
                            if choices:
                                cur = p[spec["name"]]
                                idx = choices.index(cur) if cur in choices else 0
                                if event.key == player_keys["dec"]:
                                    p[spec["name"]] = choices[(idx - 1) % len(choices)]
                                if event.key == player_keys["inc"]:
                                    p[spec["name"]] = choices[(idx + 1) % len(choices)]
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
                elif spec["type"] == "cycle":
                    raw = p[name]
                    current = _format_model_label(raw) if raw else "(none)"
                    choices = spec["choices"]
                    n = len(choices)
                    text = (
                        f"{label}: {current}  ({n} model{'s' if n != 1 else ''})  "
                        f"({pygame.key.name(keys['dec'])}/{pygame.key.name(keys['inc'])})"
                    )
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

        # --- num games + auto-skip control panel ---
        panel_w = 420
        panel_h = 142
        panel_rect = pygame.Rect(SCREEN_SIZE // 2 - panel_w // 2, ng_panel_y - 4, panel_w, panel_h)
        pygame.draw.rect(screen, BAR_BG, panel_rect, border_radius=12)
        pygame.draw.rect(screen, BORDER, panel_rect, width=1, border_radius=12)

        # Row 1: "Number of Games" title
        ng_title = tiny_font.render("Number of Games", True, LBL_COLOR)
        screen.blit(ng_title, (SCREEN_SIZE // 2 - ng_title.get_width() // 2, ng_panel_y + 6))

        # Row 2: [−]  value  [+]
        num_surf = ng_font.render(str(num_games), True, BORDER)
        screen.blit(num_surf, (SCREEN_SIZE // 2 - num_surf.get_width() // 2, ng_panel_y + 38))

        _btn_small_font = pygame.font.SysFont("Georgia", 18, bold=True)
        for btn_rect, symbol, small in (
            (ng_dec5_rect, "−−", True),
            (ng_dec_rect,  "−",  False),
            (ng_inc_rect,  "+",  False),
            (ng_inc5_rect, "++", True),
        ):
            pygame.draw.rect(screen, BOARD_BG, btn_rect, border_radius=8)
            pygame.draw.rect(screen, BORDER, btn_rect, width=1, border_radius=8)
            f = _btn_small_font if small else ng_font
            sym = f.render(symbol, True, LBL_COLOR)
            screen.blit(sym, (btn_rect.centerx - sym.get_width() // 2, btn_rect.centery - sym.get_height() // 2))

        # Divider
        div_y = ng_panel_y + 82
        pygame.draw.line(screen, LINE_COLOR,
                         (panel_rect.x + 16, div_y), (panel_rect.right - 16, div_y), 1)

        # Row 3: auto-skip checkbox + label
        cb = as_toggle_rect
        pygame.draw.rect(screen, BOARD_BG, cb, border_radius=4)
        pygame.draw.rect(screen, BORDER, cb, width=1, border_radius=4)
        if auto_skip:
            pad = 5
            pygame.draw.line(screen, X_COLOR, (cb.x + pad, cb.centery), (cb.centerx - 1, cb.bottom - pad), 3)
            pygame.draw.line(screen, X_COLOR, (cb.centerx - 1, cb.bottom - pad), (cb.right - pad, cb.y + pad), 3)
        as_lbl = tiny_font.render("Auto-advance games", True, LBL_COLOR)
        screen.blit(as_lbl, (cb.right + 10, cb.centery - as_lbl.get_height() // 2))
        as_sub = pygame.font.SysFont("Georgia", 17).render(
            "1s pause between games, no keypress needed", True, LBL_COLOR
        )
        screen.blit(as_sub, (panel_rect.centerx - as_sub.get_width() // 2, cb.bottom + 5))

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


def game_loop(
    p1_choice: str,
    p1_params: dict,
    p2_choice: str,
    p2_params: dict,
    game_num: int = 1,
    total_games: int = 1,
    score: tuple[int, int, int] = (0, 0, 0),
    auto_skip: bool = False,
) -> BoardState | None:
    """Run a single game. Returns the terminal BoardState, or None if the user quit."""
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
    draw_status_bar(
        screen, p1, p2, current, board.restriction, last_invalid_until,
        score=score, game_num=game_num, total_games=total_games,
    )
    pygame.display.flip()

    # --- now enter loop ---
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if ai_process is not None and ai_process.is_alive():
                    ai_process.terminate()
                    ai_process.join()
                return None

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
                    pending_move, elapsed = ai_result.get_nowait()
                    ai_process.join()
                    ai_process = None
                    current.record_move_time(elapsed)
                except queue.Empty:
                    if ai_process is not None and not ai_process.is_alive():
                        ai_process.join()
                        ai_process = None
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
            score=score,
            game_num=game_num,
            total_games=total_games,
        )

        if board.board_state != BoardState.NOT_FINISHED:
            draw_endgame_overlay(
                screen, board.board_state,
                score=score, game_num=game_num, total_games=total_games,
            )
            pygame.display.flip()
            if auto_skip:
                deadline = time.time() + 1.0
                while time.time() < deadline:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return None
                    clock.tick(60)
            else:
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return None
                        if event.type == pygame.KEYDOWN:
                            waiting = False
                            break
                    clock.tick(60)
            return board.board_state

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

    while True:
        choices = menu()
        if choices is None:
            break
        (p1_choice, p1_params), (p2_choice, p2_params), num_games, auto_skip = choices

        score = [0, 0, 0]  # x_wins, o_wins, draws
        p1_name = _make_player(p1_choice, Piece.X, p1_params).get_name()
        p2_name = _make_player(p2_choice, Piece.O, p2_params).get_name()

        for game_num in range(1, num_games + 1):
            result = game_loop(
                p1_choice, p1_params, p2_choice, p2_params,
                game_num=game_num,
                total_games=num_games,
                score=tuple(score),
                auto_skip=auto_skip,
            )
            if result is None:
                # user quit mid-series
                pygame.quit()
                return
            if result == BoardState.X_WON:
                score[0] += 1
            elif result == BoardState.O_WON:
                score[1] += 1
            else:
                score[2] += 1

        if num_games > 1:
            screen.fill(BG)
            draw_series_result(screen, tuple(score), p1_name, p2_name)
            pygame.display.flip()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        waiting = False
                        break
                clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
