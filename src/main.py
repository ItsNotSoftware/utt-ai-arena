from __future__ import annotations
import os
import time
import multiprocessing
import queue
from typing import Tuple

if not os.environ.get("SDL_VIDEODRIVER") and os.environ.get("WAYLAND_DISPLAY"):
    os.environ["SDL_VIDEODRIVER"] = "wayland"

import pygame

from board import Board, BoardState, Piece, board_state_to_piece, get_board, Move
from player import (
    HumanPlayer,
    MinimaxPlayer,
    Player,
    set_layout,
    MonteCarloPlayer,
    QLearningPlayer,
    DQNPlayer,
    AlphaZeroPlayer,
)

# --- constants / layout (recalculated on resize) ---
MIN_SIZE = 800
STATUS_BAR_H = 100
HEADER_H = 28
SIDEBAR_W = 28

# mutable layout globals — updated by _recalc_layout()
SCREEN_W = 1280
SCREEN_H = 1280
BOARD_SIZE = SCREEN_W - STATUS_BAR_H - HEADER_H
BOARD_LEFT = (SCREEN_W - BOARD_SIZE) // 2
BOARD_TOP = HEADER_H


def _recalc_layout(w: int, h: int) -> None:
    global SCREEN_W, SCREEN_H, BOARD_SIZE, BOARD_LEFT, BOARD_TOP
    SCREEN_W = max(w, MIN_SIZE)
    SCREEN_H = max(h, MIN_SIZE)
    BOARD_SIZE = min(SCREEN_W, SCREEN_H) - STATUS_BAR_H - HEADER_H
    BOARD_LEFT = (SCREEN_W - BOARD_SIZE) // 2
    BOARD_TOP = HEADER_H


# --- colors ---
BG = pygame.Color(30, 30, 38)
BOARD_BG = pygame.Color(40, 42, 54)
BAR_BG = pygame.Color(38, 40, 50)
BORDER = pygame.Color(62, 65, 80)
LINE_COLOR = pygame.Color(72, 76, 92)
X_COLOR = pygame.Color(255, 85, 85)
O_COLOR = pygame.Color(80, 160, 255)
LBL_COLOR = pygame.Color(160, 165, 185)
WARN_COLOR = pygame.Color(255, 70, 70)
TEXT_COLOR = pygame.Color(230, 232, 240)
ACCENT = pygame.Color(100, 110, 200)
ACCENT_HOVER = pygame.Color(120, 130, 220)
CARD_BG = pygame.Color(46, 48, 62)
CARD_BORDER = pygame.Color(70, 74, 95)
BTN_BG = pygame.Color(55, 58, 75)
BTN_HOVER = pygame.Color(70, 74, 95)
GREEN = pygame.Color(80, 200, 120)
DIM_OVERLAY = pygame.Color(20, 20, 30, 30)
HIGHLIGHT = pygame.Color(100, 110, 200, 25)

# --- grid thickness ---
GRID_THIN = 2
DIV_W = 6
INNER_MARGIN = 16

# --- fonts ---
FONT = None
FONT_BOLD = None

# --- menu defaults ---
MC_DEFAULT_ITERS = 10000
MC_DEFAULT_HEURISTICS = False
MINIMAX_DEFAULT_DEPTH = 7
MINIMAX_DEFAULT_HEURISTICS = True
MINIMAX_DEFAULT_PRUNING = True
MODELS_DIR = "models/qlearning"
DQN_MODELS_DIR = "models/dqn"
AZ_MODELS_DIR = "models/alphazero"
AZ_DEFAULT_SIMS = 1500
MAX_GAMES = 9999


def _format_model_label(name: str) -> str:
    # Strip the type prefix; the rest is already self-describing (e.g. "50k",
    # "3000it", "turbo") and the card header tells the user which algorithm
    # this is, so no extra unit suffix is needed.
    for prefix in ("q_table_", "dqn_", "az_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _parse_ep_count(name: str) -> int:
    for prefix in ("q_table_", "dqn_", "az_"):
        if name.startswith(prefix):
            ep = name[len(prefix) :]
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
    if not os.path.isdir(directory):
        return []
    checkpoint_suffix = f".ckpt{ext}"
    names = [
        f[: -len(ext)]
        for f in os.listdir(directory)
        if f.endswith(ext) and not f.endswith(checkpoint_suffix)
    ]
    return sorted(names, key=_parse_ep_count)


def _compute_ai_move(
    player: Player, board_snapshot: Board, out_q: "queue.Queue"
) -> None:
    from time import perf_counter
    import os
    import random

    # Forked subprocesses inherit the parent's PRNG state, so without reseeding
    # every AI turn would replay the same "random" choices (identical games).
    seed = int.from_bytes(os.urandom(8), "big")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed & 0xFFFFFFFF)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

    start = perf_counter()
    move = player.get_move(board_snapshot)
    elapsed = perf_counter() - start
    out_q.put((move, elapsed))


# --- setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.RESIZABLE)
clock = pygame.time.Clock()
pygame.display.set_caption("Ultimate Tic-Tac-Toe Arena")

FONT = pygame.font.SysFont("Segoe UI", 26)
FONT_BOLD = pygame.font.SysFont("Segoe UI", 30, bold=True)


def _handle_resize(event) -> None:
    """Handle a VIDEORESIZE event — update layout and surface."""
    global screen
    _recalc_layout(event.w, event.h)
    screen = pygame.display.get_surface()


def _draw_rounded_rect(surface, color, rect, radius=10, width=0):
    pygame.draw.rect(surface, color, rect, width=width, border_radius=radius)


def draw_endgame_overlay(
    screen: pygame.Surface,
    state: BoardState,
    score: tuple[int, int, int] | None = None,
    game_num: int = 1,
    total_games: int = 1,
) -> None:
    big_font = pygame.font.SysFont("Segoe UI", 100, bold=True)
    note_font = pygame.font.SysFont("Segoe UI", 26)
    score_font = pygame.font.SysFont("Segoe UI", 36, bold=True)

    if state == BoardState.X_WON:
        msg, col = "X wins!", X_COLOR
    elif state == BoardState.O_WON:
        msg, col = "O wins!", O_COLOR
    else:
        msg, col = "Draw", LBL_COLOR

    text = big_font.render(msg, True, col)

    cx = BOARD_LEFT + BOARD_SIZE // 2
    cy = BOARD_TOP + BOARD_SIZE // 2

    # dark wash
    wash = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
    wash.fill((15, 15, 20, 160))
    screen.blit(wash, (BOARD_LEFT, BOARD_TOP))

    # result card
    card_w, card_h = 500, 260
    card_rect = pygame.Rect(cx - card_w // 2, cy - card_h // 2, card_w, card_h)
    _draw_rounded_rect(screen, CARD_BG, card_rect, 16)
    _draw_rounded_rect(screen, CARD_BORDER, card_rect, 16, 2)

    tx = cx - text.get_width() // 2
    ty = card_rect.y + 30
    screen.blit(text, (tx, ty))

    below_y = ty + text.get_height() + 10

    if score is not None and total_games > 1:
        x_wins, o_wins, draws = score
        score_msg = f"X: {x_wins}   O: {o_wins}   Draw: {draws}"
        score_surf = score_font.render(score_msg, True, LBL_COLOR)
        screen.blit(score_surf, (cx - score_surf.get_width() // 2, below_y))
        below_y += score_surf.get_height() + 12

    if total_games > 1 and game_num < total_games:
        note = f"Click or press any key — game {game_num + 1} of {total_games}"
    else:
        note = "Click or press any key to continue"
    note_text = note_font.render(note, True, LBL_COLOR)
    screen.blit(note_text, (cx - note_text.get_width() // 2, below_y))


def draw_series_result(
    screen: pygame.Surface,
    score: tuple[int, int, int],
    p1_name: str,
    p2_name: str,
) -> None:
    title_font = pygame.font.SysFont("Segoe UI", 64, bold=True)
    big_font = pygame.font.SysFont("Segoe UI", 90, bold=True)
    score_font = pygame.font.SysFont("Segoe UI", 42, bold=True)
    note_font = pygame.font.SysFont("Segoe UI", 26)

    x_wins, o_wins, draws = score

    # full dark wash
    wash = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    wash.fill((15, 15, 20, 200))
    screen.blit(wash, (0, 0))

    cx = SCREEN_W // 2
    cy = SCREEN_H // 2

    # card
    card_w, card_h = 650, 420
    card_rect = pygame.Rect(cx - card_w // 2, cy - card_h // 2, card_w, card_h)
    _draw_rounded_rect(screen, CARD_BG, card_rect, 20)
    _draw_rounded_rect(screen, CARD_BORDER, card_rect, 20, 2)

    title_surf = title_font.render("Series Complete", True, TEXT_COLOR)
    screen.blit(title_surf, (cx - title_surf.get_width() // 2, card_rect.y + 30))

    if x_wins > o_wins:
        winner_msg, winner_col = "X wins the series!", X_COLOR
    elif o_wins > x_wins:
        winner_msg, winner_col = "O wins the series!", O_COLOR
    else:
        winner_msg, winner_col = "Series tied!", LBL_COLOR

    win_surf = big_font.render(winner_msg, True, winner_col)
    screen.blit(win_surf, (cx - win_surf.get_width() // 2, card_rect.y + 110))

    breakdown = f"X: {x_wins}   O: {o_wins}   Draw: {draws}"
    score_surf = score_font.render(breakdown, True, LBL_COLOR)
    screen.blit(score_surf, (cx - score_surf.get_width() // 2, card_rect.y + 230))

    # score bar
    total = x_wins + o_wins + draws
    if total > 0:
        bar_w = 500
        bar_h = 24
        bar_x = cx - bar_w // 2
        bar_y = card_rect.y + 290
        _draw_rounded_rect(
            screen,
            pygame.Color(25, 25, 35),
            pygame.Rect(bar_x, bar_y, bar_w, bar_h),
            12,
        )
        drawn = 0
        for wins, color in ((x_wins, X_COLOR), (o_wins, O_COLOR), (draws, LBL_COLOR)):
            seg_w = int(bar_w * wins / total)
            if seg_w > 0:
                seg_rect = pygame.Rect(bar_x + drawn, bar_y, seg_w, bar_h)
                pygame.draw.rect(screen, color, seg_rect)
                drawn += seg_w

    note_surf = note_font.render(
        "Click or press any key to return to menu", True, LBL_COLOR
    )
    screen.blit(note_surf, (cx - note_surf.get_width() // 2, card_rect.bottom - 50))


def idx_to_label(rc: Tuple[int, int]) -> str:
    r, c = rc
    return f"{chr(65 + r)}{c + 1}"


def render_inner_board(board: Board, size: int) -> pygame.Surface:
    surface = pygame.Surface((size, size))
    surface.fill(BOARD_BG)

    cell = size // 3
    m = INNER_MARGIN
    end = size - INNER_MARGIN

    for i in range(1, 3):
        x = i * cell
        y = i * cell
        pygame.draw.line(surface, LINE_COLOR, (x, m), (x, end), width=GRID_THIN)
        pygame.draw.line(surface, LINE_COLOR, (m, y), (end, y), width=GRID_THIN)

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
    lbl_font = pygame.font.SysFont("Segoe UI", 22)
    # top header
    header_rect = pygame.Rect(BOARD_LEFT, 0, BOARD_SIZE, HEADER_H)
    pygame.draw.rect(screen, BAR_BG, header_rect)
    pygame.draw.line(
        screen,
        BORDER,
        (BOARD_LEFT, HEADER_H - 1),
        (BOARD_LEFT + BOARD_SIZE, HEADER_H - 1),
        1,
    )

    # left sidebar
    sidebar_rect = pygame.Rect(BOARD_LEFT - SIDEBAR_W, BOARD_TOP, SIDEBAR_W, BOARD_SIZE)
    pygame.draw.rect(screen, BAR_BG, sidebar_rect)
    pygame.draw.line(
        screen,
        BORDER,
        (BOARD_LEFT - 1, BOARD_TOP),
        (BOARD_LEFT - 1, BOARD_TOP + BOARD_SIZE),
        1,
    )

    for c in range(3):
        text = lbl_font.render(str(c + 1), True, LBL_COLOR)
        x = BOARD_LEFT + c * big_cell + big_cell // 2 - text.get_width() // 2
        y = HEADER_H // 2 - text.get_height() // 2
        screen.blit(text, (x, y))

    for r in range(3):
        text = lbl_font.render(chr(65 + r), True, LBL_COLOR)
        x = BOARD_LEFT - SIDEBAR_W // 2 - text.get_width() // 2
        y = BOARD_TOP + r * big_cell + big_cell // 2 - text.get_height() // 2
        screen.blit(text, (x, y))


def draw_main_board(
    screen: pygame.Surface, board: Board, restriction: Tuple[int, int] | None
) -> int:
    inner_size = BOARD_SIZE // 3
    board_rect = pygame.Rect(BOARD_LEFT, BOARD_TOP, BOARD_SIZE, BOARD_SIZE)

    pygame.draw.rect(screen, BOARD_BG, board_rect)
    pygame.draw.rect(screen, BORDER, board_rect, width=1)

    for r in range(3):
        for c in range(3):
            x = BOARD_LEFT + c * inner_size
            y = BOARD_TOP + r * inner_size

            surface = render_inner_board(board[r][c], inner_size)
            screen.blit(surface, (x, y))

            if restriction is not None and restriction != (r, c):
                dim = pygame.Surface((inner_size, inner_size), pygame.SRCALPHA)
                dim.fill((10, 10, 15, 40))
                screen.blit(dim, (x, y))
            elif restriction == (r, c):
                hl = pygame.Surface((inner_size, inner_size), pygame.SRCALPHA)
                hl.fill((100, 110, 200, 18))
                screen.blit(hl, (x, y))

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

    pad_big = inner_size // 2 - 18
    for r in range(3):
        for c in range(3):
            val = board_state_to_piece(board[r][c].board_state)
            if val == Piece.EMPTY:
                continue
            cx = BOARD_LEFT + c * inner_size + inner_size // 2
            cy = BOARD_TOP + r * inner_size + inner_size // 2
            # dim won board
            dim = pygame.Surface((inner_size, inner_size), pygame.SRCALPHA)
            dim.fill((15, 15, 20, 100))
            screen.blit(dim, (BOARD_LEFT + c * inner_size, BOARD_TOP + r * inner_size))
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
    score_font = pygame.font.SysFont("Segoe UI", 20, bold=True)

    y0 = SCREEN_H - STATUS_BAR_H
    pygame.draw.rect(screen, BAR_BG, pygame.Rect(0, y0, SCREEN_W, STATUS_BAR_H))
    pygame.draw.line(screen, BORDER, (0, y0), (SCREEN_W, y0), 1)

    p1_col = X_COLOR if p1.piece == Piece.X else O_COLOR
    p2_col = X_COLOR if p2.piece == Piece.X else O_COLOR
    p1_label = FONT_BOLD.render(p1.get_name(), True, p1_col)
    p2_label = FONT_BOLD.render(p2.get_name(), True, p2_col)

    # current turn indicator
    turn_col = X_COLOR if current.piece == Piece.X else O_COLOR
    turn_piece = "X" if current.piece == Piece.X else "O"

    # dot indicator
    dot_radius = 6
    pad = 16

    screen.blit(p1_label, (pad, y0 + 14))
    screen.blit(p2_label, (pad, y0 + 14 + p1_label.get_height() + 6))

    # draw active indicator dot
    if current is p1:
        pygame.draw.circle(
            screen,
            p1_col,
            (pad + p1_label.get_width() + 14, y0 + 14 + p1_label.get_height() // 2),
            dot_radius,
        )
    else:
        pygame.draw.circle(
            screen,
            p2_col,
            (
                pad + p2_label.get_width() + 14,
                y0 + 14 + p2_label.get_height() + 6 + p2_label.get_height() // 2,
            ),
            dot_radius,
        )

    # center: restriction
    rest_text = "Any" if restriction is None else idx_to_label(restriction)
    rest_label = FONT.render(f"Target: {rest_text}", True, LBL_COLOR)
    mid_x = SCREEN_W // 2 - rest_label.get_width() // 2
    screen.blit(
        rest_label, (mid_x, y0 + STATUS_BAR_H // 2 - rest_label.get_height() // 2)
    )

    # right side
    right_x = SCREEN_W - pad
    if score is not None and total_games > 1:
        x_wins, o_wins, draws = score
        game_line = score_font.render(
            f"Game {game_num} / {total_games}", True, LBL_COLOR
        )
        score_line = score_font.render(
            f"X: {x_wins}   O: {o_wins}   Draw: {draws}", True, LBL_COLOR
        )
        screen.blit(game_line, (right_x - game_line.get_width(), y0 + 14))
        screen.blit(
            score_line,
            (right_x - score_line.get_width(), y0 + 14 + game_line.get_height() + 6),
        )

    # warning / thinking
    warn_y = y0 + STATUS_BAR_H // 2
    if time.time() < last_invalid_until:
        warn = FONT_BOLD.render("Invalid move!", True, WARN_COLOR)
        if total_games > 1:
            screen.blit(warn, (SCREEN_W // 2 - warn.get_width() // 2, y0 + 14))
        else:
            screen.blit(
                warn, (right_x - warn.get_width(), warn_y - warn.get_height() // 2)
            )

    if thinking:
        dots = "." * (int(time.time() * 3) % 4)
        msg = FONT.render(f"Thinking{dots}", True, ACCENT)
        if total_games > 1:
            screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, y0 + 14))
        else:
            screen.blit(
                msg, (right_x - msg.get_width(), warn_y - msg.get_height() // 2)
            )


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
    if choice == "alphazero":
        model_name = params.get("model")
        sims = int(params.get("sims", AZ_DEFAULT_SIMS))
        if model_name:
            path = os.path.join(AZ_MODELS_DIR, f"{model_name}.pt")
            if os.path.exists(path):
                p = AlphaZeroPlayer.load(path, piece=piece, num_simulations=sims)
                p.name = f"AlphaZero ({_format_model_label(model_name)})"
                return p
        return AlphaZeroPlayer(piece=piece, num_simulations=sims)
    raise ValueError(f"Unknown player choice: {choice}")


# ─── Menu UI Components ───────────────────────────────────────────────


# Algorithm catalogue. Each entry has a short, human-readable hint that the
# menu shows under the algorithm name — handy when you don't remember which
# player is which.
_ALGORITHMS: list[dict] = [
    {"label": "Human",       "key": "human",      "hint": "you, with a mouse",                "defaults": {}},
    {"label": "Minimax",     "key": "minimax",    "hint": "classical alpha-beta search",       "defaults": {"depth": MINIMAX_DEFAULT_DEPTH, "heuristics": MINIMAX_DEFAULT_HEURISTICS, "pruning": MINIMAX_DEFAULT_PRUNING}},
    {"label": "Monte Carlo", "key": "mcts",       "hint": "UCB-guided random rollouts",        "defaults": {"iters": MC_DEFAULT_ITERS}},
    {"label": "Q-Learning",  "key": "qlearning",  "hint": "tabular RL, pre-trained checkpoint","defaults": {"model": None}},
    {"label": "DQN",         "key": "dqn",        "hint": "deep Q-network, pre-trained",       "defaults": {"model": None}},
    {"label": "AlphaZero",   "key": "alphazero",  "hint": "PUCT-MCTS + policy/value net",      "defaults": {"model": None, "sims": AZ_DEFAULT_SIMS}},
]


def menu() -> tuple[tuple[str, dict], tuple[str, dict], int, bool] | None:
    """Main menu — choose the X and O opponents, set series options, start."""
    ql_models = _list_models(MODELS_DIR, ".pkl")
    dqn_models = _list_models(DQN_MODELS_DIR, ".pt")
    az_models = _list_models(AZ_MODELS_DIR, ".pt")
    model_pools = {"qlearning": ql_models, "dqn": dqn_models, "alphazero": az_models}

    # Resolve catalogue → concrete options with the latest model preselected.
    options: list[dict] = []
    for algo in _ALGORITHMS:
        defaults = dict(algo["defaults"])
        if "model" in defaults:
            pool = model_pools.get(algo["key"], [])
            defaults["model"] = pool[-1] if pool else None
        options.append({**algo, "params": defaults})

    param_specs: dict[str, list[dict]] = {
        "minimax": [
            {"name": "depth", "label": "Depth", "type": "int", "step": 1, "min": 0},
            {"name": "heuristics", "label": "Heuristics", "type": "bool"},
            {"name": "pruning", "label": "Pruning", "type": "bool"},
        ],
        "mcts": [
            {"name": "iters", "label": "Simulations", "type": "int", "step": 100, "min": 100},
        ],
        "qlearning": (
            [{"name": "model", "label": "Model", "type": "cycle", "choices": ql_models}]
            if ql_models else []
        ),
        "dqn": (
            [{"name": "model", "label": "Model", "type": "cycle", "choices": dqn_models}]
            if dqn_models else []
        ),
        "alphazero": (
            ([{"name": "model", "label": "Model", "type": "cycle", "choices": az_models}]
             if az_models else [])
            + [{"name": "sims", "label": "Simulations", "type": "int", "step": 50, "min": 10}]
        ),
    }

    # State. Default to Human vs Minimax — more interesting than Human-vs-Human.
    selected = [0, 1]
    params = [
        {opt["key"]: dict(opt["params"]) for opt in options},
        {opt["key"]: dict(opt["params"]) for opt in options},
    ]
    num_games = 1
    auto_skip = False

    # Five font roles, used consistently across the menu.
    f_title  = pygame.font.SysFont("Segoe UI", 48, bold=True)
    f_cap    = pygame.font.SysFont("Segoe UI", 14, bold=True)
    f_head   = pygame.font.SysFont("Segoe UI", 20, bold=True)
    f_body   = pygame.font.SysFont("Segoe UI", 17)
    f_body_b = pygame.font.SysFont("Segoe UI", 17, bold=True)
    f_hint   = pygame.font.SysFont("Segoe UI", 13)
    f_sym    = pygame.font.SysFont("Segoe UI", 20, bold=True)
    f_start  = pygame.font.SysFont("Segoe UI", 22, bold=True)
    f_count  = pygame.font.SysFont("Segoe UI", 22, bold=True)
    f_kbd    = pygame.font.SysFont("Segoe UI", 12)

    def _result():
        lk = options[selected[0]]["key"]
        rk = options[selected[1]]["key"]
        return (lk, dict(params[0][lk])), (rk, dict(params[1][rk])), num_games, auto_skip

    DIM = pygame.Color(115, 120, 138)

    while True:
        mouse_pos = pygame.mouse.get_pos()

        # ── responsive layout ──────────────────────────────────────────────
        HDR_H = 120
        ACTION_H = 96
        FOOTER_H = 26
        OUTER_PAD = 32
        GAP_X = 24

        cards_top = HDR_H
        cards_bot = SCREEN_H - ACTION_H - FOOTER_H - 16
        col_w = max(300, min(440, (SCREEN_W - OUTER_PAD * 2 - GAP_X) // 2))
        col_h = max(360, cards_bot - cards_top)
        cards_total_w = col_w * 2 + GAP_X
        left_x = (SCREEN_W - cards_total_w) // 2
        right_x = left_x + col_w + GAP_X
        card_rects = (
            pygame.Rect(left_x, cards_top, col_w, col_h),
            pygame.Rect(right_x, cards_top, col_w, col_h),
        )

        ALGO_TOP = 50
        ALGO_ROW_H = 34
        ALGO_GAP = 4
        n_algos = len(options)
        algo_block_h = n_algos * ALGO_ROW_H + (n_algos - 1) * ALGO_GAP
        HINT_H = 20
        SETTINGS_TOP = ALGO_TOP + algo_block_h + HINT_H + 18

        # Precompute hit rects for clicks (used by both draw and event handling).
        algo_rects: list[list[pygame.Rect]] = [[], []]
        for side, card in enumerate(card_rects):
            ix = card.x + 16
            iw = card.width - 32
            for i in range(n_algos):
                y = card.y + ALGO_TOP + i * (ALGO_ROW_H + ALGO_GAP)
                algo_rects[side].append(pygame.Rect(ix, y, iw, ALGO_ROW_H))

        def _build_param_rects() -> list[dict[tuple, pygame.Rect]]:
            """Build hit rects for the currently-selected algorithm on each side.
            Called both before event handling and again after, so a click that
            switches algorithms doesn't leave the rects out of sync with the
            new selection during drawing."""
            rects: list[dict[tuple, pygame.Rect]] = [{}, {}]
            for s, c in enumerate(card_rects):
                ix_ = c.x + 16
                iw_ = c.width - 32
                ch = options[selected[s]]["key"]
                y0_ = c.y + SETTINGS_TOP
                for i_, sp in enumerate(param_specs.get(ch, [])):
                    yy = y0_ + i_ * 32
                    if sp["type"] == "int":
                        bw_, bh_ = 26, 24
                        inc = pygame.Rect(ix_ + iw_ - bw_, yy, bw_, bh_)
                        dec = pygame.Rect(inc.x - 56 - bw_, yy, bw_, bh_)
                        rects[s][(sp["name"], "dec")] = dec
                        rects[s][(sp["name"], "inc")] = inc
                    elif sp["type"] == "bool":
                        pw_, ph_ = 44, 22
                        pill = pygame.Rect(ix_ + iw_ - pw_, yy + 1, pw_, ph_)
                        rects[s][(sp["name"], "toggle")] = pill
                    elif sp["type"] == "cycle":
                        bw_, bh_ = 24, 24
                        lw = f_body.size(sp["label"])[0]
                        dec = pygame.Rect(ix_ + lw + 14, yy, bw_, bh_)
                        inc = pygame.Rect(ix_ + iw_ - bw_, yy, bw_, bh_)
                        rects[s][(sp["name"], "dec")] = dec
                        rects[s][(sp["name"], "inc")] = inc
            return rects

        param_rects = _build_param_rects()

        # Action bar — games stepper (left), auto-advance toggle (middle), Start (right).
        bar_y = SCREEN_H - ACTION_H - FOOTER_H
        bar_h = ACTION_H - 16
        bar_w = cards_total_w
        bar_x = (SCREEN_W - bar_w) // 2
        bar_rect = pygame.Rect(bar_x, bar_y, bar_w, bar_h)

        games_btn = 30
        games_y = bar_y + (bar_h - games_btn) // 2
        games_dec = pygame.Rect(bar_x + 80, games_y, games_btn, games_btn)
        games_inc = pygame.Rect(games_dec.right + 56, games_y, games_btn, games_btn)

        as_pill_w, as_pill_h = 42, 22
        as_pill = pygame.Rect(
            bar_x + bar_w // 2 + 20, bar_y + (bar_h - as_pill_h) // 2,
            as_pill_w, as_pill_h,
        )

        start_w, start_h = 170, 46
        start_rect = pygame.Rect(
            bar_x + bar_w - start_w - 16, bar_y + (bar_h - start_h) // 2,
            start_w, start_h,
        )

        # ── events ─────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.VIDEORESIZE:
                _handle_resize(event)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_RETURN:
                    return _result()
                if event.key == pygame.K_LEFT:
                    num_games = max(1, num_games - 1)
                elif event.key == pygame.K_RIGHT:
                    num_games = min(MAX_GAMES, num_games + 1)
                elif event.key == pygame.K_a:
                    auto_skip = not auto_skip

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Snapshot the selection used to build this frame's rects, so a
                # click that switches algorithms doesn't then try to index param
                # rects of the *new* algorithm (which weren't built this frame).
                frame_selected = list(selected)
                # Parameter controls for the algorithm that was visible.
                for side in (0, 1):
                    choice = options[frame_selected[side]]["key"]
                    p = params[side][choice]
                    for spec in param_specs.get(choice, []):
                        name = spec["name"]
                        if spec["type"] == "int":
                            step = spec.get("step", 1)
                            lo = spec.get("min", 0)
                            if param_rects[side].get((name, "dec"), None) and \
                               param_rects[side][(name, "dec")].collidepoint(mx, my):
                                p[name] = max(lo, p[name] - step)
                            elif param_rects[side].get((name, "inc"), None) and \
                                 param_rects[side][(name, "inc")].collidepoint(mx, my):
                                p[name] = p[name] + step
                        elif spec["type"] == "bool":
                            if param_rects[side].get((name, "toggle"), None) and \
                               param_rects[side][(name, "toggle")].collidepoint(mx, my):
                                p[name] = not p[name]
                        elif spec["type"] == "cycle":
                            choices = spec["choices"]
                            if not choices:
                                continue
                            cur = p[name]
                            ci = choices.index(cur) if cur in choices else 0
                            if param_rects[side].get((name, "dec"), None) and \
                               param_rects[side][(name, "dec")].collidepoint(mx, my):
                                p[name] = choices[(ci - 1) % len(choices)]
                            elif param_rects[side].get((name, "inc"), None) and \
                                 param_rects[side][(name, "inc")].collidepoint(mx, my):
                                p[name] = choices[(ci + 1) % len(choices)]
                # Algorithm selection — done after param clicks so switching
                # algorithms doesn't double-fire with a stale param hit.
                for side in (0, 1):
                    for i, r in enumerate(algo_rects[side]):
                        if r.collidepoint(mx, my):
                            selected[side] = i
                # Action bar.
                if games_dec.collidepoint(mx, my):
                    num_games = max(1, num_games - 1)
                if games_inc.collidepoint(mx, my):
                    num_games = min(MAX_GAMES, num_games + 1)
                if as_pill.collidepoint(mx, my):
                    auto_skip = not auto_skip
                if start_rect.collidepoint(mx, my):
                    return _result()

        # The event loop may have switched algorithms; rebuild param rects so
        # the draw section can index them by the new selection's spec names.
        param_rects = _build_param_rects()

        # ── draw ───────────────────────────────────────────────────────────
        screen.fill(BG)

        # Title strip.
        title_surf = f_title.render("Ultimate Tic-Tac-Toe", True, TEXT_COLOR)
        screen.blit(title_surf, (SCREEN_W // 2 - title_surf.get_width() // 2, 28))
        cap_surf = f_cap.render("A I   A R E N A", True, ACCENT_HOVER)
        screen.blit(cap_surf, (SCREEN_W // 2 - cap_surf.get_width() // 2, 84))

        # Player cards.
        for side, card in enumerate(card_rects):
            _draw_rounded_rect(screen, CARD_BG, card, 14)
            _draw_rounded_rect(screen, CARD_BORDER, card, 14, 1)

            piece_color = X_COLOR if side == 0 else O_COLOR
            piece_label = "PLAYER X" if side == 0 else "PLAYER O"
            # Coloured accent rail down the left edge of the card.
            rail = pygame.Rect(card.x, card.y + 16, 4, card.height - 32)
            _draw_rounded_rect(screen, piece_color, rail, 2)

            head_surf = f_head.render(piece_label, True, piece_color)
            screen.blit(head_surf, (card.x + 16, card.y + 14))

            sel_idx = selected[side]
            sel_opt = options[sel_idx]

            # Algorithm rows.
            for i, opt in enumerate(options):
                r = algo_rects[side][i]
                is_sel = sel_idx == i
                hovered = r.collidepoint(mouse_pos)
                if is_sel:
                    _draw_rounded_rect(screen, ACCENT, r, 7)
                    tc = TEXT_COLOR
                    font = f_body_b
                else:
                    bg = BTN_HOVER if hovered else BTN_BG
                    _draw_rounded_rect(screen, bg, r, 7)
                    tc = TEXT_COLOR if hovered else LBL_COLOR
                    font = f_body
                surf = font.render(opt["label"], True, tc)
                screen.blit(surf, (r.x + 14, r.centery - surf.get_height() // 2))

            # Hint for the currently selected algorithm.
            hint_y = card.y + ALGO_TOP + algo_block_h + 6
            hint_surf = f_hint.render(sel_opt["hint"], True, DIM)
            screen.blit(hint_surf, (card.x + 16, hint_y))

            # Divider between algorithm picker and settings.
            div_y = card.y + SETTINGS_TOP - 14
            pygame.draw.line(
                screen, BORDER,
                (card.x + 16, div_y), (card.right - 16, div_y), 1,
            )

            # Settings rows for the selected algorithm.
            choice = sel_opt["key"]
            specs = param_specs.get(choice, [])
            y0 = card.y + SETTINGS_TOP
            if not specs:
                no_surf = f_hint.render("No settings for this player.", True, DIM)
                screen.blit(no_surf, (card.x + 16, y0 + 4))
            else:
                p = params[side][choice]
                for i, spec in enumerate(specs):
                    name = spec["name"]
                    y = y0 + i * 32
                    label_surf = f_body.render(spec["label"], True, LBL_COLOR)
                    screen.blit(label_surf, (card.x + 16, y + 4))

                    if spec["type"] == "int":
                        dec_r = param_rects[side][(name, "dec")]
                        inc_r = param_rects[side][(name, "inc")]
                        for r, sym in ((dec_r, "−"), (inc_r, "+")):
                            hov = r.collidepoint(mouse_pos)
                            _draw_rounded_rect(screen, BTN_HOVER if hov else BTN_BG, r, 5)
                            s = f_sym.render(sym, True, TEXT_COLOR)
                            screen.blit(s, (r.centerx - s.get_width() // 2,
                                            r.centery - s.get_height() // 2 - 1))
                        val_surf = f_body_b.render(str(p[name]), True, TEXT_COLOR)
                        mid_x = (dec_r.right + inc_r.left) // 2 - val_surf.get_width() // 2
                        screen.blit(val_surf, (mid_x, y + 4))

                    elif spec["type"] == "bool":
                        pill_r = param_rects[side][(name, "toggle")]
                        on = bool(p[name])
                        col = GREEN if on else pygame.Color(60, 62, 75)
                        _draw_rounded_rect(screen, col, pill_r, pill_r.height // 2)
                        knob_d = pill_r.height - 6
                        knob_x = (pill_r.right - knob_d - 3) if on else (pill_r.x + 3)
                        pygame.draw.circle(
                            screen, TEXT_COLOR,
                            (knob_x + knob_d // 2, pill_r.centery), knob_d // 2,
                        )

                    elif spec["type"] == "cycle":
                        choices = spec["choices"]
                        dec_r = param_rects[side][(name, "dec")]
                        inc_r = param_rects[side][(name, "inc")]
                        for r, sym in ((dec_r, "<"), (inc_r, ">")):
                            hov = r.collidepoint(mouse_pos)
                            _draw_rounded_rect(screen, BTN_HOVER if hov else BTN_BG, r, 5)
                            s = f_sym.render(sym, True, TEXT_COLOR)
                            screen.blit(s, (r.centerx - s.get_width() // 2,
                                            r.centery - s.get_height() // 2 - 1))
                        raw = p[name]
                        cur_label = _format_model_label(raw) if raw else "(none)"
                        idx_label = ""
                        if choices and raw in choices:
                            idx_label = f"  {choices.index(raw) + 1}/{len(choices)}"
                        val_text = f"{cur_label}{idx_label}"
                        val_surf = f_body_b.render(val_text, True, TEXT_COLOR)
                        mid_x = (dec_r.right + inc_r.left) // 2 - val_surf.get_width() // 2
                        screen.blit(val_surf, (mid_x, y + 4))

        # Action bar.
        _draw_rounded_rect(screen, CARD_BG, bar_rect, 14)
        _draw_rounded_rect(screen, CARD_BORDER, bar_rect, 14, 1)

        # Games stepper.
        gl = f_cap.render("GAMES", True, DIM)
        screen.blit(gl, (bar_x + 18, bar_y + 14))
        for r, sym in ((games_dec, "−"), (games_inc, "+")):
            hov = r.collidepoint(mouse_pos)
            _draw_rounded_rect(screen, BTN_HOVER if hov else BTN_BG, r, 6)
            s = f_sym.render(sym, True, TEXT_COLOR)
            screen.blit(s, (r.centerx - s.get_width() // 2,
                            r.centery - s.get_height() // 2 - 1))
        ng_surf = f_count.render(str(num_games), True, TEXT_COLOR)
        ng_x = (games_dec.right + games_inc.left) // 2 - ng_surf.get_width() // 2
        screen.blit(ng_surf, (ng_x, games_dec.centery - ng_surf.get_height() // 2))

        # Auto-advance toggle.
        al = f_cap.render("AUTO-ADVANCE", True, DIM)
        screen.blit(al, (as_pill.x - al.get_width() - 12,
                         as_pill.centery - al.get_height() // 2))
        on = auto_skip
        col = GREEN if on else pygame.Color(60, 62, 75)
        _draw_rounded_rect(screen, col, as_pill, as_pill.height // 2)
        knob_d = as_pill.height - 6
        knob_x = (as_pill.right - knob_d - 3) if on else (as_pill.x + 3)
        pygame.draw.circle(screen, TEXT_COLOR,
                           (knob_x + knob_d // 2, as_pill.centery), knob_d // 2)

        # Start button.
        hov = start_rect.collidepoint(mouse_pos)
        _draw_rounded_rect(screen, ACCENT_HOVER if hov else ACCENT, start_rect, 11)
        st_surf = f_start.render("Start", True, TEXT_COLOR)
        screen.blit(st_surf, (start_rect.centerx - st_surf.get_width() // 2,
                              start_rect.centery - st_surf.get_height() // 2))

        # Keyboard hints in the footer.
        kbd = "Enter: start    Esc: quit    Left/Right: games    A: auto-advance"
        kbd_surf = f_kbd.render(kbd, True, DIM)
        screen.blit(kbd_surf, (SCREEN_W // 2 - kbd_surf.get_width() // 2,
                               SCREEN_H - FOOTER_H + 4))

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

    screen.fill(BG)
    draw_main_board(screen, board, board.restriction)
    draw_status_bar(
        screen,
        p1,
        p2,
        current,
        board.restriction,
        last_invalid_until,
        score=score,
        game_num=game_num,
        total_games=total_games,
    )
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if ai_process is not None and ai_process.is_alive():
                    ai_process.terminate()
                    ai_process.join()
                return None
            if event.type == pygame.VIDEORESIZE:
                _handle_resize(event)
                set_layout(
                    screen_w=SCREEN_W,
                    screen_h=SCREEN_H,
                    board_size=BOARD_SIZE,
                    board_left=BOARD_LEFT,
                    board_top=BOARD_TOP,
                )

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
                screen,
                board.board_state,
                score=score,
                game_num=game_num,
                total_games=total_games,
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
                        if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                            waiting = False
                            break
                    clock.tick(60)
            return board.board_state

        pygame.display.flip()
        clock.tick(60)


def main() -> None:
    set_layout(
        screen_w=SCREEN_W,
        screen_h=SCREEN_H,
        board_size=BOARD_SIZE,
        board_left=BOARD_LEFT,
        board_top=BOARD_TOP,
    )

    while True:
        choices = menu()
        if choices is None:
            break
        (p1_choice, p1_params), (p2_choice, p2_params), num_games, auto_skip = choices

        # re-sync layout in case window was resized in menu
        set_layout(
            screen_w=SCREEN_W,
            screen_h=SCREEN_H,
            board_size=BOARD_SIZE,
            board_left=BOARD_LEFT,
            board_top=BOARD_TOP,
        )

        score = [0, 0, 0]
        p1_name = _make_player(p1_choice, Piece.X, p1_params).get_name()
        p2_name = _make_player(p2_choice, Piece.O, p2_params).get_name()

        for game_num in range(1, num_games + 1):
            result = game_loop(
                p1_choice,
                p1_params,
                p2_choice,
                p2_params,
                game_num=game_num,
                total_games=num_games,
                score=tuple(score),
                auto_skip=auto_skip,
            )
            if result is None:
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
                    if event.type == pygame.VIDEORESIZE:
                        _handle_resize(event)
                        screen.fill(BG)
                        draw_series_result(screen, tuple(score), p1_name, p2_name)
                        pygame.display.flip()
                    if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                        waiting = False
                        break
                clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
