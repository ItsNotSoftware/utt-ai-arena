from __future__ import annotations
import os
import time
import multiprocessing
import queue
from typing import Tuple

import pygame

from board import Board, BoardState, Piece, board_state_to_piece, get_board, Move
from player import HumanPlayer, MinimaxPlayer, Player, set_layout, MonteCarloPlayer, QLearningPlayer, DQNPlayer

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
MINIMAX_DEFAULT_DEPTH = 6
MINIMAX_DEFAULT_HEURISTICS = True
MINIMAX_DEFAULT_PRUNING = True
MODELS_DIR = "models/qlearning"
DQN_MODELS_DIR = "models/dqn"
MAX_GAMES = 9999


def _format_model_label(name: str) -> str:
    for prefix in ("q_table_", "dqn_"):
        if name.startswith(prefix):
            ep = name[len(prefix):]
            return f"{ep} ep"
    return name


def _parse_ep_count(name: str) -> int:
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
    if not os.path.isdir(directory):
        return []
    names = [f[: -len(ext)] for f in os.listdir(directory) if f.endswith(ext)]
    return sorted(names, key=_parse_ep_count)


def _compute_ai_move(
    player: Player, board_snapshot: Board, out_q: "queue.Queue"
) -> None:
    from time import perf_counter

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
        _draw_rounded_rect(screen, pygame.Color(25, 25, 35), pygame.Rect(bar_x, bar_y, bar_w, bar_h), 12)
        drawn = 0
        for wins, color in ((x_wins, X_COLOR), (o_wins, O_COLOR), (draws, LBL_COLOR)):
            seg_w = int(bar_w * wins / total)
            if seg_w > 0:
                seg_rect = pygame.Rect(bar_x + drawn, bar_y, seg_w, bar_h)
                pygame.draw.rect(screen, color, seg_rect)
                drawn += seg_w

    note_surf = note_font.render("Click or press any key to return to menu", True, LBL_COLOR)
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
                pygame.draw.line(surface, X_COLOR, (cx - pad, cy - pad), (cx + pad, cy + pad), width=6)
                pygame.draw.line(surface, X_COLOR, (cx - pad, cy + pad), (cx + pad, cy - pad), width=6)
            else:
                pygame.draw.circle(surface, O_COLOR, (cx, cy), pad, width=6)

    return surface


def draw_labels(screen: pygame.Surface, big_cell: int) -> None:
    lbl_font = pygame.font.SysFont("Segoe UI", 22)
    # top header
    header_rect = pygame.Rect(BOARD_LEFT, 0, BOARD_SIZE, HEADER_H)
    pygame.draw.rect(screen, BAR_BG, header_rect)
    pygame.draw.line(screen, BORDER, (BOARD_LEFT, HEADER_H - 1), (BOARD_LEFT + BOARD_SIZE, HEADER_H - 1), 1)

    # left sidebar
    sidebar_rect = pygame.Rect(BOARD_LEFT - SIDEBAR_W, BOARD_TOP, SIDEBAR_W, BOARD_SIZE)
    pygame.draw.rect(screen, BAR_BG, sidebar_rect)
    pygame.draw.line(screen, BORDER, (BOARD_LEFT - 1, BOARD_TOP), (BOARD_LEFT - 1, BOARD_TOP + BOARD_SIZE), 1)

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
        pygame.draw.rect(screen, LINE_COLOR, pygame.Rect(BOARD_LEFT, yy - DIV_W // 2, BOARD_SIZE, DIV_W))
        pygame.draw.rect(screen, LINE_COLOR, pygame.Rect(xx - DIV_W // 2, BOARD_TOP, DIV_W, BOARD_SIZE))

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
                pygame.draw.line(screen, X_COLOR, (cx - pad_big, cy - pad_big), (cx + pad_big, cy + pad_big), width=12)
                pygame.draw.line(screen, X_COLOR, (cx - pad_big, cy + pad_big), (cx + pad_big, cy - pad_big), width=12)
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
        pygame.draw.circle(screen, p1_col, (pad + p1_label.get_width() + 14, y0 + 14 + p1_label.get_height() // 2), dot_radius)
    else:
        pygame.draw.circle(screen, p2_col, (pad + p2_label.get_width() + 14, y0 + 14 + p2_label.get_height() + 6 + p2_label.get_height() // 2), dot_radius)

    # center: restriction
    rest_text = "Any" if restriction is None else idx_to_label(restriction)
    rest_label = FONT.render(f"Target: {rest_text}", True, LBL_COLOR)
    mid_x = SCREEN_W // 2 - rest_label.get_width() // 2
    screen.blit(rest_label, (mid_x, y0 + STATUS_BAR_H // 2 - rest_label.get_height() // 2))

    # right side
    right_x = SCREEN_W - pad
    if score is not None and total_games > 1:
        x_wins, o_wins, draws = score
        game_line = score_font.render(f"Game {game_num} / {total_games}", True, LBL_COLOR)
        score_line = score_font.render(f"X: {x_wins}   O: {o_wins}   Draw: {draws}", True, LBL_COLOR)
        screen.blit(game_line, (right_x - game_line.get_width(), y0 + 14))
        screen.blit(score_line, (right_x - score_line.get_width(), y0 + 14 + game_line.get_height() + 6))

    # warning / thinking
    warn_y = y0 + STATUS_BAR_H // 2
    if time.time() < last_invalid_until:
        warn = FONT_BOLD.render("Invalid move!", True, WARN_COLOR)
        if total_games > 1:
            screen.blit(warn, (SCREEN_W // 2 - warn.get_width() // 2, y0 + 14))
        else:
            screen.blit(warn, (right_x - warn.get_width(), warn_y - warn.get_height() // 2))

    if thinking:
        dots = "." * (int(time.time() * 3) % 4)
        msg = FONT.render(f"Thinking{dots}", True, ACCENT)
        if total_games > 1:
            screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, y0 + 14))
        else:
            screen.blit(msg, (right_x - msg.get_width(), warn_y - msg.get_height() // 2))


def _make_player(choice: str, piece: Piece, params: dict | None = None) -> Player:
    params = params or {}
    if choice == "human":
        return HumanPlayer(piece)
    if choice == "minimax":
        return MinimaxPlayer(
            piece,
            depth_limit=int(params.get("depth", MINIMAX_DEFAULT_DEPTH)),
            use_heuristic_eval=bool(params.get("heuristics", MINIMAX_DEFAULT_HEURISTICS)),
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


# ─── Menu UI Components ───────────────────────────────────────────────

class _Button:
    """Simple clickable button for the menu."""
    def __init__(self, rect, label, font=None, color=None, hover_color=None, text_color=None, radius=8):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.font = font
        self.color = color or BTN_BG
        self.hover_color = hover_color or BTN_HOVER
        self.text_color = text_color or TEXT_COLOR
        self.radius = radius

    def draw(self, surface, mouse_pos, selected=False):
        hovered = self.rect.collidepoint(mouse_pos)
        if selected:
            _draw_rounded_rect(surface, ACCENT, self.rect, self.radius)
            inner = self.rect.inflate(-4, -4)
            _draw_rounded_rect(surface, CARD_BG, inner, self.radius - 2)
            tc = TEXT_COLOR
        else:
            bg = self.hover_color if hovered else self.color
            _draw_rounded_rect(surface, bg, self.rect, self.radius)
            _draw_rounded_rect(surface, CARD_BORDER, self.rect, self.radius, 1)
            tc = self.text_color

        f = self.font or FONT
        text = f.render(self.label, True, tc)
        surface.blit(text, (self.rect.centerx - text.get_width() // 2,
                            self.rect.centery - text.get_height() // 2))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


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
            "label": "Monte Carlo",
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
            {"name": "depth", "label": "Depth", "type": "int", "step": 1, "min": 0, "note": "affects speed"},
            {"name": "heuristics", "label": "Heuristic Eval", "type": "bool"},
            {"name": "pruning", "label": "Alpha-Beta Pruning", "type": "bool", "note": "affects speed"},
        ],
        "mcts": [
            {"name": "iters", "label": "Simulations", "type": "int", "step": 100, "min": 100, "note": "affects speed"},
        ],
        "qlearning": (
            [{"name": "model", "label": "Model", "type": "cycle", "choices": ql_models}]
            if ql_models else []
        ),
        "dqn": (
            [{"name": "model", "label": "Model", "type": "cycle", "choices": dqn_models}]
            if dqn_models else []
        ),
    }

    selected = [0, 0]
    params = [
        {opt["key"]: dict(opt["params"]) for opt in options},
        {opt["key"]: dict(opt["params"]) for opt in options},
    ]
    num_games = 1
    auto_skip = False

    title_font = pygame.font.SysFont("Segoe UI", 44, bold=True)
    subtitle_font = pygame.font.SysFont("Segoe UI", 18)
    sub_font = pygame.font.SysFont("Segoe UI", 24, bold=True)
    param_font = pygame.font.SysFont("Segoe UI", 20)
    param_font_bold = pygame.font.SysFont("Segoe UI", 20, bold=True)
    small_font = pygame.font.SysFont("Segoe UI", 18)
    ng_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
    btn_sym_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
    small_btn_font = pygame.font.SysFont("Segoe UI", 16, bold=True)

    gap = 60
    top = 200
    opt_h = 50
    opt_gap = 10
    ng_btn_size = 36

    def _get_choices():
        left_choice = options[selected[0]]["key"]
        right_choice = options[selected[1]]["key"]
        return left_choice, params[0][left_choice], right_choice, params[1][right_choice]

    def _return_result():
        lc, lp, rc, rp = _get_choices()
        return (lc, dict(lp)), (rc, dict(rp)), num_games, auto_skip

    # Build param button rects (rebuilt each frame based on current selection)
    param_buttons = {}

    while True:
        mouse_pos = pygame.mouse.get_pos()
        # Responsive column width: scale with window, min 340, max 440
        col_w = max(340, min(440, (SCREEN_W - gap - 40) // 2))
        left_x = SCREEN_W // 2 - col_w - gap // 2
        right_x = SCREEN_W // 2 + gap // 2
        params_y = top + len(options) * (opt_h + opt_gap) + 20

        # Recalculate dynamic rects each frame
        _cx = SCREEN_W // 2
        ng_panel_y = SCREEN_H - 260
        start_rect = pygame.Rect(_cx - 150, SCREEN_H - 100, 300, 56)
        ng_dec5_rect = pygame.Rect(_cx - 140, ng_panel_y + 38, ng_btn_size, ng_btn_size)
        ng_dec_rect = pygame.Rect(_cx - 96, ng_panel_y + 38, ng_btn_size, ng_btn_size)
        ng_inc_rect = pygame.Rect(_cx + 60, ng_panel_y + 38, ng_btn_size, ng_btn_size)
        ng_inc5_rect = pygame.Rect(_cx + 104, ng_panel_y + 38, ng_btn_size, ng_btn_size)
        as_toggle_rect = pygame.Rect(_cx - 120, ng_panel_y + 88, 22, 22)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.VIDEORESIZE:
                _handle_resize(event)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_RETURN:
                    return _return_result()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # algorithm selection buttons
                for idx in range(len(options)):
                    y = top + idx * (opt_h + opt_gap)
                    if pygame.Rect(left_x, y, col_w, opt_h).collidepoint(mx, my):
                        selected[0] = idx
                    if pygame.Rect(right_x, y, col_w, opt_h).collidepoint(mx, my):
                        selected[1] = idx

                # param buttons
                for (pidx, sname, action), rect in param_buttons.items():
                    if rect.collidepoint(mx, my):
                        choice = options[selected[pidx]]["key"]
                        p = params[pidx][choice]
                        specs = param_specs.get(choice, [])
                        spec = next((s for s in specs if s["name"] == sname), None)
                        if spec is None:
                            continue
                        if spec["type"] == "int":
                            step = spec.get("step", 1)
                            min_val = spec.get("min", 0)
                            if action == "dec":
                                p[sname] = max(min_val, p[sname] - step)
                            elif action == "inc":
                                p[sname] = p[sname] + step
                        elif spec["type"] == "bool":
                            if action == "toggle":
                                p[sname] = not p[sname]
                        elif spec["type"] == "cycle":
                            choices = spec["choices"]
                            if choices:
                                cur = p[sname]
                                ci = choices.index(cur) if cur in choices else 0
                                if action == "dec":
                                    p[sname] = choices[(ci - 1) % len(choices)]
                                elif action == "inc":
                                    p[sname] = choices[(ci + 1) % len(choices)]

                # num games
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

        # ─── Draw ───
        screen.fill(BG)

        # Title
        title = title_font.render("Ultimate Tic-Tac-Toe", True, TEXT_COLOR)
        screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 30))
        subtitle = subtitle_font.render("AI Arena", True, LBL_COLOR)
        screen.blit(subtitle, (SCREEN_W // 2 - subtitle.get_width() // 2, 80))

        # Divider line
        pygame.draw.line(screen, BORDER, (SCREEN_W // 2 - 200, 115), (SCREEN_W // 2 + 200, 115), 1)

        # Player column headers
        p1_header = sub_font.render("Player X", True, X_COLOR)
        p2_header = sub_font.render("Player O", True, O_COLOR)
        screen.blit(p1_header, (left_x + col_w // 2 - p1_header.get_width() // 2, top - 38))
        screen.blit(p2_header, (right_x + col_w // 2 - p2_header.get_width() // 2, top - 38))

        # Algorithm selection buttons
        for idx, opt in enumerate(options):
            y = top + idx * (opt_h + opt_gap)
            for side, sel, sx in ((0, selected[0], left_x), (1, selected[1], right_x)):
                rect = pygame.Rect(sx, y, col_w, opt_h)
                is_sel = sel == idx
                hovered = rect.collidepoint(mouse_pos)

                if is_sel:
                    _draw_rounded_rect(screen, ACCENT, rect, 10)
                    inner = rect.inflate(-4, -4)
                    _draw_rounded_rect(screen, CARD_BG, inner, 8)
                    tc = TEXT_COLOR
                elif hovered:
                    _draw_rounded_rect(screen, BTN_HOVER, rect, 10)
                    _draw_rounded_rect(screen, CARD_BORDER, rect, 10, 1)
                    tc = TEXT_COLOR
                else:
                    _draw_rounded_rect(screen, BTN_BG, rect, 10)
                    _draw_rounded_rect(screen, CARD_BORDER, rect, 10, 1)
                    tc = LBL_COLOR

                text = FONT.render(opt["label"], True, tc)
                screen.blit(text, (rect.centerx - text.get_width() // 2,
                                   rect.centery - text.get_height() // 2))

        # ─── Parameter controls ───
        param_buttons.clear()

        def _render_param_controls(player_idx, px, choice, p):
            specs = param_specs.get(choice, [])
            y = params_y
            row_h = 32

            if not specs:
                lbl = param_font.render("No parameters", True, pygame.Color(100, 105, 120))
                screen.blit(lbl, (px + 10, y))
                return y + row_h

            for spec in specs:
                name = spec["name"]
                label = spec["label"]

                lbl_surf = param_font.render(f"{label}:", True, LBL_COLOR)
                screen.blit(lbl_surf, (px + 10, y + 4))

                if spec["type"] == "int":
                    val_surf = param_font_bold.render(str(p[name]), True, TEXT_COLOR)
                    # buttons: [-] val [+] — right-aligned within column
                    bw, bh = 30, 26
                    inc_r = pygame.Rect(px + col_w - bw - 12, y + 2, bw, bh)
                    dec_r = pygame.Rect(inc_r.x - 50 - bw, y + 2, bw, bh)
                    val_x = dec_r.right + 25 - val_surf.get_width() // 2

                    for r, sym, act in ((dec_r, "-", "dec"), (inc_r, "+", "inc")):
                        hov = r.collidepoint(mouse_pos)
                        _draw_rounded_rect(screen, BTN_HOVER if hov else BTN_BG, r, 6)
                        _draw_rounded_rect(screen, CARD_BORDER, r, 6, 1)
                        s = btn_sym_font.render(sym, True, TEXT_COLOR)
                        screen.blit(s, (r.centerx - s.get_width() // 2, r.centery - s.get_height() // 2))
                        param_buttons[(player_idx, name, act)] = r

                    screen.blit(val_surf, (val_x, y + 4))

                elif spec["type"] == "bool":
                    # toggle pill — right-aligned
                    pill_w, pill_h = 48, 24
                    pill_r = pygame.Rect(px + col_w - pill_w - 12, y + 4, pill_w, pill_h)
                    on = p[name]
                    bg_col = GREEN if on else pygame.Color(60, 62, 75)
                    _draw_rounded_rect(screen, bg_col, pill_r, pill_h // 2)
                    knob_r = pill_h - 6
                    knob_x = pill_r.right - knob_r - 3 if on else pill_r.x + 3
                    pygame.draw.circle(screen, TEXT_COLOR, (knob_x + knob_r // 2, pill_r.centery), knob_r // 2)
                    param_buttons[(player_idx, name, "toggle")] = pill_r

                elif spec["type"] == "cycle":
                    choices = spec["choices"]
                    raw = p[name]
                    current = _format_model_label(raw) if raw else "(none)"
                    n = len(choices)
                    val_text = f"{current}  ({n})"
                    val_surf = param_font.render(val_text, True, TEXT_COLOR)

                    bw, bh = 26, 26
                    # < and > buttons with full column width between label and right edge
                    dec_r = pygame.Rect(px + lbl_surf.get_width() + 20, y + 2, bw, bh)
                    inc_r = pygame.Rect(px + col_w - bw - 12, y + 2, bw, bh)
                    mid_x = (dec_r.right + inc_r.left) // 2 - val_surf.get_width() // 2

                    for r, sym, act in ((dec_r, "<", "dec"), (inc_r, ">", "inc")):
                        hov = r.collidepoint(mouse_pos)
                        _draw_rounded_rect(screen, BTN_HOVER if hov else BTN_BG, r, 6)
                        _draw_rounded_rect(screen, CARD_BORDER, r, 6, 1)
                        s = btn_sym_font.render(sym, True, TEXT_COLOR)
                        screen.blit(s, (r.centerx - s.get_width() // 2, r.centery - s.get_height() // 2))
                        param_buttons[(player_idx, name, act)] = r

                    screen.blit(val_surf, (mid_x, y + 4))

                y += row_h
            return y

        left_choice = options[selected[0]]["key"]
        right_choice = options[selected[1]]["key"]
        _render_param_controls(0, left_x, left_choice, params[0][left_choice])
        _render_param_controls(1, right_x, right_choice, params[1][right_choice])

        # ─── Number of Games panel ───
        panel_w = 380
        panel_h = 130
        panel_rect = pygame.Rect(SCREEN_W // 2 - panel_w // 2, ng_panel_y - 4, panel_w, panel_h)
        _draw_rounded_rect(screen, CARD_BG, panel_rect, 14)
        _draw_rounded_rect(screen, CARD_BORDER, panel_rect, 14, 1)

        ng_title = small_font.render("Number of Games", True, LBL_COLOR)
        screen.blit(ng_title, (SCREEN_W // 2 - ng_title.get_width() // 2, ng_panel_y + 8))

        num_surf = ng_font.render(str(num_games), True, TEXT_COLOR)
        screen.blit(num_surf, (SCREEN_W // 2 - num_surf.get_width() // 2, ng_panel_y + 40))

        for btn_rect, symbol, is_small in (
            (ng_dec5_rect, "--", True),
            (ng_dec_rect, "-", False),
            (ng_inc_rect, "+", False),
            (ng_inc5_rect, "++", True),
        ):
            hov = btn_rect.collidepoint(mouse_pos)
            _draw_rounded_rect(screen, BTN_HOVER if hov else BTN_BG, btn_rect, 8)
            _draw_rounded_rect(screen, CARD_BORDER, btn_rect, 8, 1)
            f = small_btn_font if is_small else btn_sym_font
            sym = f.render(symbol, True, TEXT_COLOR)
            screen.blit(sym, (btn_rect.centerx - sym.get_width() // 2,
                              btn_rect.centery - sym.get_height() // 2))

        # divider
        div_y = ng_panel_y + 80
        pygame.draw.line(screen, BORDER, (panel_rect.x + 16, div_y), (panel_rect.right - 16, div_y), 1)

        # auto-skip
        cb = as_toggle_rect
        cb_on = auto_skip
        cb_bg = GREEN if cb_on else pygame.Color(55, 58, 75)
        _draw_rounded_rect(screen, cb_bg, cb, 5)
        _draw_rounded_rect(screen, CARD_BORDER, cb, 5, 1)
        if cb_on:
            pad = 5
            pygame.draw.line(screen, TEXT_COLOR, (cb.x + pad, cb.centery), (cb.centerx - 1, cb.bottom - pad), 2)
            pygame.draw.line(screen, TEXT_COLOR, (cb.centerx - 1, cb.bottom - pad), (cb.right - pad, cb.y + pad), 2)
        as_lbl = small_font.render("Auto-advance between games", True, LBL_COLOR)
        screen.blit(as_lbl, (cb.right + 10, cb.centery - as_lbl.get_height() // 2))

        # ─── Start button ───
        hov = start_rect.collidepoint(mouse_pos)
        btn_col = ACCENT_HOVER if hov else ACCENT
        _draw_rounded_rect(screen, btn_col, start_rect, 12)
        start_text = FONT_BOLD.render("Start Game", True, TEXT_COLOR)
        screen.blit(start_text, (start_rect.centerx - start_text.get_width() // 2,
                                  start_rect.centery - start_text.get_height() // 2))

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
        screen, p1, p2, current, board.restriction, last_invalid_until,
        score=score, game_num=game_num, total_games=total_games,
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
                    screen_w=SCREEN_W, screen_h=SCREEN_H,
                    board_size=BOARD_SIZE, board_left=BOARD_LEFT, board_top=BOARD_TOP,
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
            screen, p1, p2, current, board.restriction, last_invalid_until,
            thinking=not isinstance(current, HumanPlayer) and pending_move is None,
            score=score, game_num=game_num, total_games=total_games,
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
            screen_w=SCREEN_W, screen_h=SCREEN_H,
            board_size=BOARD_SIZE, board_left=BOARD_LEFT, board_top=BOARD_TOP,
        )

        score = [0, 0, 0]
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
