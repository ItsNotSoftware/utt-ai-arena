#!/usr/bin/env python3
"""AlphaZero-style training for Ultimate Tic-Tac-Toe.

Self-play with PUCT-MCTS guided by a policy+value network. Each move stores
(state, π, _) where π is the visit-count distribution over actions; at game
end the result z ∈ {-1, 0, +1} is stitched in from each player's perspective.
The network is trained to match (π, z) with cross-entropy + MSE loss.

Optimized for laptop GPUs (e.g. RTX 4060):
  - Batched self-play: N games run in lockstep, all MCTS leaf evaluations
    every simulation step are fused into a single GPU forward pass.
  - MCTS tree reuse across moves (kept subtree of chosen action).
  - TF32 + cudnn benchmark + (optional) torch.compile.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message="Can't initialize NVML")

import argparse
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from board import Board, BoardState, Piece, UndoToken, WIN_LINES, get_board, swap_piece
from alphazero_model import AlphaZeroNet

MODELS_DIR = "models/alphazero"
STATE_SHAPE = (7, 9, 9)
NUM_ACTIONS = 81

TRAINING_PROFILES: dict[str, dict[str, int | float]] = {
    # Fast enough to iterate on a laptop RTX 4060 without a long compile warmup.
    "fast": {
        "iterations": 80,
        "games_per_iter": 48,
        "simulations": 64,
        "train_steps": 96,
        "batch_size": 512,
        "buffer_size": 100_000,
        "augment_symmetries": 4,
        "mcts_batch_size": 1,
        "eval_interval": 5,
        "eval_games": 40,
        "eval_sims": 64,
        "workers": 0,
        "net_channels": 32,
        "net_blocks": 3,
        "amp": False,
    },
    # Better training signal while still using the GPU efficiently.
    "balanced": {
        "iterations": 200,
        "games_per_iter": 64,
        "simulations": 64,
        "train_steps": 160,
        "batch_size": 512,
        "buffer_size": 200_000,
        "augment_symmetries": 8,
        "mcts_batch_size": 1,
        "eval_interval": 5,
        "eval_games": 40,
        "eval_sims": 64,
        "workers": 0,
        "net_channels": 32,
        "net_blocks": 3,
        "amp": False,
    },
    # Mid-size run. Trains a 64ch × 5-block net with 64 self-play sims — strong
    # enough to draw against minimax and a good warm-start for the `strong`
    # profile. Pass --amp on CUDA; leaf batching (mcts_batch_size=4) and
    # workers=4 are the benchmarked sweet spot on an RTX 4060 Laptop.
    "turbo": {
        "iterations": 3000,
        "games_per_iter": 32,
        "simulations": 64,
        "train_steps": 160,
        "batch_size": 512,
        "buffer_size": 300_000,
        "augment_symmetries": 8,
        "mcts_batch_size": 4,
        "eval_interval": 50,
        "eval_games": 40,
        "eval_sims": 64,
        "workers": 4,
        "net_channels": 64,
        "net_blocks": 5,
        "amp": True,
    },
    # Wider/deeper net (96ch × 8 blocks) and doubled MCTS depth (128 sims).
    # The extra sims are the main play-strength multiplier — they sharpen the
    # self-play policy targets so the network actually learns to beat minimax,
    # not just draw it. mcts_batch_size=8 keeps the wider-net GPU utilization
    # high; workers=4 per the benchmarked sweet spot (re-check if you change
    # the net width — see feedback in memory).
    "strong": {
        "iterations": 4500,
        "games_per_iter": 32,
        "simulations": 128,
        "train_steps": 200,
        "batch_size": 512,
        "buffer_size": 400_000,
        "augment_symmetries": 8,
        "mcts_batch_size": 8,
        "eval_interval": 50,
        "eval_games": 40,
        "eval_sims": 128,
        "workers": 4,
        "net_channels": 96,
        "net_blocks": 8,
        "amp": True,
    },
}


# ── ANSI helpers ─────────────────────────────────────────────────────────────

_IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


def bold(t):
    return _c("1", t)


def dim(t):
    return _c("2", t)


def cyan(t):
    return _c("96", t)


def green(t):
    return _c("92", t)


def red(t):
    return _c("91", t)


def yellow(t):
    return _c("93", t)


def magenta(t):
    return _c("95", t)


def blue(t):
    return _c("94", t)


def _bar(frac, width=20):
    frac = min(1.0, max(0.0, frac))
    n = round(frac * width)
    return green("█" * n) + dim("░" * (width - n))


def _pct_bar(pct, width=28):
    n = round(min(100.0, max(0.0, pct)) / 100 * width)
    return green("█" * n) + dim("░" * (width - n))


def _box(title, subtitle: str | None = None, width=72):
    inner = f"  {title}  "
    pad = max(0, width - len(inner))
    line = "═" * (width + 2)
    out = (
        f"\n{cyan(bold('╔' + line + '╗'))}\n"
        f"{cyan(bold('║'))} {bold(inner)}{' ' * pad}{cyan(bold('║'))}\n"
    )
    if subtitle:
        sub = f"  {subtitle}  "
        out += f"{cyan(bold('║'))} {dim(sub)}{' ' * max(0, width - len(sub))}{cyan(bold('║'))}\n"
    return out + f"{cyan(bold('╚' + line + '╝'))}"


def _rule(width: int = 92) -> str:
    return dim("  " + "─" * width)


def _kv(label: str, value: str, note: str = "") -> str:
    note_part = f"  {dim(note)}" if note else ""
    label_text = f"{label}:".ljust(18)
    return f"  {dim(label_text)} {value}{note_part}"


def _metric(label: str, value: str, note: str = "") -> str:
    note_part = f" {dim(note)}" if note else ""
    return f"{dim(label)} {bold(value)}{note_part}"


def _phase(label: str) -> str:
    return f"  {blue('◆')} {bold(label)}"


def _fmt_time(s):
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(int(s), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _default_name(it):
    return f"az_{it}it"


def _profile_value(args, profile: str, name: str):
    value = getattr(args, name)
    if value is not None:
        return value
    return TRAINING_PROFILES[profile][name.replace("-", "_")]


def _configure_torch_runtime(device: torch.device) -> tuple[int, int]:
    """Tune PyTorch for this laptop-style workload and return thread counts."""
    cpu_threads = max(1, min(8, os.cpu_count() or 1))
    interop_threads = max(1, min(4, cpu_threads))
    try:
        torch.set_num_threads(cpu_threads)
    except RuntimeError:
        cpu_threads = torch.get_num_threads()
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        interop_threads = torch.get_num_interop_threads()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return cpu_threads, interop_threads


def _clean_state_dict(obj):
    """Accept raw state_dicts or checkpoints and strip torch.compile prefixes."""
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    prefix = "_orig_mod."
    if isinstance(obj, dict) and any(str(k).startswith(prefix) for k in obj):
        obj = {str(k).removeprefix(prefix): v for k, v in obj.items()}
    return obj


def _sym_coord(row: int, col: int, sym: int) -> tuple[int, int]:
    if sym == 0:  # identity
        return row, col
    if sym == 1:  # rotate 90
        return col, 8 - row
    if sym == 2:  # rotate 180
        return 8 - row, 8 - col
    if sym == 3:  # rotate 270
        return 8 - col, row
    if sym == 4:  # mirror left/right
        return row, 8 - col
    if sym == 5:  # mirror top/bottom
        return 8 - row, col
    if sym == 6:  # main diagonal
        return col, row
    if sym == 7:  # anti-diagonal
        return 8 - col, 8 - row
    raise ValueError(f"Unknown symmetry id {sym}")


def _build_action_symmetry_perms() -> tuple[np.ndarray, ...]:
    perms: list[np.ndarray] = []
    for sym in range(8):
        perm = np.empty(NUM_ACTIONS, dtype=np.int64)
        for action in range(NUM_ACTIONS):
            outer_r, rem = divmod(action, 27)
            outer_c, rem = divmod(rem, 9)
            inner_r, inner_c = divmod(rem, 3)
            row = outer_r * 3 + inner_r
            col = outer_c * 3 + inner_c
            new_row, new_col = _sym_coord(row, col, sym)
            new_outer_r, new_inner_r = divmod(new_row, 3)
            new_outer_c, new_inner_c = divmod(new_col, 3)
            perm[action] = (
                new_outer_r * 27
                + new_outer_c * 9
                + new_inner_r * 3
                + new_inner_c
            )
        perms.append(perm)
    return tuple(perms)


_ACTION_SYM_PERMS = _build_action_symmetry_perms()

_CELL_WIN_LINES = tuple(
    tuple(line for line in WIN_LINES if (r, c) in line)
    for r in range(3)
    for c in range(3)
)
_ACTION_COORDS = tuple(
    (
        action // 27,
        (action % 27) // 9,
        (action % 9) // 3,
        action % 3,
        (action // 27, (action % 27) // 9),
        ((action % 9) // 3, action % 3),
    )
    for action in range(NUM_ACTIONS)
)


def _transform_state(state: np.ndarray, sym: int) -> np.ndarray:
    if sym == 0:
        return state
    if sym <= 3:
        return np.rot90(state, -sym, axes=(1, 2))
    if sym == 4:
        return np.flip(state, axis=2)
    if sym == 5:
        return np.flip(state, axis=1)
    if sym == 6:
        return np.swapaxes(state, 1, 2)
    if sym == 7:
        return np.rot90(np.swapaxes(state, 1, 2), 2, axes=(1, 2))
    raise ValueError(f"Unknown symmetry id {sym}")


def _inner_state_after_place(
    inner: Board,
    piece: Piece,
    row: int,
    col: int,
) -> BoardState:
    cells = inner.board
    for a, b, c in _CELL_WIN_LINES[row * 3 + col]:
        if (
            cells[a[0]][a[1]] == piece
            and cells[b[0]][b[1]] == piece
            and cells[c[0]][c[1]] == piece
        ):
            return BoardState.X_WON if piece == Piece.X else BoardState.O_WON
    return BoardState.NOT_FINISHED if inner.empty_cells else BoardState.DRAW


def _main_state_after_outer_change(board: Board, outer_r: int, outer_c: int) -> BoardState:
    cells = board.board
    for a, b, c in _CELL_WIN_LINES[outer_r * 3 + outer_c]:
        st = cells[a[0]][a[1]].board_state
        if (
            (st == BoardState.X_WON or st == BoardState.O_WON)
            and st == cells[b[0]][b[1]].board_state
            and st == cells[c[0]][c[1]].board_state
        ):
            return st
    return BoardState.NOT_FINISHED if board.playable_outers_list else BoardState.DRAW


def _apply_action(board: Board, action: int, piece: Piece) -> UndoToken | None:
    """Fast flat-action version of Board.make_move for trainer hot loops."""
    outer_r, outer_c, inner_r, inner_c, out_rc, in_rc = _ACTION_COORDS[action]

    if board.restriction is not None and out_rc != board.restriction:
        return None

    inner = board.board[outer_r][outer_c]
    if inner.board_state != BoardState.NOT_FINISHED:
        return None
    if inner.board[inner_r][inner_c] != Piece.EMPTY:
        return None

    token = UndoToken(
        outer=out_rc,
        inner=in_rc,
        prev_inner_state=inner.board_state,
        prev_main_state=board.board_state,
        prev_restriction=board.restriction,
    )

    inner.board[inner_r][inner_c] = piece
    inner.empty_cells.discard(in_rc)
    inner.board_state = _inner_state_after_place(inner, piece, inner_r, inner_c)

    inner_finished = inner.board_state != BoardState.NOT_FINISHED
    if inner_finished and token.prev_inner_state == BoardState.NOT_FINISHED:
        if out_rc in board.playable_outers_set:
            token.removed_outer_index = board.playable_outers_list.index(out_rc)
            board.playable_outers_set.remove(out_rc)
            del board.playable_outers_list[token.removed_outer_index]
        board.board_state = _main_state_after_outer_change(board, outer_r, outer_c)
    else:
        board.board_state = token.prev_main_state

    target = board.board[inner_r][inner_c]
    board.restriction = in_rc if target.board_state == BoardState.NOT_FINISHED else None
    return token


def _undo_action(board: Board, token: UndoToken) -> None:
    """Fast Board.undo_move equivalent for tokens produced by _apply_action."""
    out_rc = token.outer
    in_rc = token.inner
    inner = board.board[out_rc[0]][out_rc[1]]

    inner.board[in_rc[0]][in_rc[1]] = Piece.EMPTY
    inner.empty_cells.add(in_rc)
    inner.board_state = token.prev_inner_state
    board.board_state = token.prev_main_state
    board.restriction = token.prev_restriction

    if token.removed_outer_index is not None:
        board.playable_outers_set.add(out_rc)
        if token.removed_outer_index >= len(board.playable_outers_list):
            board.playable_outers_list.append(out_rc)
        else:
            board.playable_outers_list.insert(token.removed_outer_index, out_rc)


# ── MCTS node ────────────────────────────────────────────────────────────────


@dataclass
class _Node:
    """MCTS node. Edge stats are kept as parallel Python lists aligned to
    `legal`, indexed by local idx; `children[i]` is the subtree for `legal[i]`
    (or None if unexpanded). Lists outperform dicts for the small action sets
    typical in UTT (k <= 9), avoiding hashing on every PUCT iteration.
    """

    turn: Piece
    legal: list[int] = field(default_factory=list)
    priors: list[float] = field(default_factory=list)
    N: list[int] = field(default_factory=list)
    W: list[float] = field(default_factory=list)
    children: list["_Node | None"] = field(default_factory=list)
    is_terminal: bool = False
    terminal_value: float = 0.0


def _make_node(turn: Piece, legal: list[int], priors: list[float]) -> _Node:
    k = len(legal)
    return _Node(
        turn=turn,
        legal=legal,
        priors=priors,
        N=[0] * k,
        W=[0.0] * k,
        children=[None] * k,
    )


def _terminal_node(turn: Piece, value: float) -> _Node:
    return _Node(turn=turn, is_terminal=True, terminal_value=value)


def _terminal_value(board: Board, turn: Piece) -> float | None:
    st = board.board_state
    if st == BoardState.NOT_FINISHED:
        return None
    if st == BoardState.DRAW:
        return 0.0
    if (st == BoardState.X_WON and turn == Piece.X) or (
        st == BoardState.O_WON and turn == Piece.O
    ):
        return 1.0
    return -1.0


def _puct_select(node: _Node, c_puct: float) -> int:
    """Return the local idx of the highest-PUCT-score child."""
    N = node.N
    total_n = sum(N) or 1
    sqrt_total = math.sqrt(total_n)
    P = node.priors
    W = node.W
    best_idx = 0
    best_score = -math.inf
    for i in range(len(N)):
        n = N[i]
        q = W[i] / n if n > 0 else 0.0
        score = q + c_puct * P[i] * sqrt_total / (1 + n)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _backup(path: list[tuple[_Node, int]], leaf_value: float) -> None:
    """Backup leaf value (from leaf's perspective) up the path, flipping each ply."""
    v = -leaf_value
    for parent, idx in reversed(path):
        parent.N[idx] += 1
        parent.W[idx] += v
        v = -v


def _add_virtual_visit(path: list[tuple[_Node, int]]) -> None:
    for parent, idx in path:
        parent.N[idx] += 1


def _revert_virtual_visit(path: list[tuple[_Node, int]]) -> None:
    for parent, idx in path:
        parent.N[idx] -= 1


def _legal_actions(
    board: Board,
    turn: Piece,
    legal_plane: np.ndarray | None = None,
) -> list[int]:
    """Return legal flat action ids without allocating Move objects."""
    if board.board_state != BoardState.NOT_FINISHED:
        return []

    use_restr = board.restriction
    if use_restr is not None and use_restr in board.playable_outers_set:
        outers = (use_restr,)
    else:
        outers = board.playable_outers_list

    actions: list[int] = []
    append = actions.append
    outer_cells = board.board
    for R, C in outers:
        inner = outer_cells[R][C]
        if inner.board_state != BoardState.NOT_FINISHED:
            continue
        base = R * 27 + C * 9
        r_off = R * 3
        c_off = C * 3
        for r, c in inner.empty_cells:
            append(base + r * 3 + c)
            if legal_plane is not None:
                legal_plane[r_off + r, c_off + c] = 1.0
    return actions


def _encode_board_into(out: np.ndarray, board: Board, piece: Piece) -> list[int]:
    """Encode a board into a preallocated numpy array and return legal actions."""
    out.fill(0.0)
    opp = Piece.O if piece == Piece.X else Piece.X
    own_won = BoardState.X_WON if piece == Piece.X else BoardState.O_WON
    opp_won = BoardState.O_WON if piece == Piece.X else BoardState.X_WON

    outer_cells = board.board
    for R in range(3):
        for C in range(3):
            inner = outer_cells[R][C]
            inner_cells = inner.board
            r_off = R * 3
            c_off = C * 3

            for r in range(3):
                row = r_off + r
                inner_row = inner_cells[r]
                for c in range(3):
                    cell = inner_row[c]
                    if cell == piece:
                        out[0, row, c_off + c] = 1.0
                    elif cell == opp:
                        out[1, row, c_off + c] = 1.0

            st = inner.board_state
            if st == own_won:
                out[3, r_off : r_off + 3, c_off : c_off + 3] = 1.0
            elif st == opp_won:
                out[4, r_off : r_off + 3, c_off : c_off + 3] = 1.0
            elif st == BoardState.DRAW:
                out[5, r_off : r_off + 3, c_off : c_off + 3] = 1.0

    legal = _legal_actions(board, piece, out[2])

    if board.restriction is not None:
        rR, rC = board.restriction
        out[6, rR * 3 : rR * 3 + 3, rC * 3 : rC * 3 + 3] = 1.0

    return legal


def _encoded_state(board: Board, turn: Piece) -> np.ndarray:
    state = np.empty(STATE_SHAPE, dtype=np.float32)
    _encode_board_into(state, board, turn)
    return state


def _priors_from_logits_np(logits: np.ndarray, legal: list[int]) -> list[float]:
    """Softmax over `logits[legal]`, returned as a list aligned to `legal`."""
    if not legal:
        return []
    selected = logits[legal] - logits[legal].max()
    exp = np.exp(selected, dtype=np.float32)
    total = float(exp.sum())
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / len(legal)] * len(legal)
    return (exp / total).tolist()


class _BatchBuf:
    """Reusable host (pinned on CUDA) + device tensors for batched eval.

    Eliminates per-leaf `np.empty` and per-step `np.stack`/`from_numpy` host
    allocations: the host tensor is filled in place, then async-copied to the
    pre-allocated device tensor.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        host_t = torch.empty((capacity, *STATE_SHAPE), dtype=torch.float32)
        if device.type == "cuda":
            host_t = host_t.pin_memory()
            self.device_t = torch.empty(
                (capacity, *STATE_SHAPE), dtype=torch.float32, device=device
            )
        else:
            self.device_t = host_t
        self.host_t = host_t
        self.host_view: np.ndarray = host_t.numpy()

    def device_input(self, n: int) -> torch.Tensor:
        if self.device.type == "cuda":
            self.device_t[:n].copy_(self.host_t[:n], non_blocking=True)
            return self.device_t[:n]
        return self.host_t[:n]


_USE_AMP = False


def _eval_states(
    net: AlphaZeroNet,
    buf: _BatchBuf,
    n: int,
    legal_actions: list[list[int]],
) -> tuple[list[list[float]], list[float]]:
    """Run `net` on the first `n` rows of `buf` (already filled by caller)."""
    input_t = buf.device_input(n)
    if _USE_AMP and buf.device.type == "cuda":
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, values = net(input_t)
        logits = logits.float()
        values = values.float()
    else:
        with torch.inference_mode():
            logits, values = net(input_t)
    logits_np = logits.cpu().numpy()
    values_np = values.cpu().numpy()
    priors = [_priors_from_logits_np(logits_np[k], legal_actions[k]) for k in range(n)]
    return priors, values_np.tolist()


def _eval_boards(
    net: AlphaZeroNet,
    buf: _BatchBuf,
    items: list[tuple[Board, Piece]],
) -> tuple[list[list[int]], list[list[float]], list[float]]:
    """Encode `items` into `buf`, run `net`. Returns (legal_per_item, priors_per_item, values)."""
    n = len(items)
    if n == 0:
        return [], [], []
    legal_actions: list[list[int]] = []
    host_view = buf.host_view
    for k, (board, turn) in enumerate(items):
        legal_actions.append(_encode_board_into(host_view[k], board, turn))
    priors, values = _eval_states(net, buf, n, legal_actions)
    return legal_actions, priors, values


def _add_dirichlet(node: _Node, eps: float, alpha: float = 0.3) -> None:
    if eps <= 0 or not node.legal:
        return
    P = node.priors
    noise = [random.gammavariate(alpha, 1.0) for _ in P]
    s = sum(noise) or 1.0
    for i in range(len(P)):
        P[i] = (1 - eps) * P[i] + eps * (noise[i] / s)


# ── Batched self-play ────────────────────────────────────────────────────────


def _batched_selfplay(
    net: AlphaZeroNet,
    device: torch.device,
    n_games: int,
    num_simulations: int,
    dirichlet_eps: float,
    temperature_moves: int,
    mcts_batch_size: int = 1,
    c_puct: float = 1.5,
    should_stop=None,
    on_progress=None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], list[BoardState]]:
    """Play n_games concurrently with batched GPU inference.

    Each MCTS simulation step gathers leaves across all active games into one
    forward pass. This is the key throughput win on a laptop GPU.
    """
    boards = [get_board() for _ in range(n_games)]
    turns = [Piece.X] * n_games
    move_idxs = [0] * n_games
    samples: list[list[tuple[np.ndarray, np.ndarray, Piece]]] = [
        [] for _ in range(n_games)
    ]
    roots: list[_Node | None] = [None] * n_games
    results: list[BoardState] = [BoardState.NOT_FINISHED] * n_games
    active = list(range(n_games))

    net.eval()
    leaf_batch_size = max(1, mcts_batch_size)
    buf = _BatchBuf(n_games * leaf_batch_size, device)

    while active:
        # ── 1. Make sure each active game has a root with priors ─────────
        need_root = [(i, boards[i], turns[i]) for i in active if roots[i] is None]
        if need_root:
            legal_lists, priors_lists, _ = _eval_boards(
                net, buf, [(b, t) for _, b, t in need_root]
            )
            for k, (i, _, t) in enumerate(need_root):
                node = _make_node(t, legal_lists[k], priors_lists[k])
                _add_dirichlet(node, dirichlet_eps)
                roots[i] = node

        # ── 2. Run MCTS in leaf batches across games ─────────────────────
        sim_counts = {i: 0 for i in active}
        while True:
            if all(sim_counts[i] >= num_simulations for i in active):
                break
            pending: list[
                tuple[list[tuple[_Node, int]], _Node, int, Piece, list[int]]
            ] = []
            pending_edges: set[tuple[int, int]] = set()
            n_pending = 0
            progressed = False
            host_view = buf.host_view
            for i in active:
                root = roots[i]
                if root is None or not root.legal:
                    sim_counts[i] = num_simulations
                    continue

                leaves = 0
                while sim_counts[i] < num_simulations and leaves < leaf_batch_size:
                    path: list[tuple[_Node, int]] = []
                    b = boards[i]
                    undo_tokens = []
                    node = root
                    t = turns[i]
                    counted = False
                    duplicate_pending = False

                    # PUCT descent until we hit an unexpanded edge or terminal.
                    try:
                        while True:
                            idx = _puct_select(node, c_puct)
                            a = node.legal[idx]
                            path.append((node, idx))
                            token = _apply_action(b, a, t)
                            if token is None:
                                counted = True
                                break
                            undo_tokens.append(token)
                            t = swap_piece(t)

                            child = node.children[idx]
                            if child is not None:
                                node = child
                                if node.is_terminal:
                                    _backup(path, node.terminal_value)
                                    counted = True
                                    break
                                if not node.legal:
                                    counted = True
                                    break
                                continue

                            # Unexpanded edge: leaf needs creating.
                            term = _terminal_value(b, t)
                            if term is not None:
                                node.children[idx] = _terminal_node(t, term)
                                _backup(path, term)
                                counted = True
                            else:
                                if leaf_batch_size > 1:
                                    edge = (id(node), idx)
                                    if edge in pending_edges:
                                        duplicate_pending = True
                                        break
                                    pending_edges.add(edge)
                                    _add_virtual_visit(path)
                                legal = _encode_board_into(host_view[n_pending], b, t)
                                pending.append((path, node, idx, t, legal))
                                n_pending += 1
                                counted = True
                            break
                    finally:
                        for token in reversed(undo_tokens):
                            _undo_action(b, token)

                    if counted:
                        sim_counts[i] += 1
                        leaves += 1
                        progressed = True
                    if duplicate_pending:
                        break

            # Single batched forward for all leaves needing eval
            if n_pending > 0:
                legal_actions = [p[4] for p in pending]
                priors_lists, values_list = _eval_states(
                    net, buf, n_pending, legal_actions
                )
                for k in range(n_pending):
                    path, parent, idx, t, legal = pending[k]
                    if leaf_batch_size > 1:
                        _revert_virtual_visit(path)
                    parent.children[idx] = _make_node(t, legal, priors_lists[k])
                    _backup(path, values_list[k])
            elif not progressed:
                break

        # ── 3. Pick a move for each game ─────────────────────────────────
        new_active = []
        for i in active:
            root = roots[i]
            total = sum(root.N) if root is not None else 0
            chosen_idx = -1

            if total == 0:
                legal_a = _legal_actions(boards[i], turns[i])
                if not legal_a:
                    results[i] = boards[i].board_state
                    continue
                action = random.choice(legal_a)
            else:
                pi_arr = np.zeros(NUM_ACTIONS, dtype=np.float32)
                inv_total = 1.0 / total
                for j, aa in enumerate(root.legal):
                    pi_arr[aa] = root.N[j] * inv_total
                samples[i].append(
                    (_encoded_state(boards[i], turns[i]), pi_arr, turns[i])
                )

                Ns = root.N
                if move_idxs[i] < temperature_moves:
                    r = random.random() * total
                    acc = 0
                    chosen_idx = len(Ns) - 1
                    for j in range(len(Ns)):
                        acc += Ns[j]
                        if r <= acc:
                            chosen_idx = j
                            break
                else:
                    best_j = 0
                    best_n = Ns[0]
                    for j in range(1, len(Ns)):
                        if Ns[j] > best_n:
                            best_n = Ns[j]
                            best_j = j
                    chosen_idx = best_j
                action = root.legal[chosen_idx]

            _apply_action(boards[i], action, turns[i])
            turns[i] = swap_piece(turns[i])
            move_idxs[i] += 1

            if boards[i].board_state != BoardState.NOT_FINISHED:
                results[i] = boards[i].board_state
            else:
                # Tree reuse: keep the chosen subtree as the new root
                if chosen_idx >= 0 and roots[i] is not None:
                    roots[i] = roots[i].children[chosen_idx]
                else:
                    roots[i] = None
                # Re-add Dirichlet to the new root for exploration
                if roots[i] is not None and not roots[i].is_terminal:
                    _add_dirichlet(roots[i], dirichlet_eps)
                new_active.append(i)
        active = new_active

        if on_progress is not None:
            on_progress(n_games - len(active), n_games)
        if should_stop is not None and should_stop():
            break

    # ── 4. Stitch z into samples ─────────────────────────────────────────
    out_samples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for i in range(n_games):
        st = results[i]
        if st == BoardState.X_WON:
            winner = Piece.X
        elif st == BoardState.O_WON:
            winner = Piece.O
        else:
            winner = None
        for state, pi, who in samples[i]:
            if winner is None:
                z = 0.0
            elif who == winner:
                z = 1.0
            else:
                z = -1.0
            out_samples.append((state, pi, z))
    return out_samples, results


def _split_counts(total: int, parts: int) -> list[int]:
    parts = max(1, min(parts, total))
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _raw_state_dict_cpu(net: nn.Module) -> dict[str, np.ndarray]:
    raw = net._orig_mod if hasattr(net, "_orig_mod") else net
    return {k: v.detach().cpu().numpy().copy() for k, v in raw.state_dict().items()}


def _worker_init():
    warnings.filterwarnings("ignore", message="Can't initialize NVML")


def _selfplay_worker(task):
    (
        state_dict,
        device_str,
        n_games,
        num_simulations,
        dirichlet_eps,
        temperature_moves,
        mcts_batch_size,
        c_puct,
        seed,
        net_blocks,
        net_channels,
        use_amp,
    ) = task
    global _USE_AMP
    _USE_AMP = bool(use_amp)
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    torch.manual_seed(seed)

    device = torch.device(device_str)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    net = AlphaZeroNet(num_blocks=net_blocks, channels=net_channels).to(device)
    net.load_state_dict({k: torch.from_numpy(v) for k, v in state_dict.items()})
    net.eval()
    return _batched_selfplay(
        net,
        device,
        n_games,
        num_simulations,
        dirichlet_eps,
        temperature_moves,
        mcts_batch_size=mcts_batch_size,
        c_puct=c_puct,
    )


def _resolve_workers(
    requested: int | None,
    device: torch.device,
    games_per_iter: int,
    num_simulations: int,
) -> int:
    if requested is not None:
        return max(0, min(requested, games_per_iter))

    # One CUDA process with a large lockstep batch is faster than several
    # processes for the default path. Users can still override this with
    # --workers when the run is CPU/MCTS-bound.
    if device.type == "cuda":
        return 0
    if games_per_iter < 16 or num_simulations < 16 or device.type == "mps":
        return 0

    cores = os.cpu_count() or 1
    return min(8, max(2, int(cores * 0.75)), games_per_iter)


def _generate_selfplay(
    net: AlphaZeroNet,
    device: torch.device,
    n_games: int,
    num_simulations: int,
    dirichlet_eps: float,
    temperature_moves: int,
    mcts_batch_size: int,
    n_workers: int,
    pool,
    c_puct: float = 1.5,
    should_stop=None,
    on_progress=None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], list[BoardState]]:
    if n_workers <= 1 or pool is None:
        return _batched_selfplay(
            net,
            device,
            n_games,
            num_simulations,
            dirichlet_eps,
            temperature_moves,
            mcts_batch_size=mcts_batch_size,
            c_puct=c_puct,
            should_stop=should_stop,
            on_progress=on_progress,
        )

    state_dict = _raw_state_dict_cpu(net)
    raw_net = net._orig_mod if hasattr(net, "_orig_mod") else net
    net_blocks = len(raw_net.res)
    net_channels = raw_net.input_conv[0].out_channels
    chunks = _split_counts(n_games, n_workers)
    tasks = [
        (
            state_dict,
            str(device),
            count,
            num_simulations,
            dirichlet_eps,
            temperature_moves,
            mcts_batch_size,
            c_puct,
            random.randrange(2**31 - 1),
            net_blocks,
            net_channels,
            _USE_AMP,
        )
        for count in chunks
        if count > 0
    ]

    all_samples: list[tuple[np.ndarray, np.ndarray, float]] = []
    all_results: list[BoardState] = []
    games_done = 0
    for samples, results in pool.imap_unordered(_selfplay_worker, tasks):
        all_samples.extend(samples)
        all_results.extend(results)
        games_done += len(results)
        if on_progress is not None:
            on_progress(games_done, n_games)
        if should_stop is not None and should_stop():
            break
    return all_samples, all_results


# ── Eval vs random ───────────────────────────────────────────────────────────


def _quick_eval(
    net: AlphaZeroNet,
    device: torch.device,
    n: int,
    sims: int,
    minimax_depth: int = 3,
) -> tuple[float, float, float]:
    """(win%, draw%, loss%) over n games against depth-limited minimax.

    AZ moves are batched across all games in lockstep; minimax moves are run
    sequentially per game (cheap at depth 3).
    """
    from player import MinimaxPlayer

    net.eval()
    boards = [get_board() for _ in range(n)]
    turns = [Piece.X] * n
    az_pieces = [Piece.X if g % 2 == 0 else Piece.O for g in range(n)]
    opp_x = MinimaxPlayer(Piece.X, depth_limit=minimax_depth)
    opp_o = MinimaxPlayer(Piece.O, depth_limit=minimax_depth)
    active = list(range(n))
    buf = _BatchBuf(n, device)

    while active:
        new_active = []
        # AZ side: batch one move per active game where it's az's turn
        az_need = [
            i
            for i in active
            if turns[i] == az_pieces[i]
            and boards[i].board_state == BoardState.NOT_FINISHED
        ]
        if az_need:
            _batched_selfplay_eval_step(net, buf, boards, turns, az_need, sims)

        # Minimax side: sequential
        for i in active:
            if boards[i].board_state != BoardState.NOT_FINISHED:
                continue
            if turns[i] != az_pieces[i]:
                opp = opp_x if turns[i] == Piece.X else opp_o
                move = opp.get_move(boards[i])
                if move is None:
                    continue
                boards[i].make_move(move)
                turns[i] = swap_piece(turns[i])
            if boards[i].board_state == BoardState.NOT_FINISHED:
                new_active.append(i)
        active = new_active

    wins = draws = 0
    for i in range(n):
        st = boards[i].board_state
        if (st == BoardState.X_WON and az_pieces[i] == Piece.X) or (
            st == BoardState.O_WON and az_pieces[i] == Piece.O
        ):
            wins += 1
        elif st == BoardState.DRAW:
            draws += 1
    losses = n - wins - draws
    return wins / n * 100, draws / n * 100, losses / n * 100


def _batched_selfplay_eval_step(
    net: AlphaZeroNet,
    buf: _BatchBuf,
    boards: list[Board],
    turns: list[Piece],
    indices: list[int],
    sims: int,
    c_puct: float = 1.5,
) -> None:
    """Run MCTS for a subset of games and apply one move per game in `indices`."""
    if not indices:
        return
    roots: dict[int, _Node] = {}

    # Init roots
    items = [(boards[i], turns[i]) for i in indices]
    legal_lists, priors_lists, _ = _eval_boards(net, buf, items)
    for k, i in enumerate(indices):
        roots[i] = _make_node(turns[i], legal_lists[k], priors_lists[k])

    for _ in range(sims):
        pending: list[tuple[list[tuple[_Node, int]], _Node, int, Piece, list[int]]] = []
        n_pending = 0
        host_view = buf.host_view
        for i in indices:
            root = roots[i]
            if not root.legal:
                continue
            path: list[tuple[_Node, int]] = []
            b = boards[i]
            undo_tokens = []
            node = root
            t = turns[i]
            try:
                while True:
                    idx = _puct_select(node, c_puct)
                    a = node.legal[idx]
                    path.append((node, idx))
                    token = _apply_action(b, a, t)
                    if token is None:
                        break
                    undo_tokens.append(token)
                    t = swap_piece(t)
                    child = node.children[idx]
                    if child is not None:
                        node = child
                        if node.is_terminal:
                            _backup(path, node.terminal_value)
                            break
                        if not node.legal:
                            break
                        continue
                    term = _terminal_value(b, t)
                    if term is not None:
                        node.children[idx] = _terminal_node(t, term)
                        _backup(path, term)
                    else:
                        legal = _encode_board_into(host_view[n_pending], b, t)
                        pending.append((path, node, idx, t, legal))
                        n_pending += 1
                    break
            finally:
                for token in reversed(undo_tokens):
                    _undo_action(b, token)

        if n_pending > 0:
            legal_actions = [p[4] for p in pending]
            priors_lists, values_list = _eval_states(net, buf, n_pending, legal_actions)
            for k in range(n_pending):
                path, parent, idx, t, legal = pending[k]
                parent.children[idx] = _make_node(t, legal, priors_lists[k])
                _backup(path, values_list[k])

    # Apply greedy move per game
    for i in indices:
        root = roots[i]
        if not root.N or sum(root.N) == 0:
            legal_a = _legal_actions(boards[i], turns[i])
            if not legal_a:
                continue
            action = random.choice(legal_a)
        else:
            best_j = 0
            best_n = root.N[0]
            for j in range(1, len(root.N)):
                if root.N[j] > best_n:
                    best_n = root.N[j]
                    best_j = j
            action = root.legal[best_j]
        _apply_action(boards[i], action, turns[i])
        turns[i] = swap_piece(turns[i])


# ── Training loop ────────────────────────────────────────────────────────────


class _ReplayBuffer:
    """Fixed-size ring buffer.

    CUDA runs keep the replay tensors on the GPU. The buffer is small enough for
    an 8 GB RTX 4060 at the default sizes, and GPU-side sampling avoids a
    host-to-device copy for every gradient step.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        augment_symmetries: int = 1,
    ) -> None:
        self.capacity = capacity
        self.device = device
        self.augment_symmetries = max(1, min(8, augment_symmetries))
        self.on_device = device.type == "cuda"
        if self.on_device:
            self.states = torch.empty(
                (capacity, *STATE_SHAPE), dtype=torch.float32, device=device
            )
            self.policies = torch.empty(
                (capacity, NUM_ACTIONS), dtype=torch.float32, device=device
            )
            self.values = torch.empty(capacity, dtype=torch.float32, device=device)
        else:
            self.states = np.empty((capacity, *STATE_SHAPE), dtype=np.float32)
            self.policies = np.empty((capacity, NUM_ACTIONS), dtype=np.float32)
            self.values = np.empty(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def extend(self, samples: list[tuple[np.ndarray, np.ndarray, float]]) -> int:
        n = len(samples)
        if n == 0:
            return 0
        sym_count = self.augment_symmetries
        max_raw = max(1, math.ceil(self.capacity / sym_count))
        if n > max_raw:
            samples = samples[-max_raw:]
            n = len(samples)

        n_aug = n * sym_count
        states = np.empty((n_aug, *STATE_SHAPE), dtype=np.float32)
        policies = np.empty((n_aug, NUM_ACTIONS), dtype=np.float32)
        values = np.empty(n_aug, dtype=np.float32)

        out_idx = 0
        for state, pi, z in samples:
            for sym in range(sym_count):
                states[out_idx] = _transform_state(state, sym)
                if sym == 0:
                    policies[out_idx] = pi
                else:
                    policies[out_idx, _ACTION_SYM_PERMS[sym]] = pi
                values[out_idx] = z
                out_idx += 1

        if n_aug > self.capacity:
            states = states[-self.capacity :]
            policies = policies[-self.capacity :]
            values = values[-self.capacity :]
            n_aug = self.capacity

        if self.on_device:
            states_t = torch.from_numpy(states).to(self.device, non_blocking=True)
            policies_t = torch.from_numpy(policies).to(self.device, non_blocking=True)
            values_t = torch.from_numpy(values).to(self.device, non_blocking=True)
        else:
            states_t = states
            policies_t = policies
            values_t = values

        first = min(n_aug, self.capacity - self.pos)
        self.states[self.pos : self.pos + first] = states_t[:first]
        self.policies[self.pos : self.pos + first] = policies_t[:first]
        self.values[self.pos : self.pos + first] = values_t[:first]

        remaining = n_aug - first
        if remaining:
            self.states[:remaining] = states_t[first:]
            self.policies[:remaining] = policies_t[first:]
            self.values[:remaining] = values_t[first:]

        self.pos = (self.pos + n_aug) % self.capacity
        self.size = min(self.capacity, self.size + n_aug)
        return n_aug

    def sample_tensors(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.on_device:
            idx = torch.randint(self.size, (batch_size,), device=self.device)
            return (
                self.states.index_select(0, idx),
                self.policies.index_select(0, idx),
                self.values.index_select(0, idx),
            )

        idx = np.random.randint(self.size, size=batch_size)
        states = torch.from_numpy(self.states[idx]).to(device, non_blocking=False)
        policies = torch.from_numpy(self.policies[idx]).to(device, non_blocking=False)
        values = torch.from_numpy(self.values[idx]).to(device, non_blocking=False)
        return states, policies, values


def _save_checkpoint(path, net, optimizer, iteration):
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        path,
    )


def _finite_float(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _metric_points(
    rows: list[dict[str, int | float | None]],
    key: str,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        x = _finite_float(row.get("iteration"))
        y = _finite_float(row.get(key))
        if x is not None and y is not None:
            points.append((x, y))
    return points


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _learning_summary(rows: list[dict[str, int | float | None]]) -> list[str]:
    if not rows:
        return ["No completed iterations to analyze yet."]

    out: list[str] = []
    eval_pts = _metric_points(rows, "eval_win_rate")
    if len(eval_pts) >= 2:
        delta = eval_pts[-1][1] - eval_pts[-2][1]
        if delta > 2.0:
            out.append(f"Eval is still improving: last check +{delta:.1f} percentage points.")
        elif delta < -2.0:
            out.append(f"Eval dropped on the last check: {delta:.1f} percentage points.")
        else:
            out.append("Eval is roughly flat over the last two checks.")
    elif len(eval_pts) == 1:
        out.append("Only one eval point exists so far; trend needs another eval.")
    else:
        out.append("No eval points yet; enable evals or wait for the first eval interval.")

    policy = [p[1] for p in _metric_points(rows, "policy_loss")]
    if len(policy) >= 6:
        n = max(3, min(20, len(policy) // 5))
        prev = _mean(policy[-2 * n : -n])
        recent = _mean(policy[-n:])
        pct = (recent - prev) / max(abs(prev), 1e-9) * 100.0
        if pct < -2.0:
            out.append(f"Policy loss is still fitting better: recent average {abs(pct):.1f}% lower.")
        elif pct > 2.0:
            out.append(f"Policy loss is rising: recent average {pct:.1f}% higher.")
        else:
            out.append("Policy loss is mostly flat recently.")
    else:
        out.append("Loss trend needs at least a few trained iterations.")

    train_loss = [p[1] for p in _metric_points(rows, "loss")]
    check_loss = [p[1] for p in _metric_points(rows, "check_loss")]
    if len(train_loss) >= 4 and len(check_loss) >= 4:
        n = max(2, min(10, min(len(train_loss), len(check_loss)) // 4))
        train_recent = _mean(train_loss[-n:])
        check_recent = _mean(check_loss[-n:])
        gap_pct = (check_recent - train_recent) / max(abs(train_recent), 1e-9) * 100.0
        if gap_pct > 15.0:
            out.append(f"Overfit risk: replay check loss is {gap_pct:.1f}% above train loss.")
        elif train_recent > train_loss[0] * 0.95 and check_recent > check_loss[0] * 0.95:
            out.append("Underfit/not-learning signal: train and check losses are not dropping much.")
        else:
            out.append("Train/check loss gap looks reasonable.")

    return out[:3]


def _write_training_plot(
    path: str,
    rows: list[dict[str, int | float | None]],
    title: str,
    show: bool = True,
) -> None:
    if not rows:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(yellow(f"  Plot skipped: matplotlib is not installed ({exc})."))
        return

    def _rolling_mean(ys: list[float], window: int) -> list[float]:
        if window <= 1 or len(ys) < 2:
            return ys
        out: list[float] = []
        acc = 0.0
        from collections import deque
        q: deque[float] = deque()
        for y in ys:
            q.append(y)
            acc += y
            if len(q) > window:
                acc -= q.popleft()
            out.append(acc / len(q))
        return out

    def plot_series(
        ax,
        key: str,
        label: str,
        color: str,
        *,
        marker: str = ".",
        smooth: int = 1,
    ) -> bool:
        pts = _metric_points(rows, key)
        if not pts:
            return False
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if smooth > 1 and len(ys) >= smooth:
            ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.25)
            ys = _rolling_mean(ys, smooth)
        ax.plot(xs, ys, label=label, color=color, linewidth=1.8, marker=marker, markersize=4)
        return True

    outcome_window = max(1, min(40, len(rows) // 25))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=140)
    ax_loss, ax_eval, ax_speed, ax_replay = axes.ravel()
    fig.suptitle(f"{title} training metrics", fontsize=16, fontweight="bold")

    plot_series(ax_loss, "loss", "train total", "#344054")
    plot_series(ax_loss, "check_loss", "replay check", "#d92d20", marker="o")
    plot_series(ax_loss, "policy_loss", "policy", "#2563eb")
    plot_series(ax_loss, "value_loss", "value", "#dc6803")
    ax_loss.set_title("Train vs replay-check loss")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("loss")
    ax_loss.set_ylim(bottom=0)

    plot_series(ax_eval, "eval_win_rate", "eval win", "#7c3aed", marker="o")
    plot_series(ax_eval, "x_win_rate", "self-play X wins", "#039855", smooth=outcome_window)
    plot_series(ax_eval, "draw_rate", "self-play draws", "#ca8a04", smooth=outcome_window)
    plot_series(ax_eval, "o_win_rate", "self-play O wins", "#d92d20", smooth=outcome_window)
    ax_eval.set_title(
        "Eval and self-play outcomes"
        + (f"  (outcomes smoothed, window={outcome_window})" if outcome_window > 1 else "")
    )
    ax_eval.set_xlabel("iteration")
    ax_eval.set_ylabel("percent")
    ax_eval.set_ylim(0, 100)

    plot_series(ax_speed, "positions_per_sec", "raw positions/sec", "#0e9384")
    ax_speed.set_title("Self-play throughput")
    ax_speed.set_xlabel("iteration")
    ax_speed.set_ylabel("positions/sec")
    ax_speed.set_ylim(bottom=0)

    plot_series(ax_replay, "replay_size", "replay size", "#475467")
    plot_series(ax_replay, "aug_samples", "augmented samples/it", "#dd2590")
    ax_replay.set_title("Data volume")
    ax_replay.set_xlabel("iteration")
    ax_replay.set_ylabel("positions")
    ax_replay.set_ylim(bottom=0)

    for ax in axes.ravel():
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=9)
        ax.tick_params(axis="both", labelsize=9)

    summary = "\n".join(_learning_summary(rows))
    fig.text(
        0.01,
        0.01,
        f"{summary}\nRead eval trend first; self-play loss can flatten or rise when generated positions get harder.",
        fontsize=9,
        color="#344054",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))

    tmp_path = f"{path}.tmp.png"
    fig.savefig(tmp_path, bbox_inches="tight")
    os.replace(tmp_path, path)
    if show:
        try:
            plt.show()
        except Exception as exc:
            print(yellow(f"  Plot display skipped: {exc}"))
    plt.close(fig)


def train(
    iterations: int,
    games_per_iter: int,
    num_simulations: int,
    train_steps: int,
    batch_size: int,
    buffer_size: int,
    lr: float,
    dirichlet_eps: float,
    temperature_moves: int,
    augment_symmetries: int,
    mcts_batch_size: int,
    name: str,
    load_name: str | None,
    device_str: str | None,
    eval_interval: int,
    eval_games: int,
    eval_sims: int,
    workers: int | None,
    use_compile: bool,
    net_channels: int = 32,
    net_blocks: int = 3,
    use_amp: bool = False,
) -> None:
    global _USE_AMP
    os.makedirs(MODELS_DIR, exist_ok=True)
    _USE_AMP = bool(use_amp)
    save_path = os.path.join(MODELS_DIR, f"{name}.pt")
    ckpt_path = os.path.join(MODELS_DIR, f"{name}.ckpt.pt")
    plot_path = os.path.join(MODELS_DIR, f"{name}.training.png")

    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    cpu_threads, interop_threads = _configure_torch_runtime(device)

    net = AlphaZeroNet(num_blocks=net_blocks, channels=net_channels).to(device)
    optimizer_kwargs = {"lr": lr, "weight_decay": 1e-4}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = optim.Adam(net.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = optim.Adam(net.parameters(), **optimizer_kwargs)
    replay = _ReplayBuffer(buffer_size, device, augment_symmetries)
    start_iter = 0

    if load_name:
        ckpt_load = os.path.join(MODELS_DIR, f"{load_name}.ckpt.pt")
        weights_load = os.path.join(MODELS_DIR, f"{load_name}.pt")
        if os.path.exists(ckpt_load):
            ck = torch.load(ckpt_load, map_location=device, weights_only=False)
            net.load_state_dict(_clean_state_dict(ck["model"]))
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            start_iter = ck.get("iteration", 0)
            print(f"  {cyan('↺')} Resumed checkpoint ep {start_iter:,}")
        elif os.path.exists(weights_load):
            net.load_state_dict(
                _clean_state_dict(
                    torch.load(weights_load, map_location=device, weights_only=True)
                )
            )
            print(f"  {cyan('↺')} Loaded weights from {weights_load}")

    metrics: list[dict[str, int | float | None]] = []

    if use_compile and hasattr(torch, "compile"):
        net = torch.compile(net, dynamic=True)

    n_workers = _resolve_workers(workers, device, games_per_iter, num_simulations)
    pool = None
    if n_workers > 1:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(n_workers, initializer=_worker_init)

    def _shutdown_pool(terminate: bool = False) -> None:
        nonlocal pool
        if pool is None:
            return
        if terminate:
            pool.terminate()
        else:
            pool.close()
        pool.join()
        pool = None

    print(
        _box(
            "AlphaZero Training",
            "Ultimate Tic-Tac-Toe self-play + policy/value optimization",
        )
    )
    print()
    dev_disp = str(device)
    if device.type == "cuda":
        dev_disp += f"  ({torch.cuda.get_device_name(0)})"
    if n_workers <= 1:
        play_backend = "single-process · batched"
    else:
        play_backend = f"{n_workers} workers · per-worker batched"
    total_planned_games = max(0, iterations - start_iter) * games_per_iter
    print(_phase("Run Setup"))
    print(_kv("Device", dev_disp))
    print(
        _kv(
            "CPU",
            f"{os.cpu_count() or 1} logical",
            f"torch {cpu_threads} compute / {interop_threads} interop threads",
        )
    )
    print(_kv("Iterations", f"{iterations:,}", f"starting at {start_iter:,}"))
    print(
        _kv(
            "Self-play",
            green(play_backend),
            f"{games_per_iter:,} games/it · {num_simulations:,} sims/move",
        )
    )
    print(
        _kv(
            "Training",
            f"{train_steps:,} steps/it",
            f"batch {batch_size:,} · replay {buffer_size:,} ({'GPU' if replay.on_device else 'CPU'} buffer)",
        )
    )
    print(_kv("Augmentation", f"{replay.augment_symmetries}x board symmetries"))
    n_params = sum(p.numel() for p in net.parameters())
    print(
        _kv(
            "Network",
            f"{net_blocks} blocks · {net_channels} ch",
            f"{n_params:,} params",
        )
    )
    if mcts_batch_size > 1:
        print(_kv("MCTS batching", f"{mcts_batch_size} virtual leaves/game"))
    print(
        _kv(
            "Learning",
            f"lr {lr:g}",
            f"Dirichlet eps {dirichlet_eps:g} · temp moves {temperature_moves}",
        )
    )
    if eval_interval:
        print(
            _kv(
                "Eval",
                f"every {eval_interval} it",
                f"{eval_games:,} games · {eval_sims:,} sims",
            )
        )
    else:
        print(_kv("Eval", dim("off")))
    compile_str = green("on") if use_compile else dim("off")
    tf32_str = green("on") if device.type == "cuda" else dim("n/a")
    print(
        _kv(
            "Performance",
            f"compile {compile_str} · TF32 {tf32_str}",
            f"{total_planned_games:,} planned games",
        )
    )
    print(_kv("Weights", save_path))
    print(_kv("Checkpoint", ckpt_path))
    print(_kv("Plot", plot_path))
    print()
    print(_rule())
    warmup_msg = (
        "warming up CUDA / cudnn" if device.type == "cuda" else "warming up model"
    )
    print(f"  {dim(f'{warmup_msg} (first iteration is slowest)...')}", flush=True)

    interrupted = False

    def _sigint(sig, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _sigint)

    amp_on = use_amp and device.type == "cuda"

    def _train_step():
        states, target_pi, target_v = replay.sample_tensors(batch_size, device)

        net.train()
        if amp_on:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, value = net(states)
                log_probs = F.log_softmax(logits, dim=1)
                policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(value, target_v)
                loss = policy_loss + value_loss
        else:
            logits, value = net(states)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value, target_v)
            loss = policy_loss + value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        return float(loss.item()), float(policy_loss.item()), float(value_loss.item())

    def _check_loss_step():
        states, target_pi, target_v = replay.sample_tensors(batch_size, device)

        net.eval()
        with torch.inference_mode():
            logits, value = net(states)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value, target_v)
            loss = policy_loss + value_loss
        return float(loss.item()), float(policy_loss.item()), float(value_loss.item())

    last_eval_str = ""
    t0 = time.monotonic()

    for it in range(start_iter + 1, iterations + 1):
        # ── Self-play (batched) ──
        sp_t0 = time.monotonic()
        last_print = [0.0]
        last_logged_done = [-1]
        rendered = [False]
        log_stride = max(1, games_per_iter // 10)

        def _render_two(line1: str, line2: str) -> None:
            """Draw two stacked progress lines in place."""
            if _IS_TTY:
                if rendered[0]:
                    sys.stdout.write("\033[1A\r")  # back up to line 1
                sys.stdout.write(f"\033[2K{line1}\n\033[2K{line2}\r")
                sys.stdout.flush()
                rendered[0] = True
            else:
                print(line2, flush=True)

        def _clear_two() -> None:
            if _IS_TTY and rendered[0]:
                sys.stdout.write("\033[1A\r\033[2K\n\033[2K\033[1A\r")
                sys.stdout.flush()
            rendered[0] = False

        def _progress(done: int, total: int) -> None:
            now = time.monotonic()
            if _IS_TTY:
                if now - last_print[0] < 0.25 and done < total:
                    return
            else:
                if done == last_logged_done[0]:
                    return
                if done not in (0, total) and done - last_logged_done[0] < log_stride:
                    return
                last_logged_done[0] = done
            last_print[0] = now
            elapsed_sp = max(now - sp_t0, 1e-6)
            frac_g = done / total if total else 1.0
            sp_eta = ((total - done) / done * elapsed_sp) if done > 0 else 0
            games_per_s = done / elapsed_sp

            it_total = iterations - start_iter
            it_done_frac = (it - 1 - start_iter) + frac_g
            overall_elapsed = now - t0
            overall_eta = (
                (it_total - it_done_frac) / max(it_done_frac, 1e-6) * overall_elapsed
                if it_done_frac > 0
                else 0
            )
            frac_it = (it - 1 + frac_g) / iterations

            buf_str = f"buf {len(replay):>6,}"
            line1 = (
                f"  {bold('Overall')}    {_bar(frac_it, 30)} "
                f"{bold(f'{frac_it*100:5.1f}%')}  "
                f"{dim(f'it {it:>{len(str(iterations))}}/{iterations}')}  "
                f"{cyan('ETA')} {_fmt_time(overall_eta):>10}  "
                f"{dim(buf_str)}"
            )
            line2 = (
                f"  {dim('Self-play')}  {_bar(frac_g, 30)} "
                f"{bold(f'{frac_g*100:5.1f}%')}  "
                f"{dim(f'{done:>{len(str(total))}}/{total} games')}  "
                f"{cyan('eta')} {_fmt_time(sp_eta):>10}  "
                f"{dim(f'{games_per_s:.1f} g/s · {_fmt_time(elapsed_sp)}')}"
            )
            _render_two(line1, line2)

        # Initial paint so user immediately sees the bars
        _progress(0, games_per_iter)

        samples, results = _generate_selfplay(
            net,
            device,
            games_per_iter,
            num_simulations,
            dirichlet_eps,
            temperature_moves,
            mcts_batch_size,
            n_workers,
            pool,
            should_stop=lambda: interrupted,
            on_progress=_progress,
        )
        _clear_two()
        added_samples = replay.extend(samples)
        sp_dt = time.monotonic() - sp_t0

        if interrupted:
            print(yellow(f"\n  Interrupted during self-play of iteration {it}."))
            raw = net._orig_mod if hasattr(net, "_orig_mod") else net
            torch.save(raw.state_dict(), save_path)
            _save_checkpoint(ckpt_path, raw, optimizer, it - 1)
            if metrics:
                _write_training_plot(plot_path, metrics, name)
            print(f"  {cyan('✔')} Weights    : {bold(save_path)}")
            print(
                f"  {cyan('✔')} Checkpoint : {bold(ckpt_path)}  (resume at it {it - 1})"
            )
            if metrics:
                print(f"  {cyan('✔')} Plot       : {bold(plot_path)}")
            print(f"\n  Resume with: {bold(f'--load {name}')}")
            _shutdown_pool(terminate=True)
            return

        wins_x = sum(1 for r in results if r == BoardState.X_WON)
        wins_o = sum(1 for r in results if r == BoardState.O_WON)
        draws = sum(1 for r in results if r == BoardState.DRAW)
        total_games = max(1, wins_x + wins_o + draws)

        # ── Train ──
        tr_t0 = time.monotonic()
        loss_sum = pl_sum = vl_sum = 0.0
        steps_done = 0
        if len(replay) >= batch_size:
            for _ in range(train_steps):
                l, pl, vl = _train_step()
                loss_sum += l
                pl_sum += pl
                vl_sum += vl
                steps_done += 1
                if interrupted:
                    break
        tr_dt = time.monotonic() - tr_t0
        avg_loss = loss_sum / max(1, steps_done)
        avg_pl = pl_sum / max(1, steps_done)
        avg_vl = vl_sum / max(1, steps_done)
        check_loss = check_pl = check_vl = None
        if len(replay) >= batch_size:
            check_loss, check_pl, check_vl = _check_loss_step()

        # ── Report ──
        elapsed = time.monotonic() - t0
        frac = it / iterations
        done = it - start_iter
        it_per_sec = done / elapsed if elapsed > 0 else 0
        eta = (iterations - it) / it_per_sec if it_per_sec > 0 else 0
        moves_per_sec = len(samples) / sp_dt if sp_dt > 0 else 0

        # Win-rate mini-bar (X | D | O)
        def _wr_bar(width: int = 24) -> str:
            x = round(width * wins_x / total_games)
            o = round(width * wins_o / total_games)
            d = max(0, width - x - o)
            return green("█" * x) + yellow("█" * d) + red("█" * o)

        head = (
            f"  {cyan('▸')} {bold(f'Iteration {it:>{len(str(iterations))}}/{iterations}')}  "
            f"{_bar(frac, 24)} {bold(f'{frac*100:5.1f}%')}  "
            f"{dim('ETA')} {bold(_fmt_time(eta))}"
        )
        body_games = (
            f"    {dim('outcomes')} {_wr_bar(24)}  "
            f"{green(f'X {wins_x/total_games*100:4.1f}%')} "
            f"{yellow(f'D {draws/total_games*100:4.1f}%')} "
            f"{red(f'O {wins_o/total_games*100:4.1f}%')}"
        )
        body_perf = (
            f"    {_metric('self-play', _fmt_time(sp_dt), f'{moves_per_sec:,.0f} pos/s')}   "
            f"{_metric('training', _fmt_time(tr_dt), f'{steps_done:,} steps')}   "
            f"{_metric('samples', f'{len(samples):,}', f'aug {added_samples:,}' if added_samples != len(samples) else '')}   "
            f"{_metric('replay', f'{len(replay):,}/{buffer_size:,}')}"
        )
        loss_note = f"policy {avg_pl:.3f} · value {avg_vl:.3f}"
        if check_loss is not None:
            loss_note += f" · check {check_loss:.3f}"
        body_loss = f"    {_metric('loss', f'{avg_loss:.3f}', loss_note)}"

        is_eval_iter = bool(eval_interval) and it % eval_interval == 0

        print()
        print(head, flush=True)
        print(body_perf, flush=True)
        print(body_games, flush=True)
        print(body_loss, flush=True)
        # Carry the previous eval's result forward, but skip it on iterations
        # that are about to print a fresh result so we never show two eval lines.
        if last_eval_str and not is_eval_iter:
            print(f"    {last_eval_str}", flush=True)

        # ── Periodic eval ──
        eval_wr = None
        eval_dr = None
        eval_lr = None
        eval_dt = 0.0
        if is_eval_iter:
            ev_t0 = time.monotonic()
            wr, dr, lr = _quick_eval(net, device, n=eval_games, sims=eval_sims)
            eval_dt = time.monotonic() - ev_t0
            eval_wr, eval_dr, eval_lr = wr, dr, lr
            last_eval_str = (
                f"{magenta('►')} eval vs minimax d3 "
                f"{green(bold(f'W {wr:4.1f}%'))} "
                f"{yellow(bold(f'D {dr:4.1f}%'))} "
                f"{red(bold(f'L {lr:4.1f}%'))} "
                f"{dim(f'({eval_games} games · {eval_sims} sims)')}"
            )
            print(f"    {last_eval_str}", flush=True)

        metrics.append(
            {
                "iteration": it,
                "elapsed_sec": elapsed,
                "iteration_sec": sp_dt + tr_dt + eval_dt,
                "selfplay_sec": sp_dt,
                "train_sec": tr_dt,
                "eval_sec": eval_dt if eval_wr is not None else None,
                "raw_samples": len(samples),
                "aug_samples": added_samples,
                "positions_per_sec": moves_per_sec,
                "replay_size": len(replay),
                "train_steps": steps_done,
                "loss": avg_loss if steps_done else None,
                "policy_loss": avg_pl if steps_done else None,
                "value_loss": avg_vl if steps_done else None,
                "check_loss": check_loss,
                "check_policy_loss": check_pl,
                "check_value_loss": check_vl,
                "x_win_rate": wins_x / total_games * 100.0,
                "draw_rate": draws / total_games * 100.0,
                "o_win_rate": wins_o / total_games * 100.0,
                "eval_win_rate": eval_wr,
                "eval_draw_rate": eval_dr,
                "eval_loss_rate": eval_lr,
            }
        )

        # Save (weights + checkpoint)
        raw = net._orig_mod if hasattr(net, "_orig_mod") else net
        torch.save(raw.state_dict(), save_path)
        _save_checkpoint(ckpt_path, raw, optimizer, it)

        if interrupted:
            _write_training_plot(plot_path, metrics, name)
            print(yellow(f"\n  Interrupted at iteration {it}. Saved."))
            print(f"  {cyan('✔')} Weights    : {bold(save_path)}")
            print(f"  {cyan('✔')} Checkpoint : {bold(ckpt_path)}")
            print(f"  {cyan('✔')} Plot       : {bold(plot_path)}")
            print(f"\n  Resume with: {bold(f'--load {name}')}")
            _shutdown_pool(terminate=True)
            return

    _shutdown_pool()

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    if metrics:
        _write_training_plot(plot_path, metrics, name)

    elapsed = time.monotonic() - t0
    print(f"\n{_rule()}")
    print(_box("Training Complete", f"{iterations - start_iter:,} iterations finished"))
    print()
    rate = (iterations - start_iter) / elapsed if elapsed > 0 else 0
    print(_kv("Time", _fmt_time(elapsed), f"{rate:.2f} it/s"))
    print(_kv("Replay", f"{len(replay):,} positions"))
    print(_kv("Saved", green(save_path)))
    if metrics:
        print(_kv("Plot", plot_path))
        summary = _learning_summary(metrics)
        if summary:
            print(_kv("Signal", summary[0]))
            for line in summary[1:]:
                print(f"  {'':18} {line}")
    print()


# ── Eval CLI ─────────────────────────────────────────────────────────────────


def evaluate(model_name: str, episodes: int, num_simulations: int) -> None:
    path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    if not os.path.exists(path):
        print(red(f"  No model at {path}"))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_torch_runtime(device)

    net = AlphaZeroNet().to(device)
    net.load_state_dict(
        _clean_state_dict(torch.load(path, map_location=device, weights_only=True))
    )
    net.eval()

    params = sum(p.numel() for p in net.parameters())
    print(_box("AlphaZero Eval  —  vs Minimax (depth 3)"))
    print()
    print(f"  {bold('Model')}    : {path}  {dim(f'({params:,} params)')}")
    print(f"  {bold('Episodes')} : {episodes:,}   {bold('Sims')}: {num_simulations}")
    print()

    t0 = time.monotonic()
    wr, dr, lr = _quick_eval(net, device, episodes, num_simulations)
    elapsed = time.monotonic() - t0
    print(dim("  " + "─" * 52))
    print(f"  {green(bold('Wins  '))}  {_pct_bar(wr)}  {bold(f'{wr:5.1f}%')}")
    print(f"  {yellow(bold('Draws '))}  {_pct_bar(dr)}  {bold(f'{dr:5.1f}%')}")
    print(f"  {red(bold('Losses'))}  {_pct_bar(lr)}  {bold(f'{lr:5.1f}%')}")
    print(dim("  " + "─" * 52))
    print(f"\n  {dim(_fmt_time(elapsed))}\n")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlphaZero for Ultimate TTT")
    sub = parser.add_subparsers(dest="command")

    tp = sub.add_parser("train")
    tp.add_argument(
        "--profile",
        choices=tuple(TRAINING_PROFILES),
        default="fast",
        help="Hardware-aware default set; explicit flags override it",
    )
    tp.add_argument("--iterations", type=int, default=None)
    tp.add_argument("--games-per-iter", type=int, default=None)
    tp.add_argument("--simulations", type=int, default=None)
    tp.add_argument("--train-steps", type=int, default=None)
    tp.add_argument("--batch-size", type=int, default=None)
    tp.add_argument("--buffer-size", type=int, default=None)
    tp.add_argument("--lr", type=float, default=1e-3)
    tp.add_argument("--dirichlet-eps", type=float, default=0.25)
    tp.add_argument("--temperature-moves", type=int, default=15)
    tp.add_argument(
        "--augment-symmetries",
        type=int,
        choices=(1, 2, 4, 8),
        default=None,
        help="Number of board symmetries inserted per self-play sample",
    )
    tp.add_argument(
        "--mcts-batch-size",
        type=int,
        default=None,
        help="Virtual leaves gathered per game before each network eval",
    )
    tp.add_argument("--net-channels", type=int, default=None)
    tp.add_argument("--net-blocks", type=int, default=None)
    tp.add_argument("--name", type=str, default=None)
    tp.add_argument("--load", type=str, default=None)
    tp.add_argument("--device", type=str, default=None)
    tp.add_argument("--eval-interval", type=int, default=None)
    tp.add_argument("--eval-games", type=int, default=None)
    tp.add_argument("--eval-sims", type=int, default=None)
    tp.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Self-play worker processes (auto-detected by default, 0/1 = single-process batched)",
    )
    tp.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    tp.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)

    ep = sub.add_parser("eval")
    ep.add_argument("--model", type=str, required=True)
    ep.add_argument("--episodes", type=int, default=50)
    ep.add_argument("--simulations", type=int, default=200)

    args = parser.parse_args()

    if args.command == "train":
        iterations = int(_profile_value(args, args.profile, "iterations"))
        games_per_iter = int(_profile_value(args, args.profile, "games_per_iter"))
        num_simulations = int(_profile_value(args, args.profile, "simulations"))
        train_steps = int(_profile_value(args, args.profile, "train_steps"))
        batch_size = int(_profile_value(args, args.profile, "batch_size"))
        buffer_size = int(_profile_value(args, args.profile, "buffer_size"))
        augment_symmetries = int(
            _profile_value(args, args.profile, "augment_symmetries")
        )
        mcts_batch_size = int(_profile_value(args, args.profile, "mcts_batch_size"))
        eval_interval = int(_profile_value(args, args.profile, "eval_interval"))
        eval_games = int(_profile_value(args, args.profile, "eval_games"))
        eval_sims = int(_profile_value(args, args.profile, "eval_sims"))
        workers = (
            args.workers
            if args.workers is not None
            else int(_profile_value(args, args.profile, "workers"))
        )
        net_channels = int(_profile_value(args, args.profile, "net_channels"))
        net_blocks = int(_profile_value(args, args.profile, "net_blocks"))
        use_amp = bool(_profile_value(args, args.profile, "amp"))
        name = args.name or _default_name(iterations)
        train(
            iterations=iterations,
            games_per_iter=games_per_iter,
            num_simulations=num_simulations,
            train_steps=train_steps,
            batch_size=batch_size,
            buffer_size=buffer_size,
            lr=args.lr,
            dirichlet_eps=args.dirichlet_eps,
            temperature_moves=args.temperature_moves,
            augment_symmetries=augment_symmetries,
            mcts_batch_size=mcts_batch_size,
            name=name,
            load_name=args.load,
            device_str=args.device,
            eval_interval=eval_interval,
            eval_games=eval_games,
            eval_sims=eval_sims,
            workers=workers,
            use_compile=args.compile,
            net_channels=net_channels,
            net_blocks=net_blocks,
            use_amp=use_amp,
        )
    elif args.command == "eval":
        evaluate(args.model, args.episodes, args.simulations)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
