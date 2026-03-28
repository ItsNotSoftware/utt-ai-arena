#!/usr/bin/env python3
"""Headless Q-Learning training via self-play for Ultimate Tic-Tac-Toe."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool, cpu_count

# Ensure src/ is importable when run from the project root
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from board import BoardState, Piece, get_board, swap_piece
from player import QLearningPlayer

MODELS_DIR = "models/qlearning"


def _default_name(episodes: int) -> str:
    if episodes >= 1_000_000:
        n = episodes // 1_000_000
        r = (episodes % 1_000_000) // 100_000
        return f"q_table_{n}M" if r == 0 else f"q_table_{n}.{r}M"
    if episodes >= 1_000:
        n = episodes // 1_000
        return f"q_table_{n}k"
    return f"q_table_{episodes}"


# ── ANSI helpers ─────────────────────────────────────────────────────────────

_IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def cyan(t: str) -> str:
    return _c("96", t)


def green(t: str) -> str:
    return _c("92", t)


def red(t: str) -> str:
    return _c("91", t)


def yellow(t: str) -> str:
    return _c("93", t)


def blue(t: str) -> str:
    return _c("94", t)


def _bar(frac: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    n = round(frac * width)
    return green(fill * n) + dim(empty * (width - n))


def _pct_bar(pct: float, width: int = 28) -> str:
    n = round(pct / 100 * width)
    return green("█" * n) + dim("░" * (width - n))


def _box(title: str, width: int = 58) -> str:
    inner = f"  {title}  "
    pad = max(0, width - len(inner))
    line = "═" * (width + 2)
    return (
        f"\n{cyan(bold('╔' + line + '╗'))}\n"
        f"{cyan(bold('║'))} {bold(inner)}{' ' * pad}{cyan(bold('║'))}\n"
        f"{cyan(bold('╚' + line + '╝'))}"
    )


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


# ── Worker ────────────────────────────────────────────────────────────────────


def _run_episodes(args: tuple) -> dict[int, float]:
    """Worker: run a block of episodes and return the resulting Q-table."""
    (
        episodes,
        alpha,
        gamma,
        epsilon_start,
        epsilon_end,
        epsilon_decay_episodes,
        report_interval,
        worker_id,
        workers,
        max_entries,
        initial_q_table,
    ) = args

    import random as _random

    _random.seed(worker_id)

    q_table: dict[int, float] = dict(initial_q_table) if initial_q_table else {}
    p_x = QLearningPlayer(
        Piece.X, q_table=q_table, alpha=alpha, gamma=gamma, training=True
    )
    p_o = QLearningPlayer(
        Piece.O, q_table=q_table, alpha=alpha, gamma=gamma, training=True
    )
    players = {Piece.X: p_x, Piece.O: p_o}

    wins_x = wins_o = draws = 0
    t0 = time.monotonic()

    for ep in range(1, episodes + 1):
        frac = min(ep / epsilon_decay_episodes, 1.0)
        eps = epsilon_start + (epsilon_end - epsilon_start) * frac
        p_x.epsilon = eps
        p_o.epsilon = eps

        p_x.reset_episode()
        p_o.reset_episode()

        board = get_board()
        turn = Piece.X

        while board.board_state == BoardState.NOT_FINISHED:
            player = players[turn]
            move = player.get_move(board)
            if move is None:
                break
            board.make_move(move)
            turn = swap_piece(turn)

        if board.board_state == BoardState.X_WON:
            p_x.end_episode(1.0, max_entries)
            p_o.end_episode(-1.0, max_entries)
            wins_x += 1
        elif board.board_state == BoardState.O_WON:
            p_x.end_episode(-1.0, max_entries)
            p_o.end_episode(1.0, max_entries)
            wins_o += 1
        else:
            p_x.end_episode(0.0, max_entries)
            p_o.end_episode(0.0, max_entries)
            draws += 1

        if report_interval and ep % report_interval == 0:
            elapsed = time.monotonic() - t0
            total = wins_x + wins_o + draws
            pct_done = ep / episodes * 100
            bar = _bar(ep / episodes)
            line = (
                f"  {dim(f'w{worker_id:02d}')}  "
                f"{ep:>{len(str(episodes))},}/{episodes:,}  "
                f"{bar}  "
                f"{bold(f'{pct_done:5.1f}%')}  "
                f"ε={eps:.3f}  "
                f"Q={len(q_table):,}  "
                f"{green('X')}:{wins_x / total * 100:4.1f}% "
                f"{red('O')}:{wins_o / total * 100:4.1f}% "
                f"{yellow('D')}:{draws / total * 100:4.1f}%  "
                f"{dim(f'{ep / elapsed:,.0f} ep/s')}"
            )
            eps_per_sec = ep / elapsed
            eta = (episodes - ep) / eps_per_sec if eps_per_sec > 0 else 0
            line += f"  {cyan('ETA')} {_fmt_time(eta)}"
            if _IS_TTY and workers > 1:
                # Move up to this worker's reserved line, overwrite, move back down
                up = workers - worker_id
                sys.stdout.write(f"\033[{up}A\r\033[2K{line}\033[{up}B\r")
                sys.stdout.flush()
            else:
                print(line, flush=True)
            wins_x = wins_o = draws = 0

    return q_table


# ── Merge ─────────────────────────────────────────────────────────────────────


def _merge_tables(tables: list[dict[int, float]]) -> dict[int, float]:
    merged: dict[int, float] = {}
    counts: dict[int, int] = {}
    for table in tables:
        for k, v in table.items():
            if k in merged:
                merged[k] += v
                counts[k] += 1
            else:
                merged[k] = v
                counts[k] = 1
    for k in counts:
        if counts[k] > 1:
            merged[k] /= counts[k]
    return merged


# ── Train ─────────────────────────────────────────────────────────────────────


def train(
    episodes: int,
    workers: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
    name: str,
    load_name: str | None,
    report_interval: int,
    max_entries: int,
) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    load_path = os.path.join(MODELS_DIR, f"{load_name}.pkl") if load_name else None

    mem_per_worker_mb = max_entries * 120 // 1_000_000 if max_entries else 0
    cap_str = (
        f"{max_entries:,}  {dim(f'(~{mem_per_worker_mb * workers:,} MB total)')}"
        if max_entries
        else "unlimited"
    )

    print(_box("Q-Learning Training  —  Ultimate Tic-Tac-Toe "))
    print()
    print(f"  {bold('Episodes')}   : {episodes:,}")
    print(f"  {bold('Workers')}    : {workers}")
    print(f"  {bold('Max entries')} : {cap_str}")
    print(
        f"  {bold('α')}={alpha}  {bold('γ')}={gamma}  {bold('ε')} {epsilon_start:.2f} → {epsilon_end:.2f}  (decay over {epsilon_decay_episodes:,} ep/worker)"
    )
    print(f"  {bold('Save to')}    : {save_path}")
    print()

    initial_q_table: dict[int, float] = {}
    if load_path and os.path.exists(load_path):
        with open(load_path, "rb") as f:
            initial_q_table = pickle.load(f)
        print(
            f"  {cyan('↺')} Resuming from {bold(load_path)} ({len(initial_q_table):,} entries)\n"
        )

    eps_per_worker = episodes // workers
    remainder = episodes - eps_per_worker * workers
    worker_args = [
        (
            eps_per_worker + (1 if i < remainder else 0),
            alpha,
            gamma,
            epsilon_start,
            epsilon_end,
            epsilon_decay_episodes // workers,
            report_interval,
            i,
            workers,
            max_entries,
            initial_q_table,
        )
        for i in range(workers)
    ]

    print(dim("  " + "─" * 90))
    # Reserve one line per worker so they can overwrite in place
    if _IS_TTY and workers > 1:
        for _ in range(workers):
            print()
    t0 = time.monotonic()

    if workers == 1:
        tables = [_run_episodes(worker_args[0])]
    else:
        with Pool(workers) as pool:
            tables = pool.map(_run_episodes, worker_args)

    q_table = _merge_tables(tables)
    elapsed = time.monotonic() - t0
    print(dim("  " + "─" * 90))

    with open(save_path, "wb") as f:
        pickle.dump(q_table, f, protocol=pickle.HIGHEST_PROTOCOL)

    print()
    print(_box("Training Complete"))
    print()
    print(
        f"  {bold('Time')}        : {_fmt_time(elapsed)}  {dim(f'({episodes / elapsed:,.0f} ep/s combined)')}"
    )
    print(f"  {bold('Q-table')}     : {len(q_table):,} entries")
    print(f"  {bold('Saved to')}    : {green(save_path)}")
    print()


# ── Eval ──────────────────────────────────────────────────────────────────────


def evaluate(model_name: str, episodes: int) -> None:
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        print(red(f"  No model found at {model_path}"))
        available = (
            [f[:-4] for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
            if os.path.isdir(MODELS_DIR)
            else []
        )
        if available:
            print(f"  Available: {', '.join(sorted(available))}")
        return

    with open(model_path, "rb") as f:
        q_table = pickle.load(f)

    print(_box("Q-Learning Eval  —  vs Random Opponent"))
    print()
    print(f"  {bold('Model')}    : {model_path}  {dim(f'({len(q_table):,} entries)')}")
    print(f"  {bold('Episodes')} : {episodes:,}")
    print()

    import random

    wins = losses = draws = 0
    t0 = time.monotonic()

    for g in range(episodes):
        board = get_board()
        turn = Piece.X
        trained_piece = Piece.X if g % 2 == 0 else Piece.O
        trained = QLearningPlayer(trained_piece, q_table=q_table, epsilon=0.0)

        while board.board_state == BoardState.NOT_FINISHED:
            moves = board.legal_moves(turn)
            if not moves:
                break
            move = (
                trained.get_move(board)
                if turn == trained_piece
                else random.choice(moves)
            )
            if move is None:
                break
            board.make_move(move)
            turn = swap_piece(turn)

        if board.board_state == BoardState.DRAW:
            draws += 1
        elif (board.board_state == BoardState.X_WON and trained_piece == Piece.X) or (
            board.board_state == BoardState.O_WON and trained_piece == Piece.O
        ):
            wins += 1
        else:
            losses += 1

    elapsed = time.monotonic() - t0
    total = wins + losses + draws

    w_pct = wins / total * 100
    l_pct = losses / total * 100
    d_pct = draws / total * 100

    print(dim("  " + "─" * 52))
    print(
        f"  {green(bold('Wins  '))}  {wins:>6,}  {_pct_bar(w_pct)}  {bold(f'{w_pct:5.1f}%')}"
    )
    print(
        f"  {red(bold('Losses'))}  {losses:>6,}  {_pct_bar(l_pct)}  {bold(f'{l_pct:5.1f}%')}"
    )
    print(
        f"  {yellow(bold('Draws '))}  {draws:>6,}  {_pct_bar(d_pct)}  {bold(f'{d_pct:5.1f}%')}"
    )
    print(dim("  " + "─" * 52))
    print(f"\n  {dim(_fmt_time(elapsed))}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train / evaluate Q-Learning agent for Ultimate Tic-Tac-Toe"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("train", help="Train via self-play")
    tp.add_argument("--episodes", type=int, default=100_000)
    tp.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name (default: auto from episode count)",
    )
    tp.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Parallel workers (default: all cores)",
    )
    tp.add_argument("--alpha", type=float, default=0.3)
    tp.add_argument("--gamma", type=float, default=0.9)
    tp.add_argument("--epsilon-start", type=float, default=1.0)
    tp.add_argument("--epsilon-end", type=float, default=0.05)
    tp.add_argument("--epsilon-decay", type=int, default=80_000)
    tp.add_argument(
        "--load",
        type=str,
        default=None,
        help="Resume from model name in models/qlearning/",
    )
    tp.add_argument("--report-interval", type=int, default=1000)
    tp.add_argument(
        "--max-entries",
        type=int,
        default=8_000_000,
        help="Max Q-table entries per worker (0=unlimited). Default 8M ≈ 1GB/worker",
    )

    ep = sub.add_parser("eval", help="Evaluate trained model vs random")
    ep.add_argument("--episodes", type=int, default=1_000)
    ep.add_argument(
        "--model", type=str, required=True, help="Model name in models/qlearning/"
    )

    args = parser.parse_args()

    if args.command == "train":
        name = args.name or _default_name(args.episodes)
        train(
            episodes=args.episodes,
            workers=args.workers,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_episodes=args.epsilon_decay,
            name=name,
            load_name=args.load,
            report_interval=args.report_interval,
            max_entries=args.max_entries,
        )
    elif args.command == "eval":
        evaluate(model_name=args.model, episodes=args.episodes)


if __name__ == "__main__":
    main()
