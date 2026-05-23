#!/usr/bin/env python3
"""Headless DQN training via self-play for Ultimate Tic-Tac-Toe."""

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import time
from collections import deque

import numpy as np

# Ensure src/ is importable
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from board import Board, BoardState, Piece, get_board, swap_piece
from dqn_model import DQNNet, action_to_move, encode_board, legal_mask, move_to_action
from player import DQNPlayer, evaluate_board

MODELS_DIR = "models/dqn"

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


def blue(t: str) -> str:
    return _c("94", t)


def green(t: str) -> str:
    return _c("92", t)


def red(t: str) -> str:
    return _c("91", t)


def yellow(t: str) -> str:
    return _c("93", t)


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


def _default_name(episodes: int) -> str:
    if episodes >= 1_000_000:
        n = episodes // 1_000_000
        r = (episodes % 1_000_000) // 100_000
        return f"dqn_{n}M" if r == 0 else f"dqn_{n}.{r}M"
    if episodes >= 1_000:
        n = episodes // 1_000
        return f"dqn_{n}k"
    return f"dqn_{episodes}"


# ── Replay Buffer ─────────────────────────────────────────────────────────────

Transition = tuple[
    np.ndarray,  # state  (7, 9, 9)
    int,  # action
    float,  # reward
    np.ndarray,  # next_state  (7, 9, 9)
    np.ndarray,  # next_legal_mask (81,)
    bool,  # done
]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buf: list[Transition] = []
        self.pos = 0

    def push(self, t: Transition) -> None:
        if len(self.buf) < self.capacity:
            self.buf.append(t)
        else:
            self.buf[self.pos] = t
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        k = min(batch_size, len(self.buf))
        indices = np.random.randint(0, len(self.buf), size=k)
        return [self.buf[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buf)


# ── Self-play episodes (batched) ─────────────────────────────────────────────


_SHAPING_SCALE = 1.0 / 500.0  # normalises heuristic deltas to ~[-0.2, 0.2] range

_NEG_INF = float("-inf")
_ZERO_STATE = np.zeros((7, 9, 9), dtype=np.float32)
_NEG_INF_MASK = np.full((81,), _NEG_INF, dtype=np.float32)


def _flush_trajectories(
    trajectories: dict[Piece, list],
    result: BoardState,
    replay: ReplayBuffer,
) -> None:
    """Push a finished game's trajectories into the replay buffer."""
    for p, traj in trajectories.items():
        if not traj:
            continue
        if result == BoardState.DRAW:
            terminal_reward = 0.0
        elif (result == BoardState.X_WON and p == Piece.X) or (
            result == BoardState.O_WON and p == Piece.O
        ):
            terminal_reward = 1.0
        else:
            terminal_reward = -1.0

        for i, (state, action, mask, shaping_reward) in enumerate(traj):
            done = i == len(traj) - 1
            reward = (terminal_reward if done else 0.0) + shaping_reward
            if done:
                next_state = _ZERO_STATE
                next_mask = _NEG_INF_MASK
            else:
                next_state, _, next_mask, _ = traj[i + 1]
            replay.push((state, action, reward, next_state, next_mask, done))


def play_episodes_batched(
    policy_net: DQNNet,
    epsilon: float,
    replay: ReplayBuffer,
    device: torch.device,
    n_games: int = 32,
    use_shaping: bool = True,
) -> list[BoardState]:
    """Play n_games simultaneously with batched model inference."""
    # Per-game state
    boards = [get_board() for _ in range(n_games)]
    turns = [Piece.X] * n_games
    trajectories = [{Piece.X: [], Piece.O: []} for _ in range(n_games)]
    prev_scores = (
        [evaluate_board(b) for b in boards] if use_shaping else [0.0] * n_games
    )
    active = set(range(n_games))
    results: list[BoardState] = [BoardState.NOT_FINISHED] * n_games

    policy_net.eval()

    while active:
        # Gather states for active games that have legal moves
        batch_indices = []  # indices into active games
        states_list = []
        masks_list = []
        moves_list = []

        for idx in sorted(active):
            moves = boards[idx].legal_moves(turns[idx])
            if not moves:
                results[idx] = boards[idx].board_state
                _flush_trajectories(trajectories[idx], results[idx], replay)
                active.discard(idx)
                continue
            batch_indices.append(idx)
            moves_list.append(moves)
            states_list.append(encode_board(boards[idx], turns[idx]))
            masks_list.append(legal_mask(boards[idx], turns[idx]))

        if not batch_indices:
            break

        # Single batched forward pass for all active games
        with torch.no_grad():
            state_batch = torch.stack(states_list).to(device)
            mask_batch = torch.stack(masks_list).to(device)
            q_batch = policy_net(state_batch) + mask_batch

        # Dispatch actions per game
        for j, idx in enumerate(batch_indices):
            moves = moves_list[j]
            turn = turns[idx]

            if random.random() < epsilon:
                move = random.choice(moves)
                action = move_to_action(move)
            else:
                action = q_batch[j].argmax().item()
                move = action_to_move(action, turn)

            boards[idx].make_move(move)

            # Shaping reward
            if use_shaping:
                curr_score = evaluate_board(boards[idx])
                sign = 1.0 if turn == Piece.X else -1.0
                shaping = (curr_score - prev_scores[idx]) * sign * _SHAPING_SCALE
                prev_scores[idx] = curr_score
            else:
                shaping = 0.0

            trajectories[idx][turn].append(
                (states_list[j].numpy(), action, masks_list[j].numpy(), shaping)
            )
            turns[idx] = swap_piece(turn)

            if boards[idx].board_state != BoardState.NOT_FINISHED:
                results[idx] = boards[idx].board_state
                _flush_trajectories(trajectories[idx], results[idx], replay)
                active.discard(idx)

    return results


# ── Multiprocessing self-play workers ────────────────────────────────────────

_worker_net: DQNNet | None = None


def _init_selfplay_worker() -> None:
    """Pool initializer: create a thread-local model once per worker process."""
    global _worker_net
    torch.set_num_threads(1)  # workers do CPU self-play; no benefit from extra threads
    _worker_net = DQNNet()
    _worker_net.eval()


def _run_selfplay(
    args: tuple[dict | None, float, int, bool],
) -> tuple[list[BoardState], list[Transition]]:
    """Pool task: load weights (if provided), play batched games, return results + transitions."""
    state_dict, epsilon, n_games, use_shaping = args
    assert _worker_net is not None
    if state_dict is not None:
        _worker_net.load_state_dict(state_dict)
    buf = ReplayBuffer(n_games * 80)
    device = torch.device("cpu")
    results = play_episodes_batched(
        _worker_net,
        epsilon,
        buf,
        device,
        n_games,
        use_shaping,
    )
    return results, list(buf.buf)


# ── Training ──────────────────────────────────────────────────────────────────


def _save_checkpoint(
    path: str,
    raw_net: DQNNet,
    optimizer: optim.Optimizer,
    scheduler,
    episode: int,
) -> None:
    """Save full training state so training can be resumed exactly."""
    torch.save(
        {
            "model": raw_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "episode": episode,
        },
        path,
    )


def train(
    episodes: int,
    lr: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
    batch_size: int,
    buffer_size: int,
    target_update: int,
    eval_interval: int,
    name: str,
    load_name: str | None,
    report_interval: int,
    grad_steps: int = 4,
    device_str: str | None = None,
    checkpoint_interval: int = 0,
    batch_games: int = 256,
    use_shaping: bool = True,
    use_compile: bool = False,
    n_workers: int = 0,
) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"{name}.pt")
    ckpt_path = os.path.join(MODELS_DIR, f"{name}.ckpt.pt")

    # Device selection: CUDA > MPS > XPU > CPU
    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif (
        hasattr(torch, "xpu")
        and hasattr(torch.xpu, "is_available")
        and torch.xpu.is_available()
    ):
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    policy_net = DQNNet().to(device)
    target_net = DQNNet().to(device)

    # DataParallel for multi-GPU
    use_dataparallel = device.type == "cuda" and torch.cuda.device_count() > 1
    if use_dataparallel:
        policy_net = nn.DataParallel(policy_net)

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # T_max = expected total optimizer steps (not episodes)
    effective_batch = (
        batch_games if n_workers == 0 else max(1, batch_games // n_workers) * n_workers
    )
    total_opt_steps = max(1, (episodes // effective_batch) * grad_steps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_opt_steps, eta_min=1e-6
    )

    start_episode = 0
    if load_name:
        # Prefer checkpoint (full state) over weights-only file
        ckpt_load = os.path.join(MODELS_DIR, f"{load_name}.ckpt.pt")
        weights_load = os.path.join(MODELS_DIR, f"{load_name}.pt")
        if os.path.exists(ckpt_load):
            ckpt = torch.load(ckpt_load, map_location=device, weights_only=False)
            raw = policy_net.module if use_dataparallel else policy_net
            raw.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_episode = ckpt["episode"]
            print(
                f"  {cyan('↺')} Resumed checkpoint from {bold(ckpt_load)}  (ep {start_episode:,})\n"
            )
        elif os.path.exists(weights_load):
            raw = policy_net.module if use_dataparallel else policy_net
            raw.load_state_dict(
                torch.load(weights_load, map_location=device, weights_only=True)
            )
            print(f"  {cyan('↺')} Loaded weights from {bold(weights_load)}\n")

    # Get state_dict (handle DataParallel's .module wrapper)
    policy_dict = (
        policy_net.module.state_dict() if use_dataparallel else policy_net.state_dict()
    )
    target_net.load_state_dict(policy_dict)
    target_net.eval()

    # Compile for faster GPU execution (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile"):
        torch.set_float32_matmul_precision("high")  # enable TF32 tensor cores
        policy_net = torch.compile(policy_net, dynamic=True)
        target_net = torch.compile(target_net, dynamic=True)
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(buffer_size)

    print(_box("DQN Training  —  Ultimate Tic-Tac-Toe"))
    print()
    print(f"  {bold('Episodes')}     : {episodes:,}")
    device_str = str(device)
    if device.type == "cuda":
        device_str += f"  ({torch.cuda.device_count()} GPU{'s' if torch.cuda.device_count() > 1 else ''}{'  DataParallel' if use_dataparallel else ''})"
    print(f"  {bold('Device')}       : {device_str}")
    print(f"  {bold('LR')}           : {lr}")
    print(
        f"  {bold('Batch')}        : {batch_size}   {bold('Buffer')}: {buffer_size:,}   {bold('Grad steps')}: {grad_steps}"
    )
    print(f"  {bold('Target sync')}  : every {target_update:,} ep")
    print(
        f"  {bold('γ')}={gamma}  {bold('ε')} {epsilon_start:.2f} → {epsilon_end:.2f}  (decay over {epsilon_decay_episodes:,} ep)"
    )
    print(f"  {bold('Batch games')}  : {batch_games}")
    workers_str = (
        f"{n_workers} processes" if n_workers > 0 else dim("off (single-process)")
    )
    shaping_str = green("on") if use_shaping else dim("off")
    compile_str = green("on") if use_compile else dim("off")
    print(f"  {bold('Workers')}      : {workers_str}")
    print(
        f"  {bold('Shaping')}      : {shaping_str}   {bold('Compile')}: {compile_str}"
    )
    print(f"  {bold('Save to')}      : {save_path}")
    print()
    print(dim("  " + "─" * 90))

    # Create worker pool for parallel self-play
    pool = None
    if n_workers > 0:
        ctx = mp.get_context("forkserver")
        pool = ctx.Pool(n_workers, initializer=_init_selfplay_worker)

    wins_x = wins_o = draws = 0
    total_loss = 0.0
    loss_count = 0
    last_eval_str = ""
    t0 = time.monotonic()

    # ── SIGINT handler: save and exit cleanly on Ctrl+C ──────────────────────
    _interrupted = False

    def _handle_interrupt(sig, frame):
        nonlocal _interrupted
        _interrupted = True

    signal.signal(signal.SIGINT, _handle_interrupt)

    def _get_raw() -> DQNNet:
        raw = policy_net.module if use_dataparallel else policy_net
        return raw._orig_mod if hasattr(raw, "_orig_mod") else raw

    ep = start_episode
    _need_weight_sync = True  # workers need initial weights
    pending_async = None  # AsyncResult from pool for pipelining

    def _dispatch_selfplay(eps_val: float, n_games: int):
        """Fire off self-play work (async if pool, blocking otherwise)."""
        nonlocal _need_weight_sync
        if pool is not None:
            sd = (
                {k: v.cpu() for k, v in _get_raw().state_dict().items()}
                if _need_weight_sync
                else None
            )
            _need_weight_sync = False
            per_w = max(1, n_games // n_workers)
            tasks = [(sd, eps_val, per_w, use_shaping)] * n_workers
            return pool.map_async(_run_selfplay, tasks)
        return None  # single-process: run inline

    def _collect_selfplay(async_result, eps_val: float, n_games: int):
        """Collect self-play results (from async workers or inline)."""
        all_results: list[BoardState] = []
        if async_result is not None:
            for batch_results, transitions in async_result.get():
                for t in transitions:
                    replay.push(t)
                all_results.extend(batch_results)
        else:
            all_results = play_episodes_batched(
                policy_net,
                eps_val,
                replay,
                device,
                n_games=n_games,
                use_shaping=use_shaping,
            )
        return all_results

    # Pre-allocate numpy buffers for batch assembly (avoid repeated allocation)
    _buf_s = np.empty((batch_size, 7, 9, 9), dtype=np.float32)
    _buf_ns = np.empty((batch_size, 7, 9, 9), dtype=np.float32)
    _buf_nm = np.empty((batch_size, 81), dtype=np.float32)
    _buf_a = np.empty(batch_size, dtype=np.int64)
    _buf_r = np.empty(batch_size, dtype=np.float32)
    _buf_d = np.empty(batch_size, dtype=np.bool_)

    def _train_step():
        """Run one gradient step on the GPU."""
        batch = replay.sample(batch_size)
        k = len(batch)

        # Fill pre-allocated buffers
        for i, t in enumerate(batch):
            _buf_s[i] = t[0]
            _buf_a[i] = t[1]
            _buf_r[i] = t[2]
            _buf_ns[i] = t[3]
            _buf_nm[i] = t[4]
            _buf_d[i] = t[5]

        states = torch.from_numpy(_buf_s[:k]).to(device, non_blocking=True)
        next_states = torch.from_numpy(_buf_ns[:k]).to(device, non_blocking=True)
        next_masks = torch.from_numpy(_buf_nm[:k]).to(device, non_blocking=True)
        actions = torch.from_numpy(_buf_a[:k]).to(device, non_blocking=True)
        rewards = torch.from_numpy(_buf_r[:k]).to(device, non_blocking=True)
        dones = torch.from_numpy(_buf_d[:k]).to(device, non_blocking=True)

        # Q(s, a)
        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: policy_net selects action, target_net evaluates it
        with torch.no_grad():
            next_actions = (policy_net(next_states) + next_masks).argmax(dim=1)
            next_q = (
                target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            )
            next_q[dones] = 0.0

        target = rewards + gamma * next_q
        loss = loss_fn(q_values, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        return loss.item()

    while ep < episodes:
        # How many games in this round (don't overshoot total)
        if pool is not None:
            games_per_worker = max(1, batch_games // n_workers)
            n = games_per_worker * n_workers
        else:
            n = batch_games
        n = min(n, episodes - ep)

        # Epsilon decay (use midpoint of batch)
        mid_ep = ep + n // 2
        frac = min(mid_ep / epsilon_decay_episodes, 1.0)
        eps = epsilon_start + (epsilon_end - epsilon_start) * frac

        # ── Pipelined: collect previous round + dispatch next round ──────
        if pending_async is not None:
            # Collect previous round's results
            all_results = _collect_selfplay(pending_async, prev_eps, prev_n)
        else:
            # First iteration or single-process: run self-play now
            all_results = _collect_selfplay(
                _dispatch_selfplay(eps, n) if pool is not None else None,
                eps,
                n,
            )

        n_actual = len(all_results)
        ep += n_actual

        for result in all_results:
            if result == BoardState.X_WON:
                wins_x += 1
            elif result == BoardState.O_WON:
                wins_o += 1
            else:
                draws += 1

        # Dispatch NEXT round of self-play (runs in background while we train)
        next_n = min(
            batch_games if pool is None else games_per_worker * n_workers, episodes - ep
        )
        next_mid = ep + next_n // 2
        next_frac = min(next_mid / epsilon_decay_episodes, 1.0)
        next_eps = epsilon_start + (epsilon_end - epsilon_start) * next_frac
        if pool is not None and next_n > 0 and not _interrupted:
            pending_async = _dispatch_selfplay(next_eps, next_n)
            prev_eps, prev_n = next_eps, next_n
        else:
            pending_async = None

        # Train on mini-batches (GPU works while workers play next round)
        if len(replay) >= batch_size:
            policy_net.train()
            for _ in range(grad_steps):
                l = _train_step()
                total_loss += l
                loss_count += 1

        # Sync target network (and flag workers for weight update)
        if ep // target_update > (ep - n) // target_update:
            policy_dict = (
                policy_net.module.state_dict()
                if use_dataparallel
                else policy_net.state_dict()
            )
            target_net.load_state_dict(policy_dict)
            if pool is not None:
                _need_weight_sync = True

        # Report
        if ep // report_interval > (ep - n) // report_interval:
            elapsed = time.monotonic() - t0
            total = wins_x + wins_o + draws
            avg_loss = total_loss / loss_count if loss_count else 0.0
            pct = ep / episodes * 100
            bar = _bar(ep / episodes)
            trained = ep - start_episode
            eps_per_sec = trained / elapsed if elapsed > 0 else 0
            eta = (episodes - ep) / eps_per_sec if eps_per_sec > 0 else 0
            current_lr = scheduler.get_last_lr()[0]
            line = (
                f"  {ep:>{len(str(episodes))},}/{episodes:,}  "
                f"{bar}  "
                f"{bold(f'{pct:5.1f}%')}  "
                f"ε={eps:.3f}  "
                f"lr={current_lr:.2e}  "
                f"loss={avg_loss:.4f}  "
                f"{green('X')}:{wins_x / total * 100:4.1f}% "
                f"{red('O')}:{wins_o / total * 100:4.1f}% "
                f"{yellow('D')}:{draws / total * 100:4.1f}%  "
                f"{dim(f'{eps_per_sec:,.0f} ep/s')}  "
                f"{cyan('ETA')} {_fmt_time(eta)}"
            )
            if last_eval_str:
                line = f"{line}  {last_eval_str}"
            if _IS_TTY:
                sys.stdout.write(f"\r\033[2K{line}")
                sys.stdout.flush()
            else:
                print(line, flush=True)
            wins_x = wins_o = draws = 0
            total_loss = 0.0
            loss_count = 0

        # Periodic eval
        if eval_interval and ep // eval_interval > (ep - n) // eval_interval:
            wr = _quick_eval(policy_net, device, n=100)
            last_eval_str = f"{cyan('►')} eval:{bold(f'{wr:.0f}%')}"

        # Periodic checkpoint
        if (
            checkpoint_interval
            and ep // checkpoint_interval > (ep - n) // checkpoint_interval
        ):
            _save_checkpoint(ckpt_path, _get_raw(), optimizer, scheduler, ep)

        # Ctrl+C: save checkpoint and exit
        if _interrupted:
            if pool is not None:
                pool.terminate()
                pool.join()
            print(yellow(f"\n\n  Interrupted at episode {ep:,} — saving checkpoint…"))
            _save_checkpoint(ckpt_path, _get_raw(), optimizer, scheduler, ep)
            torch.save(_get_raw().state_dict(), save_path)
            print(f"  {cyan('✔')} Checkpoint : {bold(ckpt_path)}")
            print(f"  {cyan('✔')} Weights    : {bold(save_path)}")
            print(f"\n  Resume with: {bold(f'--load {name}')}")
            return

    if pool is not None:
        pool.close()
        pool.join()

    print(f"\n\n{dim('  ' + '─' * 90)}")

    # Save state_dict, stripping DataParallel / torch.compile wrappers
    torch.save(_get_raw().state_dict(), save_path)
    # Clean up checkpoint file if training completed fully
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    elapsed = time.monotonic() - t0

    print()
    print(_box("Training Complete"))
    print()
    trained = episodes - start_episode
    print(
        f"  {bold('Time')}     : {_fmt_time(elapsed)}  {dim(f'({trained / elapsed:,.0f} ep/s)')}"
    )
    print(f"  {bold('Saved to')} : {green(save_path)}")
    print()


# ── Quick eval ────────────────────────────────────────────────────────────────


def _quick_eval(net: DQNNet, device: torch.device, n: int = 200) -> float:
    """Win rate (%) of net vs random over n games."""
    net.eval()
    wins = 0
    for g in range(n):
        board = get_board()
        turn = Piece.X
        trained_piece = Piece.X if g % 2 == 0 else Piece.O

        while board.board_state == BoardState.NOT_FINISHED:
            moves = board.legal_moves(turn)
            if not moves:
                break
            if turn == trained_piece:
                state = encode_board(board, turn).to(device)
                mask = legal_mask(board, turn).to(device)
                with torch.no_grad():
                    q = net(state.unsqueeze(0)).squeeze(0) + mask
                action = q.argmax().item()
                move = action_to_move(action, turn)
            else:
                move = random.choice(moves)
            board.make_move(move)
            turn = swap_piece(turn)

        if (board.board_state == BoardState.X_WON and trained_piece == Piece.X) or (
            board.board_state == BoardState.O_WON and trained_piece == Piece.O
        ):
            wins += 1
    return wins / n * 100


# ── Eval command ──────────────────────────────────────────────────────────────


def evaluate(model_name: str, episodes: int, device_str: str | None = None) -> None:
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    if not os.path.exists(model_path):
        print(red(f"  No model found at {model_path}"))
        available = (
            sorted(f[:-3] for f in os.listdir(MODELS_DIR) if f.endswith(".pt"))
            if os.path.isdir(MODELS_DIR)
            else []
        )
        if available:
            print(f"  Available: {', '.join(available)}")
        return

    # Device selection: CUDA > MPS > XPU > CPU
    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif (
        hasattr(torch, "xpu")
        and hasattr(torch.xpu, "is_available")
        and torch.xpu.is_available()
    ):
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    net = DQNNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    params = sum(p.numel() for p in net.parameters())
    print(_box("DQN Eval  —  vs Random Opponent"))
    print()
    print(f"  {bold('Model')}    : {model_path}  {dim(f'({params:,} params)')}")
    print(f"  {bold('Episodes')} : {episodes:,}")
    print()

    t0 = time.monotonic()
    wins = losses = draws = 0
    net.eval()
    for g in range(episodes):
        board = get_board()
        turn = Piece.X
        trained_piece = Piece.X if g % 2 == 0 else Piece.O
        while board.board_state == BoardState.NOT_FINISHED:
            moves = board.legal_moves(turn)
            if not moves:
                break
            if turn == trained_piece:
                state = encode_board(board, turn).to(device)
                mask = legal_mask(board, turn).to(device)
                with torch.no_grad():
                    q = net(state.unsqueeze(0)).squeeze(0) + mask
                action = q.argmax().item()
                move = action_to_move(action, turn)
            else:
                move = random.choice(moves)
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


def _auto_workers() -> int:
    """Pick worker count based on CPU cores (self-play is CPU-bound regardless of device)."""
    cores = os.cpu_count() or 1
    # Leave a few threads for main process (training + GPU) — ~75% of cores, capped at 8
    return max(0, min(cores * 3 // 4, 8))


def _auto_compile(args: argparse.Namespace) -> bool:
    """torch.compile: off by default (small model, marginal benefit). Opt-in with --compile."""
    if args.no_compile:
        return False
    return bool(args.compile)


def _print_usage() -> None:
    print(f"""
{bold('Ultimate Tic-Tac-Toe — DQN Training')}

{bold('Usage:')}
  uv run scripts/train_dqn.py train                   Train with defaults (250k episodes)
  uv run scripts/train_dqn.py train --episodes 500000  Custom episode count
  uv run scripts/train_dqn.py train --name my_model    Custom model name
  uv run scripts/train_dqn.py train --load dqn_250k    Resume from checkpoint
  uv run scripts/train_dqn.py eval  --model dqn_250k   Evaluate a trained model

{bold('Options:')}
  --episodes N     Number of training episodes (default: 250,000)
  --name NAME      Model name (default: dqn_<episodes>)
  --load NAME      Resume training from a saved model/checkpoint
  --batch-size N   Training batch size (default: 1024)

{dim('Device and parallelism are auto-detected. Press Ctrl+C to stop and save.')}
""")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        _print_usage()
        return

    parser = argparse.ArgumentParser(
        description="Train / evaluate DQN agent for Ultimate Tic-Tac-Toe",
        add_help=False,
    )
    sub = parser.add_subparsers(dest="command")

    tp = sub.add_parser("train", help="Train via self-play")
    tp.add_argument("--episodes", type=int, default=25_000, help="Training episodes")
    tp.add_argument("--name", type=str, default=None, help="Model name (default: auto)")
    tp.add_argument("--load", type=str, default=None, help="Resume from model name")
    tp.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size (default: 1024)",
    )
    # Advanced (hidden) — still usable, just not shown in --help
    S = argparse.SUPPRESS
    tp.add_argument("--workers", type=int, default=None, help=S)
    tp.add_argument("--device", type=str, default=None, help=S)
    tp.add_argument("--lr", type=float, default=3e-4, help=S)
    tp.add_argument("--gamma", type=float, default=0.99, help=S)
    tp.add_argument("--epsilon-start", type=float, default=1.0, help=S)
    tp.add_argument("--epsilon-end", type=float, default=0.05, help=S)
    tp.add_argument("--epsilon-decay", type=int, default=None, help=S)
    tp.add_argument("--buffer-size", type=int, default=500_000, help=S)
    tp.add_argument("--target-update", type=int, default=1_000, help=S)
    tp.add_argument("--grad-steps", type=int, default=4, help=S)
    tp.add_argument("--eval-interval", type=int, default=2_000, help=S)
    tp.add_argument("--report-interval", type=int, default=100, help=S)
    tp.add_argument("--checkpoint-interval", type=int, default=5_000, help=S)
    tp.add_argument("--batch-games", type=int, default=256, help=S)
    tp.add_argument("--no-shaping", action="store_true", help=S)
    tp.add_argument("--compile", action="store_true", default=None, help=S)
    tp.add_argument("--no-compile", action="store_true", help=S)

    ep = sub.add_parser("eval", help="Evaluate trained model vs random")
    ep.add_argument(
        "--model", type=str, required=True, help="Model name in models/dqn/"
    )
    ep.add_argument("--episodes", type=int, default=1_000, help=S)
    ep.add_argument("--device", type=str, default=None, help=S)

    args = parser.parse_args()

    if not args.command:
        _print_usage()
        return

    if args.command == "train":
        name = args.name or _default_name(args.episodes)
        epsilon_decay = (
            args.epsilon_decay
            if args.epsilon_decay is not None
            else int(args.episodes * 0.8)
        )
        n_workers = args.workers if args.workers is not None else _auto_workers()
        train(
            episodes=args.episodes,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_episodes=epsilon_decay,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            target_update=args.target_update,
            eval_interval=args.eval_interval,
            name=name,
            load_name=args.load,
            report_interval=args.report_interval,
            grad_steps=args.grad_steps,
            device_str=args.device,
            checkpoint_interval=args.checkpoint_interval,
            batch_games=args.batch_games,
            use_shaping=not args.no_shaping,
            use_compile=_auto_compile(args),
            n_workers=n_workers,
        )
    elif args.command == "eval":
        evaluate(model_name=args.model, episodes=args.episodes, device_str=args.device)


if __name__ == "__main__":
    main()
