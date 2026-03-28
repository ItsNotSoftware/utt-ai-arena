#!/usr/bin/env python3
"""Headless DQN training via self-play for Ultimate Tic-Tac-Toe."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import deque

# Ensure src/ is importable
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import torch
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
    torch.Tensor,  # state  (7, 9, 9)
    int,  # action
    float,  # reward
    torch.Tensor,  # next_state  (7, 9, 9)
    torch.Tensor,  # next_legal_mask (81,)
    bool,  # done
]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buf.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buf, min(batch_size, len(self.buf)))

    def __len__(self) -> int:
        return len(self.buf)


# ── Self-play episode ─────────────────────────────────────────────────────────


_SHAPING_SCALE = 1.0 / 500.0  # normalises heuristic deltas to ~[-0.2, 0.2] range


def play_episode(
    policy_net: DQNNet,
    epsilon: float,
    replay: ReplayBuffer,
    device: torch.device,
) -> BoardState:
    """Play one self-play game, storing transitions in the replay buffer."""
    board = get_board()
    turn = Piece.X

    # Per-player trajectory: (state, action, mask_of_state, shaping_reward)
    trajectories: dict[Piece, list[tuple[torch.Tensor, int, torch.Tensor, float]]] = {
        Piece.X: [],
        Piece.O: [],
    }

    policy_net.eval()
    prev_score = evaluate_board(board)

    while board.board_state == BoardState.NOT_FINISHED:
        state = encode_board(board, turn)  # CPU tensor for replay buffer
        mask = legal_mask(board, turn)  # store for use as next_mask later
        moves = board.legal_moves(turn)
        if not moves:
            break

        # Epsilon-greedy
        if random.random() < epsilon:
            move = random.choice(moves)
            action = move_to_action(move)
        else:
            with torch.no_grad():
                q = policy_net(state.unsqueeze(0).to(device)).squeeze(0) + mask.to(
                    device
                )
            action = q.argmax().item()
            move = action_to_move(action, turn)

        board.make_move(move)

        # Heuristic shaping reward from this player's perspective
        curr_score = evaluate_board(board)
        sign = 1.0 if turn == Piece.X else -1.0
        shaping_reward = (curr_score - prev_score) * sign * _SHAPING_SCALE
        prev_score = curr_score

        trajectories[turn].append((state, action, mask, shaping_reward))
        turn = swap_piece(turn)

    # Assign terminal rewards and push transitions
    result = board.board_state
    for p, traj in trajectories.items():
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
                next_state = torch.zeros(7, 9, 9)
                next_mask = torch.full((81,), float("-inf"))
            else:
                # Use the actual state and legal mask from the next turn
                next_state, _, next_mask, _ = traj[i + 1]

            replay.push((state, action, reward, next_state, next_mask, done))

    return result


# ── Training ──────────────────────────────────────────────────────────────────


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
    device_str: str | None = None,
) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"{name}.pt")

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

    if load_name:
        load_path = os.path.join(MODELS_DIR, f"{load_name}.pt")
        if os.path.exists(load_path):
            policy_net.load_state_dict(
                torch.load(load_path, map_location=device, weights_only=True)
            )
            print(f"  {cyan('↺')} Resumed from {bold(load_path)}\n")

    # Get state_dict (handle DataParallel's .module wrapper)
    policy_dict = (
        policy_net.module.state_dict() if use_dataparallel else policy_net.state_dict()
    )
    target_net.load_state_dict(policy_dict)
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=episodes, eta_min=1e-6
    )
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
        f"  {bold('Batch')}        : {batch_size}   {bold('Buffer')}: {buffer_size:,}"
    )
    print(f"  {bold('Target sync')}  : every {target_update:,} ep")
    print(
        f"  {bold('γ')}={gamma}  {bold('ε')} {epsilon_start:.2f} → {epsilon_end:.2f}  (decay over {epsilon_decay_episodes:,} ep)"
    )
    print(f"  {bold('Save to')}      : {save_path}")
    print()
    print(dim("  " + "─" * 90))

    wins_x = wins_o = draws = 0
    total_loss = 0.0
    loss_count = 0
    last_eval_str = dim("  (awaiting eval...)")
    t0 = time.monotonic()

    # Reserve a second line for the eval result
    if _IS_TTY:
        sys.stdout.write(f"{last_eval_str}\n\033[1A")
        sys.stdout.flush()

    for ep in range(1, episodes + 1):
        # Epsilon decay
        frac = min(ep / epsilon_decay_episodes, 1.0)
        eps = epsilon_start + (epsilon_end - epsilon_start) * frac

        # Play one episode
        result = play_episode(policy_net, eps, replay, device)
        if result == BoardState.X_WON:
            wins_x += 1
        elif result == BoardState.O_WON:
            wins_o += 1
        else:
            draws += 1

        # Train on a mini-batch
        if len(replay) >= batch_size:
            policy_net.train()
            batch = replay.sample(batch_size)
            states = torch.stack([t[0] for t in batch]).to(device)
            actions = torch.tensor(
                [t[1] for t in batch], dtype=torch.long, device=device
            )
            rewards = torch.tensor(
                [t[2] for t in batch], dtype=torch.float32, device=device
            )
            next_states = torch.stack([t[3] for t in batch]).to(device)
            dones = torch.tensor([t[5] for t in batch], dtype=torch.bool, device=device)

            next_masks = torch.stack([t[4] for t in batch]).to(device)

            # Q(s, a)
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Double DQN: policy_net selects action, target_net evaluates it
            with torch.no_grad():
                next_actions = (policy_net(next_states) + next_masks).argmax(dim=1)
                next_q = (
                    target_net(next_states)
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze(1)
                )
                next_q[dones] = 0.0

            target = rewards + gamma * next_q
            loss = loss_fn(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loss_count += 1

        # Sync target network
        if ep % target_update == 0:
            policy_dict = (
                policy_net.module.state_dict()
                if use_dataparallel
                else policy_net.state_dict()
            )
            target_net.load_state_dict(policy_dict)

        # Report
        if ep % report_interval == 0:
            elapsed = time.monotonic() - t0
            total = wins_x + wins_o + draws
            avg_loss = total_loss / loss_count if loss_count else 0.0
            pct = ep / episodes * 100
            bar = _bar(ep / episodes)
            eps_per_sec = ep / elapsed
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
            if _IS_TTY:
                sys.stdout.write(f"\r\033[2K{line}\n\r\033[2K{last_eval_str}\033[1A")
                sys.stdout.flush()
            else:
                print(line, flush=True)
            wins_x = wins_o = draws = 0
            total_loss = 0.0
            loss_count = 0

        # Periodic eval
        if eval_interval and ep % eval_interval == 0:
            wr = _quick_eval(policy_net, device, n=100)
            last_eval_str = (
                f"  {cyan('►')} Eval vs random: {bold(f'{wr:.0f}%')} win rate"
            )
            if _IS_TTY:
                sys.stdout.write(f"\n\r\033[2K{last_eval_str}\033[1A")
                sys.stdout.flush()
            else:
                print(last_eval_str, flush=True)

    if _IS_TTY:
        sys.stdout.write("\n\n")
    print(dim("  " + "─" * 90))

    # Save state_dict, handling DataParallel wrapper
    save_dict = (
        policy_net.module.state_dict() if use_dataparallel else policy_net.state_dict()
    )
    torch.save(save_dict, save_path)
    elapsed = time.monotonic() - t0

    print()
    print(_box("Training Complete"))
    print()
    print(
        f"  {bold('Time')}     : {_fmt_time(elapsed)}  {dim(f'({episodes / elapsed:,.0f} ep/s)')}"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train / evaluate DQN agent for Ultimate Tic-Tac-Toe "
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("train", help="Train via self-play")
    tp.add_argument("--episodes", type=int, default=50_000)
    tp.add_argument("--name", type=str, default=None, help="Model name (default: auto)")
    tp.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    tp.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    tp.add_argument("--epsilon-start", type=float, default=1.0)
    tp.add_argument("--epsilon-end", type=float, default=0.05)
    tp.add_argument("--epsilon-decay", type=int, default=40_000)
    tp.add_argument("--batch-size", type=int, default=128)
    tp.add_argument("--buffer-size", type=int, default=200_000)
    tp.add_argument("--target-update", type=int, default=500)
    tp.add_argument("--eval-interval", type=int, default=2_000)
    tp.add_argument("--load", type=str, default=None, help="Resume from model name")
    tp.add_argument("--report-interval", type=int, default=200)
    tp.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cuda:0, mps, xpu, cpu (default: auto-detect)",
    )

    ep = sub.add_parser("eval", help="Evaluate trained model vs random")
    ep.add_argument(
        "--model", type=str, required=True, help="Model name in models/dqn/"
    )
    ep.add_argument("--episodes", type=int, default=1_000)
    ep.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cuda:0, mps, xpu, cpu (default: auto-detect)",
    )

    args = parser.parse_args()

    if args.command == "train":
        name = args.name or _default_name(args.episodes)
        train(
            episodes=args.episodes,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_episodes=args.epsilon_decay,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            target_update=args.target_update,
            eval_interval=args.eval_interval,
            name=name,
            load_name=args.load,
            report_interval=args.report_interval,
            device_str=args.device,
        )
    elif args.command == "eval":
        evaluate(model_name=args.model, episodes=args.episodes, device_str=args.device)


if __name__ == "__main__":
    main()
