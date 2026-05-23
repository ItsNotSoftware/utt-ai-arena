"""AlphaZero network for Ultimate Tic-Tac-Toe.

Two-headed ResNet (policy + value) operating on the same (7, 9, 9) board
encoding used by the DQN. Kept separate from player.py so torch is only
imported when AlphaZero is actually used.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Reuse the DQN encoder — it already exposes everything we need.
from dqn_model import encode_board, legal_mask, move_to_action, action_to_move  # noqa: F401


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class AlphaZeroNet(nn.Module):
    """ResNet trunk with policy (81 logits) and value ([-1,1]) heads.

    Input: (batch, 7, 9, 9).
    """

    def __init__(self, num_blocks: int = 3, channels: int = 32) -> None:
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(7, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res = nn.Sequential(*(_ResBlock(channels) for _ in range(num_blocks)))

        # Policy head: 1×1 conv → flatten → 81 logits
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(2 * 81, 81)

        # Value head: 1×1 conv → FC → tanh scalar
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc1 = nn.Linear(81, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res(self.input_conv(x))

        p = self.policy_conv(x).flatten(1)
        p = self.policy_fc(p)  # raw logits

        v = self.value_conv(x).flatten(1)
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return p, v
