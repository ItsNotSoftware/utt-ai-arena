"""DQN network and board encoding for Ultimate Tic-Tac-Toe.

Kept separate from player.py so that `import torch` only happens when DQN
is actually used — the rest of the game works fine without PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from board import Board, BoardState, Move, Piece, board_state_to_piece


class DQNNet(nn.Module):
    """3-layer CNN → FC head.  Input: (batch, 7, 9, 9)  Output: (batch, 81)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 81, 256),
            nn.ReLU(),
            nn.Linear(256, 81),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


def encode_board(board: Board, piece: Piece) -> torch.Tensor:
    """Encode board as (7, 9, 9) float tensor from *piece*'s perspective.

    Channels:
      0  current player's pieces
      1  opponent's pieces
      2  legal moves
      3  inner boards won by current player  (entire 3×3 block = 1)
      4  inner boards won by opponent        (entire 3×3 block = 1)
      5  drawn inner boards                  (entire 3×3 block = 1)
      6  restriction plane                   (restricted 3×3 block = 1)
    """
    t = torch.zeros(7, 9, 9)
    opp = Piece.O if piece == Piece.X else Piece.X
    own_won = BoardState.X_WON if piece == Piece.X else BoardState.O_WON
    opp_won = BoardState.O_WON if piece == Piece.X else BoardState.X_WON

    for R in range(3):
        for C in range(3):
            inner = board[R][C]
            r_off = R * 3
            c_off = C * 3

            # Channels 0-1: piece positions
            for r in range(3):
                for c in range(3):
                    cell = inner[r][c]
                    if cell == piece:
                        t[0, r_off + r, c_off + c] = 1.0
                    elif cell == opp:
                        t[1, r_off + r, c_off + c] = 1.0

            # Channels 3-5: inner board outcomes (fill entire 3x3 block)
            st = inner.board_state
            if st == own_won:
                t[3, r_off : r_off + 3, c_off : c_off + 3] = 1.0
            elif st == opp_won:
                t[4, r_off : r_off + 3, c_off : c_off + 3] = 1.0
            elif st == BoardState.DRAW:
                t[5, r_off : r_off + 3, c_off : c_off + 3] = 1.0

    # Channel 2: legal moves
    for m in board.legal_moves(piece):
        t[2, m.outer[0] * 3 + m.inner[0], m.outer[1] * 3 + m.inner[1]] = 1.0

    # Channel 6: restriction
    if board.restriction is not None:
        rR, rC = board.restriction
        t[6, rR * 3 : rR * 3 + 3, rC * 3 : rC * 3 + 3] = 1.0

    return t


def move_to_action(move: Move) -> int:
    """Move → flat action index in [0, 81)."""
    return move.outer[0] * 27 + move.outer[1] * 9 + move.inner[0] * 3 + move.inner[1]


def action_to_move(action: int, piece: Piece) -> Move:
    """Flat action index → Move."""
    outer_r, rem = divmod(action, 27)
    outer_c, rem = divmod(rem, 9)
    inner_r, inner_c = divmod(rem, 3)
    return Move(piece, (outer_r, outer_c), (inner_r, inner_c))


def legal_mask(board: Board, piece: Piece) -> torch.Tensor:
    """Returns a (81,) tensor: 0.0 for legal actions, -inf for illegal."""
    mask = torch.full((81,), float("-inf"))
    for m in board.legal_moves(piece):
        mask[move_to_action(m)] = 0.0
    return mask
