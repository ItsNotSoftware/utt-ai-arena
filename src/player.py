from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
from dataclasses import dataclass, field
import math
import random
import pickle
import os

from board import (
    Board,
    BoardState,
    Move,
    Piece,
    WIN_LINES,
    board_state_to_piece,
    swap_piece,
)

# Layout
_screen_w = 0
_screen_h = 0
_board_size = 0
_board_left = 0
_board_top = 0

# Heuristic scores
heuristics = {
    "win": 1_000,
    "draw": 0,
    "two_in_row_outer": 100,
    "inner_win": 50,
    "two_in_row_inner": 10,
    "center_corner": 3,
}

IMPORTANT_POSITIONS = frozenset(((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)))


def _is_empty_board(board: Board) -> bool:
    """True if no piece has been placed yet. The opening position is
    symmetric, so deterministic players would otherwise always pick the same
    cell and produce identical games on every launch."""
    if board.restriction is not None:
        return False
    for r in range(3):
        for c in range(3):
            inner = board[r][c]
            if (
                inner.board_state != BoardState.NOT_FINISHED
                or len(inner.empty_cells) != 9
            ):
                return False
    return True
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


def _two_in_row_score(a: Piece, b: Piece, c: Piece, amount: int) -> float:
    line_sum = a.value + b.value + c.value
    if line_sum == 2 and (a == Piece.EMPTY or b == Piece.EMPTY or c == Piece.EMPTY):
        return float(amount)
    if line_sum == -2 and (a == Piece.EMPTY or b == Piece.EMPTY or c == Piece.EMPTY):
        return float(-amount)
    return 0.0


def evaluate_board(board: Board) -> float:
    "Heuristic eval"
    heur = heuristics
    score = 0.0

    # Outer board heuristic: based on inner board outcomes
    outer_values = [
        [board_state_to_piece(board[r][c].board_state) for c in range(3)]
        for r in range(3)
    ]

    for a, b, c in WIN_LINES:
        score += _two_in_row_score(
            outer_values[a[0]][a[1]],
            outer_values[b[0]][b[1]],
            outer_values[c[0]][c[1]],
            heur["two_in_row_outer"],
        )

    # Evaluate each inner board
    for R in range(3):
        for C in range(3):
            inner = board[R][C]
            st = inner.board_state

            if st == BoardState.X_WON:
                score += heur["inner_win"]
                if (R, C) in IMPORTANT_POSITIONS:
                    score += heur["center_corner"]
            elif st == BoardState.O_WON:
                score -= heur["inner_win"]
                if (R, C) in IMPORTANT_POSITIONS:
                    score -= heur["center_corner"]

            cells = inner.board

            # Reward center/corner occupancy inside inner boards
            for r, c in IMPORTANT_POSITIONS:
                piece = cells[r][c]
                if piece == Piece.X:
                    score += heur["center_corner"]
                elif piece == Piece.O:
                    score -= heur["center_corner"]

            if st != BoardState.NOT_FINISHED:
                continue

            # Two-in-a-row patterns inside an unfinished inner board
            for a, b, c in WIN_LINES:
                score += _two_in_row_score(
                    cells[a[0]][a[1]],
                    cells[b[0]][b[1]],
                    cells[c[0]][c[1]],
                    heur["two_in_row_inner"],
                )

    return score


def set_layout(
    screen_w: int, screen_h: int, board_size: int, board_left: int, board_top: int
) -> None:
    """Sets layout info for input mapping."""
    global _screen_w, _screen_h, _board_size, _board_left, _board_top
    _screen_w, _screen_h = screen_w, screen_h
    _board_size, _board_left, _board_top = board_size, board_left, board_top


class Player(ABC):
    """Abstract player."""

    def __init__(self, piece: Piece) -> None:
        self.piece = piece
        self.name = "Player"
        self._move_count = 0
        self._move_time_total = 0.0

    def get_name(self) -> str:
        return f"{'X' if self.piece == Piece.X else 'O'} – {self.name}"

    def get_move(self, board: Board) -> Move | None:
        return self._select_move(board)

    def record_move_time(self, duration: float) -> None:
        """Record a move duration and print timing stats (called from main process)."""
        self._move_count += 1
        self._move_time_total += duration
        avg = self._move_time_total / self._move_count
        print(
            f"{self.get_name()} move time: {duration:.3f}s "
            f"(avg {avg:.3f}s over {self._move_count} move{'s' if self._move_count != 1 else ''})"
        )

    @abstractmethod
    def _select_move(self, board: Board) -> Move | None: ...


class HumanPlayer(Player):
    """Mouse-based human."""

    def __init__(self, piece: Piece) -> None:
        super().__init__(piece)
        self._prev_down = False
        self.name = "HumanPlayer"

    def _select_move(self, board: Board) -> Move | None:
        from pygame import mouse

        # single click
        down = mouse.get_pressed()[0]
        if not down:
            self._prev_down = False
            return None
        if self._prev_down:
            return None
        self._prev_down = True

        x, y = mouse.get_pos()

        # Must be inside the board square
        if not (
            _board_left <= x < _board_left + _board_size
            and _board_top <= y < _board_top + _board_size
        ):
            return None

        # local coords
        lx = x - _board_left
        ly = y - _board_top

        big = _board_size / 3
        small = _board_size / 9

        out_l = int(ly // big)
        out_c = int(lx // big)
        in_l = int(ly // small) % 3
        in_c = int(lx // small) % 3

        return Move(self.piece, (out_l, out_c), (in_l, in_c))


class MinimaxPlayer(Player):
    """Minimax."""

    def __init__(
        self,
        piece: Piece,
        depth_limit: int = 6,
        use_heuristic_eval=True,
        use_pruning=True,
        use_cache: bool = True,
        max_cache: int = 200_000,
    ) -> None:
        super().__init__(piece)
        self.name = "Minimax"
        self.depth_limit = depth_limit
        features = []
        if use_heuristic_eval:
            features.append("heuristic")
        if use_pruning:
            features.append("pruning")
        if features:
            self.name += " (" + ", ".join(features) + ")"

        self.use_heuristic_eval = use_heuristic_eval
        self.use_pruning = use_pruning
        self.use_cache = use_cache
        self._tt: dict[tuple[int, int], tuple[float, int]] = {}
        self._tt_max = max_cache

    def _store_tt(
        self, tt_key: tuple[int, int] | None, value: float, flag: int = TT_EXACT
    ) -> None:
        if not self.use_cache or tt_key is None:
            return
        if len(self._tt) >= self._tt_max:
            self._tt.clear()
        self._tt[tt_key] = (value, flag)

    def _minimax(
        self,
        piece: Piece,
        board: Board,
        depth: int,
        depth_limit: int,
        alpha: float | None,
        beta: float | None,
    ) -> float:
        self._nodes_visited += 1
        tt_key = None
        alpha_orig = alpha
        beta_orig = beta
        if self.use_cache:
            depth_remaining = depth_limit - depth
            tt_key = (board.packed_key(piece), depth_remaining)
            entry = self._tt.get(tt_key)
            if entry is not None:
                value, flag = entry
                if flag == TT_EXACT:
                    return value
                if self.use_pruning and alpha is not None and beta is not None:
                    if flag == TT_LOWER:
                        alpha = max(alpha, value)
                    elif flag == TT_UPPER:
                        beta = min(beta, value)
                    if alpha >= beta:
                        return value

        # Terminal?
        match board.board_state:
            case BoardState.DRAW:
                value = heuristics["draw"]
                self._store_tt(tt_key, value)
                return value
            case BoardState.X_WON:
                value = heuristics["win"]
                self._store_tt(tt_key, value)
                return value
            case BoardState.O_WON:
                value = -heuristics["win"]
                self._store_tt(tt_key, value)
                return value

        # Depth limit
        if depth >= depth_limit:
            value = evaluate_board(board) if self.use_heuristic_eval else 0.0
            self._store_tt(tt_key, value)
            return value

        # Children
        moves = board.legal_moves(piece)
        if not moves:
            return 0.0
        if self.use_pruning and len(moves) > 1:
            moves = self._order_moves_quick(board, moves, piece)

        maximizing = piece == Piece.X
        best = -math.inf if maximizing else math.inf
        next_piece = swap_piece(piece)
        cutoff = False

        for m in moves:
            token = board.make_move(m)
            if token is None:
                continue  # should not happen with legal_moves
            match board.board_state:
                case BoardState.DRAW:
                    score = heuristics["draw"]
                case BoardState.X_WON:
                    score = heuristics["win"]
                case BoardState.O_WON:
                    score = -heuristics["win"]
                case _:
                    score = self._minimax(
                        next_piece, board, depth + 1, depth_limit, alpha, beta
                    )
            board.undo_move(token)

            if maximizing:
                if score > best:
                    best = score
                    if self.use_pruning and alpha is not None and beta is not None:
                        alpha = max(alpha, best)
                        if alpha >= beta:
                            cutoff = True
                            break
            else:
                if score < best:
                    best = score
                    if self.use_pruning and alpha is not None and beta is not None:
                        beta = min(beta, best)
                        if beta <= alpha:
                            cutoff = True
                            break

        flag = TT_EXACT
        if cutoff and alpha_orig is not None and beta_orig is not None:
            flag = TT_LOWER if maximizing else TT_UPPER
        self._store_tt(tt_key, best, flag)
        return best

    def _move_priority(self, board: Board, move: Move, piece: Piece) -> int:
        score = 0
        inner = board[move.outer[0]][move.outer[1]]
        cells = inner.board
        opponent = swap_piece(piece)
        wins_inner = False

        for a, b, c in WIN_LINES:
            if move.inner != a and move.inner != b and move.inner != c:
                continue
            line = (cells[a[0]][a[1]], cells[b[0]][b[1]], cells[c[0]][c[1]])
            own_count = line.count(piece)
            opp_count = line.count(opponent)
            empty_count = line.count(Piece.EMPTY)
            if own_count == 2 and empty_count == 1:
                score += 1_000
                wins_inner = True
            elif opp_count == 2 and empty_count == 1:
                score += 500
            elif own_count == 1 and opp_count == 0 and empty_count == 2:
                score += 20

        if move.inner == (1, 1):
            score += 30
        elif move.inner in IMPORTANT_POSITIONS:
            score += 15

        if wins_inner:
            own_state = BoardState.X_WON if piece == Piece.X else BoardState.O_WON
            opp_state = BoardState.O_WON if piece == Piece.X else BoardState.X_WON
            for a, b, c in WIN_LINES:
                if move.outer != a and move.outer != b and move.outer != c:
                    continue
                states = []
                for r, col in (a, b, c):
                    states.append(
                        own_state
                        if (r, col) == move.outer
                        else board[r][col].board_state
                    )
                own_count = states.count(own_state)
                opp_count = states.count(opp_state)
                if own_count == 3:
                    score += 100_000
                elif own_count == 2 and opp_count == 0:
                    score += 3_000
        return score

    def _order_moves_quick(
        self, board: Board, moves: list[Move], piece: Piece
    ) -> list[Move]:
        moves.sort(key=lambda m: self._move_priority(board, m, piece), reverse=True)
        return moves

    def _order_moves(self, board: Board, moves: list[Move], piece: Piece) -> list[Move]:
        if not self.use_heuristic_eval or len(moves) < 2:
            return moves
        scored: list[tuple[float, Move]] = []
        for m in moves:
            token = board.make_move(m)
            if token is None:
                continue
            score = evaluate_board(board)
            board.undo_move(token)
            scored.append((score, m))
        if not scored:
            return moves
        reverse = piece == Piece.X
        scored.sort(key=lambda item: item[0], reverse=reverse)
        return [m for _, m in scored]

    def _select_move(self, board: Board) -> Move | None:
        # get legal moves
        moves = board.legal_moves(self.piece)

        if not moves:
            return None

        if _is_empty_board(board):
            return random.choice(moves)

        if self.use_cache:
            self._tt.clear()
        self._nodes_visited = 0

        maximizing = self.piece == Piece.X
        best_score = -math.inf if maximizing else math.inf

        # order moves for better pruning
        if self.use_heuristic_eval:
            moves = self._order_moves(board, moves, self.piece)
        else:
            random.shuffle(moves)
        best_move = moves[0]

        # Evaluate candidate moves sequentially
        if self.use_pruning:
            alpha = -math.inf
            beta = math.inf
        else:
            alpha = None
            beta = None

        for m in moves:
            token = board.make_move(m)
            if token is None:
                # Should not happen with legal_moves;
                continue
            if (board.board_state == BoardState.X_WON and self.piece == Piece.X) or (
                board.board_state == BoardState.O_WON and self.piece == Piece.O
            ):
                board.undo_move(token)
                return m
            score = self._minimax(
                swap_piece(self.piece), board, 1, self.depth_limit, alpha, beta
            )
            board.undo_move(token)

            if maximizing:
                if score > best_score:
                    best_score, best_move = score, m
                    if self.use_pruning and alpha is not None:
                        alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score, best_move = score, m
                    if self.use_pruning and beta is not None:
                        beta = min(beta, best_score)
            if (
                self.use_pruning
                and alpha is not None
                and beta is not None
                and beta <= alpha
            ):
                break

        return best_move


@dataclass(slots=True)
class McNode:
    board: Board
    parent: McNode | None
    children: dict[Move, McNode]
    turn: Piece
    untried_moves: list[Move] = field(default_factory=list)
    total_value: float = 0.0
    n_visits: int = 0


class MonteCarloPlayer(Player):

    def __init__(
        self, piece: Piece, iter_nr: int = 10000, use_heuristics: bool = False
    ) -> None:
        super().__init__(piece)
        self.name = "MonteCarlo Tree Search"
        self.iter_nr = iter_nr
        self.root: McNode | None = None
        self.use_heuristics = use_heuristics

    @staticmethod
    def Q(s, a) -> None:
        pass

    def _ucb(self, parent: McNode, node: McNode, c: float = math.sqrt(2)) -> float:
        if node.n_visits == 0:
            return math.inf
        parent_visits = max(1, parent.n_visits)
        exploitation = node.total_value / node.n_visits
        exploration = c * math.sqrt(math.log(parent_visits) / node.n_visits)
        return exploitation + exploration

    def _new_node(self, board: Board, parent: McNode | None, turn: Piece) -> McNode:
        return McNode(
            board=board,
            parent=parent,
            children={},
            turn=turn,
            untried_moves=board.legal_moves(turn),
        )

    def _select(self, root: McNode) -> McNode:
        current = root

        while True:
            if not current.untried_moves and not current.children:
                return current
            if current.untried_moves:
                return current
            max_ucb = -math.inf
            best_children = []
            log_parent = math.log(max(1, current.n_visits))
            exploration_scale = math.sqrt(2)

            for child in current.children.values():
                if child.n_visits == 0:
                    ucb = math.inf
                else:
                    exploitation = child.total_value / child.n_visits
                    exploration = exploration_scale * math.sqrt(
                        log_parent / child.n_visits
                    )
                    ucb = exploitation + exploration

                if ucb > max_ucb:
                    max_ucb = ucb
                    best_children = [child]
                elif ucb == max_ucb:
                    best_children.append(child)

            # tie-break randomly among best
            current = random.choice(best_children)

    def _expand(self, node: McNode) -> McNode:
        if not node.untried_moves:
            return node
        move = node.untried_moves.pop(random.randrange(0, len(node.untried_moves)))
        new_board = node.board.clone()
        token = new_board.make_move(move)
        if token is None:
            return node
        child = self._new_node(new_board, node, swap_piece(node.turn))
        node.children[move] = child
        return child

    def _simulate(self, node: McNode) -> int:
        base = 1 if self.piece == Piece.X else -1
        sim_board = node.board
        undo_tokens = []
        turn = node.turn

        try:
            while True:
                match sim_board.board_state:
                    case BoardState.DRAW:
                        return 0
                    case BoardState.X_WON:
                        return base
                    case BoardState.O_WON:
                        return -base

                if self.use_heuristics:
                    moves = sim_board.legal_moves(turn)
                    if not moves:
                        return 0
                    best_score = -math.inf if turn == Piece.X else math.inf
                    best_moves = []
                    for m in moves:
                        token = sim_board.make_move(m)
                        if token is None:
                            continue
                        score = evaluate_board(sim_board)
                        sim_board.undo_move(token)
                        if turn == Piece.X:
                            if score > best_score:
                                best_score = score
                                best_moves = [m]
                            elif score == best_score:
                                best_moves.append(m)
                        else:
                            if score < best_score:
                                best_score = score
                                best_moves = [m]
                            elif score == best_score:
                                best_moves.append(m)
                    move = (
                        random.choice(best_moves) if best_moves else random.choice(moves)
                    )
                else:
                    move = sim_board.random_legal_move(turn)
                    if move is None:
                        return 0

                token = sim_board.make_move(move)
                if token is None:
                    return 0
                undo_tokens.append(token)
                turn = swap_piece(turn)
        finally:
            for token in reversed(undo_tokens):
                sim_board.undo_move(token)

    def _backprop(self, node: McNode, value: float) -> None:
        current = node
        while current is not None:
            current.n_visits += 1
            current.total_value += value
            current = current.parent

    def _root_for_board(self, board: Board) -> McNode:
        target_key = board.packed_key()
        if self.root is not None:
            stack: list[tuple[McNode, int]] = [(self.root, 0)]
            while stack:
                node, depth = stack.pop()
                if node.turn == self.piece and node.board.packed_key() == target_key:
                    node.parent = None
                    return node
                if depth < 2:
                    for child in node.children.values():
                        stack.append((child, depth + 1))
        return self._new_node(board.clone(), None, self.piece)

    def _select_move(self, board: Board) -> Move | None:
        moves = board.legal_moves(self.piece)
        if not moves:
            return None
        self.root = self._root_for_board(board)

        for _ in range(self.iter_nr):
            node = self._select(self.root)
            expanded = self._expand(node)
            score = self._simulate(expanded)
            self._backprop(expanded, score)

        best_move = None
        best_visits = -1
        for m, child in self.root.children.items():
            if child.n_visits > best_visits:
                best_visits = child.n_visits
                best_move = m
        if best_move is not None:
            child = self.root.children.get(best_move)
            self.root = child
            if self.root is not None:
                self.root.parent = None
            return best_move

        self.root = None
        return random.choice(moves)


class QLearningPlayer(Player):
    """Tabular Q-Learning with Monte Carlo end-of-episode updates.

    State is encoded as a single integer (board cells in base-3, normalized
    so the current player's pieces are always +1). This lets one Q-table
    serve both sides during self-play training.
    """

    def __init__(
        self,
        piece: Piece,
        q_table: dict[int, float] | None = None,
        alpha: float = 0.3,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        training: bool = False,
    ) -> None:
        super().__init__(piece)
        self.name = "Q-Learning"
        self.q_table: dict[int, float] = q_table if q_table is not None else {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self._history: list[int] = []  # sa-keys for current episode

    # ---- compact encoding --------------------------------------------------

    def _state_key(self, board: Board) -> int:
        """Encode entire board as a single integer.

        Cells are mapped to {0,1,2} (opponent=-1→0, empty=0→1, self=+1→2)
        and packed in base-3.  Restriction is appended as a base-10 digit.
        """
        sign = self.piece.value  # +1 for X, -1 for O
        val = 0
        for R in range(3):
            for C in range(3):
                inner = board[R][C]
                for r in range(3):
                    for c in range(3):
                        val = val * 3 + (inner[r][c].value * sign + 1)
        restr = (
            0
            if board.restriction is None
            else board.restriction[0] * 3 + board.restriction[1] + 1
        )
        return val * 10 + restr

    @staticmethod
    def _action_key(move: Move) -> int:
        return (
            move.outer[0] * 27 + move.outer[1] * 9 + move.inner[0] * 3 + move.inner[1]
        )

    @staticmethod
    def _sa_key(state: int, action: int) -> int:
        return state * 81 + action

    # ---- episode management ------------------------------------------------

    def reset_episode(self) -> None:
        self._history.clear()

    def end_episode(self, reward: float, max_entries: int = 0) -> None:
        """Backprop discounted return through this episode's history.

        max_entries: if > 0, new state-action pairs are only added while the
        table is below that size. Existing entries are always updated.
        """
        g = reward
        alpha = self.alpha
        q = self.q_table
        at_cap = max_entries > 0 and len(q) >= max_entries
        for sa in reversed(self._history):
            if sa in q:
                q[sa] = q[sa] + alpha * (g - q[sa])
            elif not at_cap:
                q[sa] = alpha * g  # 0 + alpha * (g - 0)
                at_cap = max_entries > 0 and len(q) >= max_entries
            g *= self.gamma
        self._history.clear()

    # ---- move selection ----------------------------------------------------

    def _select_move(self, board: Board) -> Move | None:
        moves = board.legal_moves(self.piece)
        if not moves:
            return None

        if not self.training and _is_empty_board(board):
            return random.choice(moves)

        state = self._state_key(board)
        q = self.q_table

        if self.training and random.random() < self.epsilon:
            move = random.choice(moves)
        else:
            best_q = -math.inf
            best_moves: list[Move] = []
            for m in moves:
                v = q.get(self._sa_key(state, self._action_key(m)), 0.0)
                if v > best_q:
                    best_q = v
                    best_moves = [m]
                elif v == best_q:
                    best_moves.append(m)
            move = random.choice(best_moves)

        if self.training:
            self._history.append(self._sa_key(state, self._action_key(move)))

        return move

    # ---- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str, piece: Piece, **kwargs) -> "QLearningPlayer":
        with open(path, "rb") as f:
            q_table = pickle.load(f)
        return cls(piece=piece, q_table=q_table, **kwargs)


class DQNPlayer(Player):
    """Deep Q-Network player.

    Uses a CNN that takes a (7, 9, 9) board encoding and outputs Q-values
    for all 81 actions. Illegal actions are masked before argmax.

    The model is lazily loaded to keep torch out of the module-level imports,
    and to survive pickling across multiprocessing boundaries.
    """

    def __init__(
        self,
        piece: Piece,
        state_dict: dict | None = None,
        epsilon: float = 0.0,
        training: bool = False,
    ) -> None:
        super().__init__(piece)
        self.name = "DQN"
        self._state_dict = state_dict
        self._model = None  # built lazily
        self.epsilon = epsilon
        self.training = training

    # ---- lazy model ----------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return
        from dqn_model import DQNNet, DQNNetLegacy
        import torch

        # Auto-detect architecture from state dict keys
        if self._state_dict and "conv.0.weight" in self._state_dict:
            self._model = DQNNetLegacy()
        else:
            self._model = DQNNet()
        if self._state_dict:
            self._model.load_state_dict(self._state_dict)
        self._model.eval()

    # ---- move selection ------------------------------------------------------

    def _select_move(self, board: Board) -> Move | None:
        from dqn_model import encode_board, legal_mask, action_to_move
        import torch

        moves = board.legal_moves(self.piece)
        if not moves:
            return None

        if not self.training and _is_empty_board(board):
            return random.choice(moves)

        if self.training and random.random() < self.epsilon:
            return random.choice(moves)

        self._ensure_model()
        state = encode_board(board, self.piece)
        mask = legal_mask(board, self.piece)

        with torch.no_grad():
            q_values = self._model(state.unsqueeze(0)).squeeze(0)
        q_values = q_values + mask
        action = q_values.argmax().item()
        return action_to_move(action, self.piece)

    # ---- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        import torch

        self._ensure_model()
        torch.save(self._model.state_dict(), path)

    @classmethod
    def load(cls, path: str, piece: Piece, **kwargs) -> "DQNPlayer":
        import torch

        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        # Strip torch.compile prefix if present
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        return cls(piece=piece, state_dict=state_dict, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# AlphaZero
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AzNode:
    """Node in the AlphaZero MCTS tree.

    Children are keyed by action index (0..80). Edge stats are kept as
    parallel arrays for the actions present in `priors`.
    """

    turn: Piece
    priors: dict[int, float]  # action -> P(s,a)
    children: dict[int, "AzNode"] = field(default_factory=dict)
    N: dict[int, int] = field(default_factory=dict)  # visit counts
    W: dict[int, float] = field(default_factory=dict)  # cumulative value
    is_terminal: bool = False
    terminal_value: float = 0.0  # from current player's perspective


class AlphaZeroPlayer(Player):
    """AlphaZero-style player: PUCT-guided MCTS with policy+value network.

    No random rollouts — leaf evaluation uses the value head. Move is the
    most-visited child of the root.
    """

    def __init__(
        self,
        piece: Piece,
        state_dict: dict | None = None,
        num_simulations: int = 900,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.0,
        training: bool = False,
    ) -> None:
        super().__init__(piece)
        self.name = "AlphaZero"
        self._state_dict = state_dict
        self._model = None
        self._device = None
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.training = training
        self._last_root_pi: dict[int, float] | None = None

    # ---- lazy model ----------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return
        from alphazero_model import AlphaZeroNet
        import torch

        self._device = torch.device("cpu")
        num_blocks, channels = self._infer_architecture(self._state_dict)
        self._model = AlphaZeroNet(num_blocks=num_blocks, channels=channels)
        if self._state_dict:
            self._model.load_state_dict(self._state_dict)
        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def _infer_architecture(state_dict: dict | None) -> tuple[int, int]:
        if not state_dict:
            return 3, 32
        channels = state_dict["input_conv.0.weight"].shape[0]
        num_blocks = 1 + max(
            (int(k.split(".")[1]) for k in state_dict if k.startswith("res.")),
            default=-1,
        )
        return num_blocks, channels

    # ---- network evaluation --------------------------------------------------

    def _evaluate(self, board: Board, turn: Piece) -> tuple[dict[int, float], float]:
        """Run network on `board` from `turn`'s perspective.

        Returns (priors over legal actions, value in [-1, 1]).
        """
        from alphazero_model import encode_board
        from dqn_model import move_to_action
        import torch

        self._ensure_model()
        moves = board.legal_moves(turn)
        if not moves:
            return {}, 0.0

        state = encode_board(board, turn).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits, value = self._model(state)
        logits = logits.squeeze(0)
        value = float(value.item())

        legal_actions = [move_to_action(m) for m in moves]
        legal_logits = logits[legal_actions]
        probs = torch.softmax(legal_logits, dim=0).tolist()
        priors = {a: p for a, p in zip(legal_actions, probs)}
        return priors, value

    # ---- MCTS ----------------------------------------------------------------

    def _terminal_value(self, board: Board, turn: Piece) -> float | None:
        """If terminal, return value from `turn`'s perspective. Else None."""
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

    def _puct_select(self, node: AzNode) -> int:
        """Pick action with max PUCT score."""
        total_n = sum(node.N.values()) or 1
        sqrt_total = math.sqrt(total_n)
        best_a = -1
        best_score = -math.inf
        for a, p in node.priors.items():
            n = node.N.get(a, 0)
            q = (node.W.get(a, 0.0) / n) if n > 0 else 0.0
            u = self.c_puct * p * sqrt_total / (1 + n)
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def _make_child_node(self, board: Board, turn: Piece) -> tuple[AzNode, float]:
        term = self._terminal_value(board, turn)
        if term is not None:
            return AzNode(turn=turn, priors={}, is_terminal=True, terminal_value=term), term
        priors, value = self._evaluate(board, turn)
        return AzNode(turn=turn, priors=priors), value

    def _simulate(self, root: AzNode, root_board: Board) -> None:
        """One MCTS simulation from root."""
        from dqn_model import action_to_move

        path: list[tuple[AzNode, int]] = []
        undo_tokens: list = []
        board = root_board
        node = root
        turn = root.turn
        leaf_value: float | None = None

        try:
            # Selection: descend until we hit a leaf (no child for chosen action)
            while not node.is_terminal:
                if not node.priors:
                    break
                a = self._puct_select(node)
                path.append((node, a))
                token = board.make_move(action_to_move(a, turn))
                if token is None:
                    break
                undo_tokens.append(token)
                turn = swap_piece(turn)

                if a in node.children:
                    node = node.children[a]
                else:
                    child, leaf_value = self._make_child_node(board, turn)
                    node.children[a] = child
                    node = child
                    break

            # Backup: leaf value from leaf's perspective; flip per ply on the way up.
            if leaf_value is not None:
                value = leaf_value
            elif node.is_terminal:
                value = node.terminal_value
            else:
                value = 0.0

            # `value` is from the perspective of `turn` at the leaf.
            # The action was taken by the parent, whose turn was `swap(turn)`.
            v = -value
            for parent, a in reversed(path):
                parent.N[a] = parent.N.get(a, 0) + 1
                parent.W[a] = parent.W.get(a, 0.0) + v
                v = -v
        finally:
            for token in reversed(undo_tokens):
                board.undo_move(token)

    def _select_move(self, board: Board) -> Move | None:
        from dqn_model import action_to_move, move_to_action

        moves = board.legal_moves(self.piece)
        if not moves:
            return None

        if _is_empty_board(board):
            return random.choice(moves)

        # Build root and add Dirichlet exploration noise (training only)
        root, _ = self._make_child_node(board, self.piece)
        if not root.priors:
            return random.choice(moves)

        if self.training and self.dirichlet_eps > 0:
            actions = list(root.priors.keys())
            noise = [random.gammavariate(self.dirichlet_alpha, 1.0) for _ in actions]
            s = sum(noise) or 1.0
            noise = [n / s for n in noise]
            eps = self.dirichlet_eps
            for a, n in zip(actions, noise):
                root.priors[a] = (1 - eps) * root.priors[a] + eps * n

        for _ in range(self.num_simulations):
            self._simulate(root, board)

        # Build visit-count distribution π over legal actions.
        total_n = sum(root.N.values())
        if total_n == 0:
            return random.choice(moves)
        pi = {a: root.N.get(a, 0) / total_n for a in root.priors}
        self._last_root_pi = pi

        # Move selection
        if self.training and self.temperature > 0.1:
            counts = [root.N.get(a, 0) ** (1.0 / self.temperature) for a in root.priors]
            tot = sum(counts) or 1.0
            probs = [c / tot for c in counts]
            actions = list(root.priors.keys())
            r = random.random()
            acc = 0.0
            chosen = actions[-1]
            for a, p in zip(actions, probs):
                acc += p
                if r <= acc:
                    chosen = a
                    break
        else:
            chosen = max(root.N.items(), key=lambda kv: kv[1])[0]

        return action_to_move(chosen, self.piece)

    # ---- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        import torch

        self._ensure_model()
        torch.save(self._model.state_dict(), path)

    @classmethod
    def load(cls, path: str, piece: Piece, **kwargs) -> "AlphaZeroPlayer":
        import torch

        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if (
            isinstance(state_dict, dict)
            and "model" in state_dict
            and isinstance(state_dict["model"], dict)
        ):
            state_dict = state_dict["model"]
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        return cls(piece=piece, state_dict=state_dict, **kwargs)
