from __future__ import annotations
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Tuple
from dataclasses import dataclass, field
import math
import random

from board import (
    Board,
    BoardState,
    Move,
    Piece,
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


def evaluate_board(board: Board) -> float:
    "Heuristic eval"
    boards = [board[i][j] for i in range(3) for j in range(3)]
    heur = heuristics
    score = 0.0

    # Score utilities
    def inner_line_score(values: list[Piece]) -> float:
        """Two-in-a-row patterns inside an inner board."""
        x_count = values.count(Piece.X)
        o_count = values.count(Piece.O)
        empty_count = values.count(Piece.EMPTY)

        if x_count == 2 and o_count == 0 and empty_count == 1:
            return heur["two_in_row_inner"]
        if o_count == 2 and x_count == 0 and empty_count == 1:
            return -heur["two_in_row_inner"]
        return 0.0

    def outer_line_score(values: list[Piece]) -> float:
        """Two-in-a-row patterns across inner board results."""
        x_count = values.count(Piece.X)
        o_count = values.count(Piece.O)
        empty_count = values.count(Piece.EMPTY)

        if x_count == 2 and o_count == 0 and empty_count == 1:
            return heur["two_in_row_outer"]
        if o_count == 2 and x_count == 0 and empty_count == 1:
            return -heur["two_in_row_outer"]
        return 0.0

    important_positions = {(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)}

    # Outer board heuristic: based on inner board outcomes
    outer_values = [
        [board_state_to_piece(board[r][c].board_state) for c in range(3)]
        for r in range(3)
    ]

    for r in range(3):
        score += outer_line_score([outer_values[r][c] for c in range(3)])
        score += outer_line_score([outer_values[c][r] for c in range(3)])
    score += outer_line_score([outer_values[i][i] for i in range(3)])
    score += outer_line_score([outer_values[2 - i][i] for i in range(3)])

    # Reward controlling critical inner boards (outer center/corners)
    for r in range(3):
        for c in range(3):
            inner_board = board[r][c]
            value = board_state_to_piece(inner_board.board_state)
            if (r, c) in important_positions:
                if value == Piece.X:
                    score += heur["center_corner"]
                elif value == Piece.O:
                    score -= heur["center_corner"]

    # Evaluate each inner board
    for inner in boards:
        if inner.board_state == BoardState.X_WON:
            score += heur["inner_win"]
        elif inner.board_state == BoardState.O_WON:
            score -= heur["inner_win"]

        # Reward center/corner occupancy inside inner boards
        for r in range(3):
            for c in range(3):
                piece = inner[r][c]
                if piece == Piece.EMPTY:
                    continue
                if (r, c) in important_positions:
                    if piece == Piece.X:
                        score += heur["center_corner"]
                    elif piece == Piece.O:
                        score -= heur["center_corner"]

        if inner.board_state != BoardState.NOT_FINISHED:
            continue

        # Two-in-a-row patterns inside an unfinished inner board
        for r in range(3):
            row_values = [inner[r][c] for c in range(3)]
            col_values = [inner[c][r] for c in range(3)]
            score += inner_line_score(row_values)
            score += inner_line_score(col_values)

        score += inner_line_score([inner[i][i] for i in range(3)])
        score += inner_line_score([inner[2 - i][i] for i in range(3)])

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
        start = perf_counter()
        move = self._select_move(board)
        if self.name != "HumanPlayer":
            duration = perf_counter() - start
            self._move_count += 1
            self._move_time_total += duration
            avg = self._move_time_total / self._move_count

            print(
                f"{self.get_name()} move time: {duration:.3f}s "
                f"(avg {avg:.3f}s over {self._move_count} move{'s' if self._move_count != 1 else ''})"
            )
        return move

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
        if use_cache:
            features.append("cache")
        if features:
            self.name += " (" + ", ".join(features) + ")"

        self.use_heuristic_eval = use_heuristic_eval
        self.use_pruning = use_pruning
        self.use_cache = use_cache
        self._tt: dict[tuple, float] = {}
        self._tt_max = max_cache

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
        if self.use_cache:
            depth_remaining = depth_limit - depth
            tt_key = (board.key(piece), depth_remaining)
            if tt_key in self._tt:
                return self._tt[tt_key]

        # Terminal?
        match board.board_state:
            case BoardState.DRAW:
                value = heuristics["draw"]
                if self.use_cache and tt_key is not None:
                    if len(self._tt) >= self._tt_max:
                        self._tt.clear()
                    self._tt[tt_key] = value
                return value
            case BoardState.X_WON:
                value = heuristics["win"]
                if self.use_cache and tt_key is not None:
                    if len(self._tt) >= self._tt_max:
                        self._tt.clear()
                    self._tt[tt_key] = value
                return value
            case BoardState.O_WON:
                value = -heuristics["win"]
                if self.use_cache and tt_key is not None:
                    if len(self._tt) >= self._tt_max:
                        self._tt.clear()
                    self._tt[tt_key] = value
                return value

        # Depth limit
        if depth >= depth_limit:
            value = evaluate_board(board) if self.use_heuristic_eval else 0.0
            if self.use_cache and tt_key is not None:
                if len(self._tt) >= self._tt_max:
                    self._tt.clear()
                self._tt[tt_key] = value
            return value

        # Children
        moves = board.legal_moves(piece)
        if not moves:
            return 0.0

        maximizing = piece == Piece.X
        best = -math.inf if maximizing else math.inf

        for m in moves:
            token = board.make_move(m)
            if token is None:
                continue  # should not happen with legal_moves
            score = self._minimax(
                swap_piece(piece), board, depth + 1, depth_limit, alpha, beta
            )
            board.undo_move(token)

            if maximizing:
                if score > best:
                    best = score
                    if self.use_pruning and alpha is not None and beta is not None:
                        alpha = max(alpha, best)
                        if alpha >= beta:
                            break
            else:
                if score < best:
                    best = score
                    if self.use_pruning and alpha is not None and beta is not None:
                        beta = min(beta, best)
                        if beta <= alpha:
                            break

        if self.use_cache and tt_key is not None:
            if len(self._tt) >= self._tt_max:
                self._tt.clear()
            self._tt[tt_key] = best
        return best

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

        if self.piece == Piece.X and board.restriction is None:
            is_first_move = True
            for r in range(3):
                for c in range(3):
                    inner = board[r][c]
                    if (
                        inner.board_state != BoardState.NOT_FINISHED
                        or len(inner.empty_cells) != 9
                    ):
                        is_first_move = False
                        break
                if not is_first_move:
                    break
            if is_first_move:
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


@dataclass
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
        self.root = None
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

            for m, child in current.children.items():
                ucb = self._ucb(current, child)

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
        sim_board = node.board.clone()
        turn = node.turn

        while True:
            match sim_board.board_state:
                case BoardState.DRAW:
                    return 0
                case BoardState.X_WON:
                    return base
                case BoardState.O_WON:
                    return -base

            moves = sim_board.legal_moves(turn)
            if not moves:
                return 0

            if self.use_heuristics:
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
                move = random.choice(best_moves) if best_moves else random.choice(moves)
            else:
                move = random.choice(moves)

            token = sim_board.make_move(move)
            if token is None:
                return 0
            turn = swap_piece(turn)

    def _backprop(self, node: McNode, value: float) -> None:
        current = node
        while current is not None:
            current.n_visits += 1
            current.total_value += value
            current = current.parent

    def _select_move(self, board: Board) -> Move | None:
        moves = board.legal_moves(self.piece)
        if not moves:
            return None
        self.root = self._new_node(board.clone(), None, self.piece)

        for i in range(self.iter_nr):
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
        return best_move if best_move is not None else random.choice(moves)
