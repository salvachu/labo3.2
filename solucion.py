import numpy as np
from game_2048 import Game2048

class Agent:
    def __init__(self):
        self.game = Game2048()

        self.W_EMPTY = 250.0
        self.W_SMOOTH = -6.0
        self.W_CORNER = 800.0
        self.W_GRAD = 4.0
        self.W_MAXLOG = 80.0

        self.grads = [
            np.array([[15,14,13,12],
                      [ 8, 9,10,11],
                      [ 7, 6, 5, 4],
                      [ 0, 1, 2, 3]], dtype=np.float32),
            np.array([[12,13,14,15],
                      [11,10, 9, 8],
                      [ 4, 5, 6, 7],
                      [ 3, 2, 1, 0]], dtype=np.float32),
            np.array([[ 3, 2, 1, 0],
                      [ 4, 5, 6, 7],
                      [11,10, 9, 8],
                      [12,13,14,15]], dtype=np.float32),
            np.array([[ 0, 1, 2, 3],
                      [ 7, 6, 5, 4],
                      [ 8, 9,10,11],
                      [15,14,13,12]], dtype=np.float32),
        ]

        self.action_order = ("left", "up", "right", "down")

    def act(self, board, legal_actions) -> str:
        b = np.array(board, dtype=np.int64)

        best_a = legal_actions[0]
        best_v = -1e18

        ordered_legals = self._order_legals(legal_actions)

        for a in ordered_legals:
            nb, moved, reward = self._simulate(b, a)
            if not moved:
                continue
            v = reward + self._heuristic(nb)
            if v > best_v:
                best_v = v
                best_a = a

        return best_a

    def _simulate(self, board: np.ndarray, action: str):
        b2 = board.copy()
        moved, reward = self.game._apply_move(action, board=b2)
        return b2, moved, reward

    def _order_legals(self, legal_actions):
        s = set(legal_actions)
        ordered = [a for a in self.action_order if a in s]
        for a in legal_actions:
            if a not in ordered:
                ordered.append(a)
        return ordered

    def _heuristic(self, board: np.ndarray) -> float:
        empty = float(np.sum(board == 0))
        max_tile = int(board.max())

        logb = np.zeros_like(board, dtype=np.float32)
        nz = board > 0
        logb[nz] = np.log2(board[nz]).astype(np.float32)

        smooth = 0.0
        smooth += float(np.sum(np.abs(logb[:, 1:] - logb[:, :-1])))
        smooth += float(np.sum(np.abs(logb[1:, :] - logb[:-1, :])))

        corners = (board[0,0], board[0,3], board[3,0], board[3,3])
        corner_bonus = self.W_CORNER if max_tile in corners else 0.0

        grad_score = max(float(np.sum(g * logb)) for g in self.grads)
        maxlog = 0.0 if max_tile <= 0 else float(np.log2(max_tile))

        return (
            self.W_EMPTY * empty +
            self.W_SMOOTH * smooth +
            corner_bonus +
            self.W_GRAD * grad_score +
            self.W_MAXLOG * maxlog
        )