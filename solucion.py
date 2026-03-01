import numpy as np

# =========================================================
# 2048 bitboard agent
# - 64-bit board (4 bits por celda = exponente de 2)
# - tablas precomputadas de movimientos por fila
# - heurística rápida con tablas
# - lookahead selectivo + cache
# =========================================================


def _reverse_row(row: int) -> int:
    return (
        ((row & 0x000F) << 12)
        | ((row & 0x00F0) << 4)
        | ((row & 0x0F00) >> 4)
        | ((row & 0xF000) >> 12)
    )


def _transpose(board: int) -> int:
    # Transpose 4x4 nibbles (bit-hack correcto)
    a1 = board & 0xF0F00F0FF0F00F0F
    a2 = board & 0x0000F0F00000F0F0
    a3 = board & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


# ---------------------------------------------------------
# Tablas por fila (0..65535)
# Cada fila guarda 4 exponentes: 0 vacío, 1->2, 2->4, etc.
# ---------------------------------------------------------
ROW_LEFT = np.zeros(65536, dtype=np.uint16)
ROW_RIGHT = np.zeros(65536, dtype=np.uint16)
ROW_REWARD_LEFT = np.zeros(65536, dtype=np.int32)
ROW_REWARD_RIGHT = np.zeros(65536, dtype=np.int32)

ROW_EMPTY = np.zeros(65536, dtype=np.int8)
ROW_MAXEXP = np.zeros(65536, dtype=np.int8)
ROW_SMOOTH = np.zeros(65536, dtype=np.int16)
ROW_MERGES = np.zeros(65536, dtype=np.int8)
ROW_LIST = np.zeros((65536, 4), dtype=np.uint8)

# 4 snakes
_SNAKES = np.array(
    [
        [[15, 14, 13, 12], [8, 9, 10, 11], [7, 6, 5, 4], [0, 1, 2, 3]],
        [[12, 13, 14, 15], [11, 10, 9, 8], [4, 5, 6, 7], [3, 2, 1, 0]],
        [[3, 2, 1, 0], [4, 5, 6, 7], [11, 10, 9, 8], [12, 13, 14, 15]],
        [[0, 1, 2, 3], [7, 6, 5, 4], [8, 9, 10, 11], [15, 14, 13, 12]],
    ],
    dtype=np.int16,
)

ROW_GRAD_TABLE = np.zeros((4, 4, 65536), dtype=np.int16)

for row in range(65536):
    e0 = row & 0xF
    e1 = (row >> 4) & 0xF
    e2 = (row >> 8) & 0xF
    e3 = (row >> 12) & 0xF

    ROW_LIST[row, 0] = e0
    ROW_LIST[row, 1] = e1
    ROW_LIST[row, 2] = e2
    ROW_LIST[row, 3] = e3

    ROW_EMPTY[row] = int((e0 == 0) + (e1 == 0) + (e2 == 0) + (e3 == 0))
    ROW_MAXEXP[row] = max(e0, e1, e2, e3)
    ROW_SMOOTH[row] = abs(e0 - e1) + abs(e1 - e2) + abs(e2 - e3)
    ROW_MERGES[row] = int(
        ((e0 != 0) and (e0 == e1))
        + ((e1 != 0) and (e1 == e2))
        + ((e2 != 0) and (e2 == e3))
    )

    vals = []
    if e0:
        vals.append(e0)
    if e1:
        vals.append(e1)
    if e2:
        vals.append(e2)
    if e3:
        vals.append(e3)

    out = []
    reward = 0
    i = 0
    n = len(vals)

    while i < n:
        if i + 1 < n and vals[i] == vals[i + 1]:
            nv = vals[i] + 1
            out.append(nv)
            reward += (1 << nv)
            i += 2
        else:
            out.append(vals[i])
            i += 1

    while len(out) < 4:
        out.append(0)

    new_row = out[0] | (out[1] << 4) | (out[2] << 8) | (out[3] << 12)
    ROW_LEFT[row] = new_row
    ROW_REWARD_LEFT[row] = reward

    for ori in range(4):
        for ridx in range(4):
            w = _SNAKES[ori, ridx]
            ROW_GRAD_TABLE[ori, ridx, row] = (
                w[0] * e0 + w[1] * e1 + w[2] * e2 + w[3] * e3
            )

for row in range(65536):
    rr = _reverse_row(row)
    ROW_RIGHT[row] = _reverse_row(int(ROW_LEFT[rr]))
    ROW_REWARD_RIGHT[row] = ROW_REWARD_LEFT[rr]


# aliases rápidos
E = ROW_EMPTY
MX = ROW_MAXEXP
SM = ROW_SMOOTH
MG = ROW_MERGES

G00 = ROW_GRAD_TABLE[0, 0]
G01 = ROW_GRAD_TABLE[0, 1]
G02 = ROW_GRAD_TABLE[0, 2]
G03 = ROW_GRAD_TABLE[0, 3]

G10 = ROW_GRAD_TABLE[1, 0]
G11 = ROW_GRAD_TABLE[1, 1]
G12 = ROW_GRAD_TABLE[1, 2]
G13 = ROW_GRAD_TABLE[1, 3]

G20 = ROW_GRAD_TABLE[2, 0]
G21 = ROW_GRAD_TABLE[2, 1]
G22 = ROW_GRAD_TABLE[2, 2]
G23 = ROW_GRAD_TABLE[2, 3]

G30 = ROW_GRAD_TABLE[3, 0]
G31 = ROW_GRAD_TABLE[3, 1]
G32 = ROW_GRAD_TABLE[3, 2]
G33 = ROW_GRAD_TABLE[3, 3]


# ---------------------------------------------------------
# Moves sobre bitboard
# ---------------------------------------------------------
def _move_left(board: int):
    new_board = 0
    reward = 0
    for r in range(4):
        row = (board >> (16 * r)) & 0xFFFF
        nr = int(ROW_LEFT[row])
        reward += int(ROW_REWARD_LEFT[row])
        new_board |= nr << (16 * r)
    return new_board, reward


def _move_right(board: int):
    new_board = 0
    reward = 0
    for r in range(4):
        row = (board >> (16 * r)) & 0xFFFF
        nr = int(ROW_RIGHT[row])
        reward += int(ROW_REWARD_RIGHT[row])
        new_board |= nr << (16 * r)
    return new_board, reward


def _move_up(board: int):
    t = _transpose(board)
    mt, reward = _move_left(t)
    return _transpose(mt), reward


def _move_down(board: int):
    t = _transpose(board)
    mt, reward = _move_right(t)
    return _transpose(mt), reward


MOVE_FUNCS = {
    "left": _move_left,
    "right": _move_right,
    "up": _move_up,
    "down": _move_down,
}


# =========================================================
# Agent
# =========================================================
class Agent:
    def __init__(self, seed=None):
        self.action_order = ("left", "up", "right", "down")

        # -----------------------------
        # Heurística
        # -----------------------------
        self.W_EMPTY = 265.0
        self.W_SMOOTH = -7.0
        self.W_CORNER = 980.0
        self.W_GRAD = 4.6
        self.W_MAXLOG = 92.0
        self.W_MERGE = 40.0

        # -----------------------------
        # Lookahead selectivo
        # -----------------------------
        self.L2_WEIGHT = 0.62
        self.L2_EMPTY_TH = 8
        self.L2_TOPK = 3

        # Spawn expectation solo en críticos
        self.SPAWN_EMPTY_TH = 3
        self.SPAWN_SAMPLES = 2
        self.SPAWN_PROB2 = 0.9

        # caches
        self.eval_cache = {}
        self.local_best_cache = {}
        self.local_expect_cache = {}

    # -----------------------------------------------------
    # API requerida
    # -----------------------------------------------------
    def act(self, board, legal_actions) -> str:
        self.local_best_cache.clear()
        self.local_expect_cache.clear()

        b = self._board_from_array(np.asarray(board, dtype=np.int64))
        empties = self._empties(b)

        legal_set = set(legal_actions)
        ordered_legals = [a for a in self.action_order if a in legal_set]

        candidates = []
        for a in ordered_legals:
            nb, reward = MOVE_FUNCS[a](b)
            if nb == b:
                continue
            v1 = reward + self._heuristic(nb)
            candidates.append((v1, a, nb))

        if not candidates:
            return legal_actions[0]

        candidates.sort(key=lambda x: x[0], reverse=True)

        # tablero abierto -> 1-ply basta
        if empties > self.L2_EMPTY_TH:
            return candidates[0][1]

        best_a = candidates[0][1]
        best_v = -1e18

        topk = candidates[: min(self.L2_TOPK, len(candidates))]
        for v1, a, nb in topk:
            child_empties = self._empties(nb)
            if child_empties <= self.SPAWN_EMPTY_TH:
                v2 = self._expect_after_spawn(nb)
            else:
                v2 = self._best_next(nb)

            v = v1 + self.L2_WEIGHT * v2
            if v > best_v:
                best_v = v
                best_a = a

        return best_a

    # -----------------------------------------------------
    # Conversión board -> bitboard
    # -----------------------------------------------------
    def _board_from_array(self, board) -> int:
        b = 0
        k = 0
        for r in range(4):
            row = board[r]

            v = int(row[0])
            b |= ((0 if v == 0 else v.bit_length() - 1) & 0xF) << (4 * k)
            k += 1

            v = int(row[1])
            b |= ((0 if v == 0 else v.bit_length() - 1) & 0xF) << (4 * k)
            k += 1

            v = int(row[2])
            b |= ((0 if v == 0 else v.bit_length() - 1) & 0xF) << (4 * k)
            k += 1

            v = int(row[3])
            b |= ((0 if v == 0 else v.bit_length() - 1) & 0xF) << (4 * k)
            k += 1

        return b

    def _empties(self, b: int) -> int:
        return (
            int(E[b & 0xFFFF])
            + int(E[(b >> 16) & 0xFFFF])
            + int(E[(b >> 32) & 0xFFFF])
            + int(E[(b >> 48) & 0xFFFF])
        )

    # -----------------------------------------------------
    # Heurística rápida
    # -----------------------------------------------------
    def _heuristic(self, b: int) -> float:
        cached = self.eval_cache.get(b)
        if cached is not None:
            return cached

        r0 = b & 0xFFFF
        r1 = (b >> 16) & 0xFFFF
        r2 = (b >> 32) & 0xFFFF
        r3 = (b >> 48) & 0xFFFF

        t = _transpose(b)
        c0 = t & 0xFFFF
        c1 = (t >> 16) & 0xFFFF
        c2 = (t >> 32) & 0xFFFF
        c3 = (t >> 48) & 0xFFFF

        empty = int(E[r0]) + int(E[r1]) + int(E[r2]) + int(E[r3])
        maxe = max(int(MX[r0]), int(MX[r1]), int(MX[r2]), int(MX[r3]))

        smooth = (
            int(SM[r0])
            + int(SM[r1])
            + int(SM[r2])
            + int(SM[r3])
            + int(SM[c0])
            + int(SM[c1])
            + int(SM[c2])
            + int(SM[c3])
        )

        merges = (
            int(MG[r0])
            + int(MG[r1])
            + int(MG[r2])
            + int(MG[r3])
            + int(MG[c0])
            + int(MG[c1])
            + int(MG[c2])
            + int(MG[c3])
        )

        # corners
        e0 = r0 & 0xF
        e3 = (r0 >> 12) & 0xF
        e12 = r3 & 0xF
        e15 = (r3 >> 12) & 0xF
        corner_bonus = self.W_CORNER if maxe in (e0, e3, e12, e15) else 0.0

        # best snake
        g0 = int(G00[r0]) + int(G01[r1]) + int(G02[r2]) + int(G03[r3])
        g1 = int(G10[r0]) + int(G11[r1]) + int(G12[r2]) + int(G13[r3])
        g2 = int(G20[r0]) + int(G21[r1]) + int(G22[r2]) + int(G23[r3])
        g3 = int(G30[r0]) + int(G31[r1]) + int(G32[r2]) + int(G33[r3])
        grad = max(g0, g1, g2, g3)

        val = (
            self.W_EMPTY * empty
            + self.W_SMOOTH * smooth
            + corner_bonus
            + self.W_GRAD * grad
            + self.W_MAXLOG * maxe
            + self.W_MERGE * merges
        )

        if len(self.eval_cache) > 400000:
            self.eval_cache.clear()

        self.eval_cache[b] = val
        return val

    # -----------------------------------------------------
    # 1-step best next
    # -----------------------------------------------------
    def _best_next(self, b: int) -> float:
        cached = self.local_best_cache.get(b)
        if cached is not None:
            return cached

        best = -1e18
        for a in self.action_order:
            nb, reward = MOVE_FUNCS[a](b)
            if nb == b:
                continue
            v = reward + self._heuristic(nb)
            if v > best:
                best = v

        if best == -1e18:
            best = self._heuristic(b)

        self.local_best_cache[b] = best
        return best

    # -----------------------------------------------------
    # Expectation sobre spawn (solo crítico)
    # -----------------------------------------------------
    def _expect_after_spawn(self, b: int) -> float:
        cached = self.local_expect_cache.get(b)
        if cached is not None:
            return cached

        empties = []
        for idx in range(16):
            if ((b >> (4 * idx)) & 0xF) == 0:
                empties.append(idx)

        n = len(empties)
        if n == 0:
            v = self._best_next(b)
            self.local_expect_cache[b] = v
            return v

        if n > self.SPAWN_SAMPLES:
            # muestreo determinista espaciado
            inds = []
            seen = set()
            for i in range(self.SPAWN_SAMPLES):
                idx = empties[round(i * (n - 1) / (self.SPAWN_SAMPLES - 1))]
                if idx not in seen:
                    inds.append(idx)
                    seen.add(idx)
        else:
            inds = empties

        total = 0.0
        p2 = self.SPAWN_PROB2

        for idx in inds:
            shift = 4 * idx
            total += p2 * self._best_next(b | (1 << shift))
            total += (1.0 - p2) * self._best_next(b | (2 << shift))

        v = total / len(inds)
        self.local_expect_cache[b] = v
        return v
