#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inertia Solver  --  AI search project (uninformed vs. informed search)
=====================================================================

Puzzle: "Inertia" from Simon Tatham's Portable Puzzle Collection.
The ball slides in one of 8 directions until it hits a wall / the border or
lands on a STOP square.  Goal: collect every gem without touching a mine.

State  : (ball position, set of collected gems)  ->  one integer
         state = collected_mask * (W*H) + position
Action : one of 8 directions (numeric-keypad digits 7 8 9 / 4 6 / 1 2 3)
Cost   : 1 per move (default)   or   number of cells travelled (--cost cells)
Goal   : every gem collected

Algorithms
  uninformed : bfs, ucs, dfs, dls, ids, ils
  informed   : astar  (consistent+admissible heuristic: exact maximum set of
               pairwise "non collinear" gems, see GemHeuristic)

Pure standard library (Python >= 3.8).  GUI (optional) lives in inertia_gui.py.

Quick start
  python inertia_solver.py samples/sample_10x8.txt
  python inertia_solver.py my_save_file.sav --algo bfs,ids,astar
  python inertia_solver.py --benchmark 10 --size 8x6 --algo bfs,ids,astar
  python inertia_solver.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import gc
import heapq
import os
import platform
import random
import re
import sys
import time
from collections import deque

__version__ = "1.0"

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
BLANK, GEM, MINE, STOP, WALL, START = "b", "g", "m", "s", "w", "S"
VALID_CHARS = set("bgmswS")

# index -> (dx, dy)   (y grows downward, like the screen)
DIRS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
DIR_NAMES = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"]
DIR_KEYS = ["7", "8", "9", "4", "6", "1", "2", "3"]      # numeric keypad
KEY_TO_DIR = {k: i for i, k in enumerate(DIR_KEYS)}

ALL_ALGOS = ["bfs", "ucs", "dfs", "dls", "ids", "ils", "astar"]
ALGO_TITLES = {
    "bfs": "BFS", "ucs": "UCS", "dfs": "DFS", "dls": "DLS",
    "ids": "IDS", "ils": "ILS", "astar": "A*",
}

popcount = int.bit_count if hasattr(int, "bit_count") else (lambda x: bin(x).count("1"))


class PuzzleError(ValueError):
    """Raised for any malformed / unsupported puzzle input."""


# --------------------------------------------------------------------------
# Puzzle model
# --------------------------------------------------------------------------
class Puzzle:
    """Immutable Inertia board + pre-computed move table."""

    def __init__(self, w, h, cells, start_is_stop=True):
        cells = list(cells)
        if w < 1 or h < 1:
            raise PuzzleError("board size must be positive")
        if len(cells) != w * h:
            raise PuzzleError(f"expected {w * h} cells for a {w}x{h} board, got {len(cells)}")
        bad = sorted(set(cells) - VALID_CHARS)
        if bad:
            raise PuzzleError(f"unknown cell character(s): {bad}")
        starts = [i for i, c in enumerate(cells) if c == START]
        if len(starts) != 1:
            raise PuzzleError(f"board must contain exactly one start square 'S' (found {len(starts)})")

        self.w, self.h, self.n = w, h, w * h
        self.cells = "".join(cells)                 # raw description (with 'S')
        self.start_is_stop = bool(start_is_stop)
        self.start = starts[0]

        grid = list(cells)                          # working grid
        grid[self.start] = STOP if self.start_is_stop else BLANK
        self.grid = grid
        self.gems = [i for i, c in enumerate(grid) if c == GEM]
        self.gem_index = {p: k for k, p in enumerate(self.gems)}
        self.num_gems = len(self.gems)
        self.full = (1 << self.num_gems) - 1

        # moves_from[pos] = [(dir, end_pos, gem_mask, cells_travelled), ...]
        self.moves_from = [[] for _ in range(self.n)]
        for pos in range(self.n):
            if grid[pos] in (WALL, MINE):
                continue
            for d in range(8):
                st, end, gm, cnt, _ = self.simulate(pos, d)
                if st == "ok":
                    self.moves_from[pos].append((d, end, gm, cnt))

    # -- basic helpers ----------------------------------------------------
    def xy(self, pos):
        return pos % self.w, pos // self.w

    def game_id(self):
        return f"{self.w}x{self.h}:{self.cells}"

    def simulate(self, pos, d):
        """Slide from `pos` in direction `d`.
        Returns (status, end_pos, gem_mask, cells, visited_positions);
        status is 'ok' | 'blocked' (cannot move) | 'dead' (hits a mine)."""
        w, h, grid = self.w, self.h, self.grid
        dx, dy = DIRS[d]
        x, y = pos % w, pos // w
        mask, cnt, visited = 0, 0, []
        while True:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                break
            c = grid[ny * w + nx]
            if c == WALL:
                break
            x, y = nx, ny
            cnt += 1
            visited.append(y * w + x)
            if c == MINE:
                return "dead", y * w + x, mask, cnt, visited
            if c == GEM:
                mask |= 1 << self.gem_index[y * w + x]
            if c == STOP:
                break
        if cnt == 0:
            return "blocked", pos, 0, 0, visited
        return "ok", y * w + x, mask, cnt, visited

    def unreachable_gems(self):
        """Bitmask of gems that no sequence of moves can ever collect
        (position graph is independent of the gem set -> exact test)."""
        seen = {self.start}
        stack = [self.start]
        got = 0
        while stack:
            pos = stack.pop()
            for _d, end, gm, _c in self.moves_from[pos]:
                got |= gm
                if end not in seen:
                    seen.add(end)
                    stack.append(end)
        return self.full & ~got

    def ascii(self, trail=None):
        sym = {BLANK: ".", GEM: "*", MINE: "X", STOP: "o", WALL: "#", START: "@"}
        trail = set(trail or ())
        rows = []
        for y in range(self.h):
            row = []
            for x in range(self.w):
                i = y * self.w + x
                ch = self.cells[i]
                s = sym[ch]
                if i in trail and ch in (BLANK,):
                    s = ","
                row.append(s)
            rows.append(" ".join(row))
        return "\n".join(rows)


# --------------------------------------------------------------------------
# Parsing  (Tatham save-file, game id, plain grid, bare DESC string)
# --------------------------------------------------------------------------
_ALIASES = {
    "b": "b", "B": "b", ".": "b", "_": "b", "-": "b",
    "g": "g", "G": "g", "*": "g",
    "m": "m", "M": "m", "x": "m", "X": "m",
    "s": "s", "o": "s", "O": "s", "0": "s",
    "w": "w", "W": "w", "#": "w",
    "S": "S", "@": "S",
}


def _infer_size(n, size):
    if size:
        if size[0] * size[1] != n:
            raise PuzzleError(f"--size {size[0]}x{size[1]} needs {size[0] * size[1]} cells but the description has {n}")
        return size
    if n == 80:
        return 10, 8                      # Tatham's default Inertia size
    cands = [(w, n // w) for w in range(2, n) if n % w == 0 and n // w >= 2]
    if len(cands) == 1:
        return cands[0]
    raise PuzzleError(f"cannot infer board shape from {n} cells; pass --size WxH")


def parse_size(text):
    m = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", text or "")
    if not m:
        raise PuzzleError(f"bad size '{text}', expected WxH (e.g. 10x8)")
    w, h = int(m.group(1)), int(m.group(2))
    if w < 2 or h < 2:
        raise PuzzleError("size must be at least 2x2")
    return w, h


def parse_puzzle(text, size=None, start_is_stop=True):
    """Return a Puzzle from any supported textual form."""
    text = (text or "").lstrip("\ufeff").replace("\x00", "")
    if not text.strip():
        raise PuzzleError("empty puzzle input")

    # 1) Tatham save file ("DESC    :80:....")
    m_desc = re.search(r"^DESC\s*:\s*(\d+)\s*:([^\r\n]*)", text, re.M)
    if m_desc or re.search(r"^SAVEFILE\s*:", text, re.M):
        m_game = re.search(r"^GAME\s*:\s*\d+\s*:([^\r\n]*)", text, re.M)
        if m_game and "inertia" not in m_game.group(1).lower():
            raise PuzzleError(f"this save file belongs to game '{m_game.group(1).strip()}', not Inertia")
        if not m_desc:
            raise PuzzleError("save file has no DESC section")
        desc = m_desc.group(2)[: int(m_desc.group(1))].strip()
        m_par = re.search(r"^PARAMS\s*:\s*\d+\s*:\s*(\d+)\s*x\s*(\d+)", text, re.M)
        if m_par:
            w, h = int(m_par.group(1)), int(m_par.group(2))
            if size and size != (w, h):
                raise PuzzleError("--size disagrees with PARAMS in the save file")
        else:
            w, h = _infer_size(len(desc), size)
        return Puzzle(w, h, _check_desc(desc, w, h), start_is_stop)

    t = text.strip()
    # 2) game id  "10x8:bwbm..."
    m = re.fullmatch(r"(\d+)\s*x\s*(\d+)\s*:\s*([A-Za-z]+)", t)
    if m:
        w, h, desc = int(m.group(1)), int(m.group(2)), m.group(3)
        return Puzzle(w, h, _check_desc(desc, w, h), start_is_stop)
    if re.match(r"\d+\s*x\s*\d+\s*#", t):
        raise PuzzleError("random-seed ids (WxH#seed) cannot be decoded; export the DESC / save file instead")

    # 3) multi-line grid
    lines = [ln.rstrip("\r\n") for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        lines = [ln.strip() for ln in lines]
        compact = [ln.replace(" ", "") for ln in lines]
        if len({len(c) for c in compact}) != 1:
            raise PuzzleError("grid rows have different lengths")
        cells = []
        for r, row in enumerate(compact):
            for c, ch in enumerate(row):
                if ch not in _ALIASES:
                    raise PuzzleError(f"unknown symbol {ch!r} at row {r + 1}, column {c + 1}")
                cells.append(_ALIASES[ch])
        if size and size != (len(compact[0]), len(compact)):
            raise PuzzleError("--size disagrees with the grid dimensions")
        return Puzzle(len(compact[0]), len(compact), cells, start_is_stop)

    # 4) bare description string
    desc = re.sub(r"\s+", "", t)
    w, h = _infer_size(len(desc), size)
    return Puzzle(w, h, _check_desc(desc, w, h), start_is_stop)


def _check_desc(desc, w, h):
    if len(desc) != w * h:
        raise PuzzleError(f"description has {len(desc)} cells but {w}x{h} needs {w * h}")
    bad = sorted(set(desc) - VALID_CHARS)
    if bad:
        raise PuzzleError(f"description contains unexpected characters: {bad}")
    return list(desc)


def read_text_file(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-16", "cp1256", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("latin-1", errors="replace")


def parse_moves(text):
    """'6 2 3,1' / '623' / 'E S SE' -> [4, 6, 7, ...]"""
    toks = re.findall(r"[1-9]|NW|NE|SW|SE|N|S|E|W", (text or "").upper())
    out = []
    for t in toks:
        if t in DIR_NAMES:
            out.append(DIR_NAMES.index(t))
        elif t in KEY_TO_DIR:
            out.append(KEY_TO_DIR[t])
        elif t == "5":
            raise PuzzleError("key 5 is not a direction")
    return out


# --------------------------------------------------------------------------
# Independent verification (does NOT reuse Puzzle.simulate / move tables)
# --------------------------------------------------------------------------
def verify_solution(puzzle, moves):
    raw, w, h = puzzle.cells, puzzle.w, puzzle.h
    idx = raw.index(START)
    y, x = divmod(idx, w)
    got = set()
    for k, d in enumerate(moves, 1):
        if not isinstance(d, int) or not 0 <= d < 8:
            return False, f"move {k}: invalid direction {d!r}"
        dx, dy = DIRS[d]
        steps = 0
        while True:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                break
            c = raw[ny * w + nx]
            if c == WALL:
                break
            x, y, steps = nx, ny, steps + 1
            if c == MINE:
                return False, f"move {k} ({DIR_KEYS[d]}) runs into a mine at ({x},{y})"
            if c == GEM:
                got.add((x, y))
            if c == STOP or (c == START and puzzle.start_is_stop):
                break
        if steps == 0:
            return False, f"move {k} ({DIR_KEYS[d]}) is blocked: the ball cannot move"
    total = raw.count(GEM)
    if len(got) != total:
        return False, f"{total - len(got)} of {total} gems are still uncollected"
    return True, f"OK: {len(moves)} moves collect all {total} gems, no mine touched"


def describe_moves(puzzle, moves):
    """Human readable step-by-step trace."""
    pos, mask, lines = puzzle.start, 0, []
    for k, d in enumerate(moves, 1):
        st, end, gm, cnt, _ = puzzle.simulate(pos, d)
        if st != "ok":
            lines.append(f"{k:3d}. {DIR_KEYS[d]} ({DIR_NAMES[d]}): INVALID ({st})")
            break
        new = popcount(gm & ~mask)
        mask |= gm
        x0, y0 = puzzle.xy(pos)
        x1, y1 = puzzle.xy(end)
        lines.append(f"{k:3d}. press {DIR_KEYS[d]} ({DIR_NAMES[d]:>2}): ({x0},{y0}) -> ({x1},{y1})  "
                     f"{cnt} cells, +{new} gem(s)  [{popcount(mask)}/{puzzle.num_gems}]")
        pos = end
    return "\n".join(lines)



# --------------------------------------------------------------------------
# Interactive game state (used by the GUI; testable without any GUI)
# --------------------------------------------------------------------------
class PlayState:
    """Manual play with undo.  press(d) -> (status, visited_positions)."""

    def __init__(self, puzzle):
        self.p = puzzle
        self.reset()

    def reset(self):
        self.pos, self.mask, self.dead = self.p.start, 0, False
        self.history, self.moves, self.trail = [], [], {self.p.start}

    @property
    def won(self):
        return (not self.dead) and self.mask == self.p.full

    def press(self, d):
        if self.dead or self.won:
            return "over", []
        st, end, gm, _cnt, visited = self.p.simulate(self.pos, d)
        if st == "blocked":
            return st, []
        self.history.append((self.pos, self.mask, self.dead, set(self.trail)))
        self.moves.append(d)
        self.trail.update(visited)
        self.pos = end
        if st == "dead":
            self.dead = True
        else:
            self.mask |= gm
        return st, visited

    def undo(self):
        if not self.history:
            return False
        self.pos, self.mask, self.dead, self.trail = self.history.pop()
        self.moves.pop()
        return True

# --------------------------------------------------------------------------
# Heuristic
# --------------------------------------------------------------------------
class GemHeuristic:
    """h(state) = size of the largest set of remaining gems that are pairwise
    NOT collinear (not on a common row / column / diagonal).

    Admissible : one move travels along a single straight line, so it can only
                 collect gems that are mutually collinear; hence every gem of such
                 an 'independent' set needs its own move  (>= 1 move each, and
                 every move costs >= 1 under both cost models).
    Consistent : a move collects collinear gems only; an independent set contains
                 at most one of them, so MIS(G) <= MIS(G minus collected) + 1
                 <= c(n,n') + h(n').   (we compute the *exact* MIS -> consistent)
    """

    EXACT_LIMIT = 40          # above this many gems fall back to h in {0,1}

    def __init__(self, puzzle):
        self.n = puzzle.n
        self.full = puzzle.full
        g = puzzle.gems
        self.exact = len(g) <= self.EXACT_LIMIT
        self.conf = []
        for i, a in enumerate(g):
            ax, ay = puzzle.xy(a)
            m = 0
            for j, b in enumerate(g):
                if i == j:
                    continue
                bx, by = puzzle.xy(b)
                dx, dy = abs(ax - bx), abs(ay - by)
                if dx == 0 or dy == 0 or dx == dy:
                    m |= 1 << j
            self.conf.append(m)
        self.memo = {}

    def mis(self, mask):
        if mask == 0:
            return 0
        r = self.memo.get(mask)
        if r is not None:
            return r
        conf = self.conf
        best_v, best_d = -1, -1
        m = mask
        while m:
            low = m & -m
            m ^= low
            v = low.bit_length() - 1
            d = popcount(conf[v] & mask)
            if d <= 1:                     # degree 0/1 vertex: always safe to take
                r = 1 + self.mis(mask & ~(conf[v] | low))
                self.memo[mask] = r
                return r
            if d > best_d:
                best_d, best_v = d, v
        v = best_v
        bit = 1 << v
        r = max(self.mis(mask & ~bit), 1 + self.mis(mask & ~(conf[v] | bit)))
        if len(self.memo) > 2_000_000:
            self.memo.clear()
        self.memo[mask] = r
        return r

    def __call__(self, state):
        rem = self.full & ~(state // self.n)
        if not rem:
            return 0
        if not self.exact:
            return 1
        return self.mis(rem)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
class LimitExceeded(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Limits:
    def __init__(self, time_limit=60.0, max_nodes=5_000_000, max_stored=3_000_000):
        self.time_limit = time_limit
        self.max_nodes = max_nodes
        self.max_stored = max_stored


class Result:
    def __init__(self, algo, label):
        self.algo, self.label = algo, label
        self.status = "NO_SOLUTION"      # SOLVED | NO_SOLUTION | LIMIT
        self.reason = ""
        self.moves = None
        self.cells = None
        self.expanded = self.generated = self.peak = 0
        self.time_ms = 0.0
        self.extra = {}
        self.verified = None             # True / False / None

    @property
    def n_moves(self):
        return None if self.moves is None else len(self.moves)

    def keys(self):
        return "".join(DIR_KEYS[d] for d in self.moves) if self.moves is not None else ""


class Solver:
    def __init__(self, puzzle, cost="moves", limits=None):
        if cost not in ("moves", "cells"):
            raise ValueError("cost must be 'moves' or 'cells'")
        self.p = puzzle
        self.cost = cost
        self.use_cells = cost == "cells"
        self.limits = limits or Limits()
        self.n = puzzle.n
        self.start = puzzle.start                     # mask 0 -> state == position
        self.goal_min = puzzle.full * puzzle.n        # state >= goal_min  <=>  mask == full
        self.h = GemHeuristic(puzzle)
        self._reset()

    # -- bookkeeping ------------------------------------------------------
    def _reset(self):
        self.expanded = self.generated = self.peak = 0
        self.extra = {}
        self._deadline = float("inf")

    def _tick(self, stored):
        self.expanded += 1
        if stored > self.peak:
            self.peak = stored
        lim = self.limits
        if self.expanded > lim.max_nodes:
            raise LimitExceeded(f"expanded-node limit reached ({lim.max_nodes:,})")
        if stored > lim.max_stored:
            raise LimitExceeded(f"stored-node (memory) limit reached ({lim.max_stored:,})")
        if not (self.expanded & 255) and time.perf_counter() > self._deadline:
            raise LimitExceeded(f"time limit reached ({lim.time_limit:g}s)")

    def _succ(self, s):
        n = self.n
        pos = s % n
        mask = s // n
        return [(d, (mask | gm) * n + end, c) for d, end, gm, c in self.p.moves_from[pos]]

    def _rebuild(self, parent, s):
        out = []
        while parent[s] is not None:
            s, d = parent[s]
            out.append(d)
        out.reverse()
        return out

    def path_cells(self, moves):
        pos, total = self.start, 0
        for d in moves:
            st, end, _gm, c, _v = self.p.simulate(pos, d)
            if st != "ok":
                return None
            total += c
            pos = end
        return total

    # -- public API -------------------------------------------------------
    def solve(self, algo, weight=1.0, depth_limit=25, tt=False, precheck=True):
        algo = algo.lower()
        if algo not in ALL_ALGOS:
            raise ValueError(f"unknown algorithm '{algo}' (choose from {', '.join(ALL_ALGOS)})")
        label = ALGO_TITLES[algo]
        if algo == "astar" and weight != 1.0:
            label = f"A*(w={weight:g})"
        if algo in ("dls",):
            label = f"DLS(L={depth_limit})"
        if algo in ("dls", "ids", "ils") and tt:
            label += "+TT"
        res = Result(algo, label)
        self._reset()

        if precheck:
            bad = self.p.unreachable_gems()
            if bad:
                pos = [self.p.xy(self.p.gems[i]) for i in range(self.p.num_gems) if bad >> i & 1]
                res.reason = f"pre-analysis: gem(s) at {pos} can never be collected"
                return res

        fn = {
            "bfs": self._bfs, "ucs": self._ucs, "dfs": self._dfs,
            "dls": lambda: self._dls(depth_limit, tt), "ids": lambda: self._ids(tt),
            "ils": lambda: self._ils(tt), "astar": lambda: self._astar(weight),
        }[algo]

        was = gc.isenabled()
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        self._deadline = t0 + self.limits.time_limit
        try:
            if self.p.full == 0:
                path = []
            else:
                path = fn()
            if path is None:
                res.status, res.reason = "NO_SOLUTION", "search space exhausted - no solution exists"
            else:
                res.status, res.reason, res.moves = "SOLVED", "goal reached", path
        except LimitExceeded as e:
            res.status, res.reason = "LIMIT", e.reason
        finally:
            res.time_ms = (time.perf_counter() - t0) * 1000.0
            if was:
                gc.enable()
        res.expanded, res.generated, res.peak = self.expanded, self.generated, self.peak
        res.extra = dict(self.extra)
        if res.moves is not None:
            res.cells = self.path_cells(res.moves)
            res.verified = verify_solution(self.p, res.moves)[0]
        return res

    # -- uninformed: breadth-first ---------------------------------------
    def _bfs(self):
        start, goal_min = self.start, self.goal_min
        parent = {start: None}
        q = deque([start])
        while q:
            s = q.popleft()
            self._tick(len(parent))
            for d, ns, _c in self._succ(s):
                self.generated += 1
                if ns in parent:
                    continue
                parent[ns] = (s, d)
                if ns >= goal_min:
                    self.peak = max(self.peak, len(parent))
                    return self._rebuild(parent, ns)
                q.append(ns)
        return None

    # -- uninformed: uniform cost -----------------------------------------
    def _ucs(self):
        start, goal_min, use_cells = self.start, self.goal_min, self.use_cells
        g = {start: 0}
        parent = {start: None}
        heap = [(0, 0, start)]
        cnt = 0
        while heap:
            gs, _, s = heapq.heappop(heap)
            if gs > g[s]:
                continue
            if s >= goal_min:
                return self._rebuild(parent, s)
            self._tick(len(g))
            for d, ns, c in self._succ(s):
                self.generated += 1
                ng = gs + (c if use_cells else 1)
                old = g.get(ns)
                if old is None or ng < old:
                    g[ns] = ng
                    parent[ns] = (s, d)
                    cnt += 1
                    heapq.heappush(heap, (ng, cnt, ns))
        return None

    # -- uninformed: depth-first (graph search, visited set) ---------------
    def _dfs(self):
        start, goal_min = self.start, self.goal_min
        parent = {start: None}
        stack = [start]
        while stack:
            s = stack.pop()
            self._tick(len(parent))
            for d, ns, _c in reversed(self._succ(s)):
                self.generated += 1
                if ns in parent:
                    continue
                parent[ns] = (s, d)
                if ns >= goal_min:
                    self.peak = max(self.peak, len(parent))
                    return self._rebuild(parent, ns)
                stack.append(ns)
        return None

    # -- bounded DFS used by DLS / IDS / ILS --------------------------------
    def _bounded_dfs(self, limit, use_cost, tt):
        goal_min = self.goal_min
        start = self.start
        states, dirs, gs = [start], [], [0]
        its = [iter(self._succ(start))]
        onpath = {start}
        cutoff, min_exceed = False, None
        while states:
            try:
                d, ns, c = next(its[-1])
            except StopIteration:
                onpath.discard(states.pop())
                its.pop()
                gs.pop()
                if dirs:
                    dirs.pop()
                continue
            self.generated += 1
            if not (self.generated & 4095) and time.perf_counter() > self._deadline:
                raise LimitExceeded(f"time limit reached ({self.limits.time_limit:g}s)")
            ng = gs[-1] + (c if use_cost else 1)
            if ng > limit:
                cutoff = True
                if min_exceed is None or ng < min_exceed:
                    min_exceed = ng
                continue
            if ns in onpath:
                continue
            if tt is not None:
                old = tt.get(ns)
                if old is not None and old <= ng:
                    continue
                tt[ns] = ng
            if ns >= goal_min:
                return dirs + [d], cutoff, min_exceed
            self._tick(len(states) + 1 + (len(tt) if tt is not None else 0))
            states.append(ns)
            dirs.append(d)
            gs.append(ng)
            its.append(iter(self._succ(ns)))
            onpath.add(ns)
        return None, cutoff, min_exceed

    def _dls(self, depth_limit, tt):
        res, cut, _ = self._bounded_dfs(depth_limit, False, {} if tt else None)
        if res is not None:
            return res
        if cut:
            raise LimitExceeded(f"depth limit {depth_limit} reached without finding a solution (cutoff)")
        return None

    def _ids(self, tt):
        limit, it = 0, 0
        while True:
            it += 1
            self.extra["iterations"] = it
            self.extra["last_depth"] = limit
            res, cut, _ = self._bounded_dfs(limit, False, {} if tt else None)
            if res is not None:
                return res
            if not cut:
                return None
            limit += 1

    def _ils(self, tt):
        """Iterative lengthening: iterative deepening on PATH COST."""
        limit, it = 0, 0
        while True:
            it += 1
            self.extra["iterations"] = it
            self.extra["last_cost_limit"] = limit
            res, cut, nxt = self._bounded_dfs(limit, self.use_cells, {} if tt else None)
            if res is not None:
                return res
            if not cut:
                return None
            limit = nxt

    # -- informed: A* ---------------------------------------------------------
    def _astar(self, weight):
        start, goal_min, use_cells, h = self.start, self.goal_min, self.use_cells, self.h
        g = {start: 0}
        parent = {start: None}
        h0 = h(start)
        heap = [(weight * h0, h0, 0, 0, start)]       # (f, h, tie, g, state)
        cnt = 0
        while heap:
            _f, _h, _t, gs, s = heapq.heappop(heap)
            if gs > g[s]:
                continue
            if s >= goal_min:
                return self._rebuild(parent, s)
            self._tick(len(g))
            for d, ns, c in self._succ(s):
                self.generated += 1
                ng = gs + (c if use_cells else 1)
                old = g.get(ns)
                if old is None or ng < old:
                    g[ns] = ng
                    parent[ns] = (s, d)
                    hn = h(ns)
                    cnt += 1
                    heapq.heappush(heap, (ng + weight * hn, hn, cnt, ng, ns))
        return None


# --------------------------------------------------------------------------
# Random puzzle generation (always solvable)
# --------------------------------------------------------------------------
def generate_puzzle(w, h, seed=0, gems=None, mines=None, walls=None, stops=None,
                    start_is_stop=True, tries=2000):
    n = w * h
    gems = max(3, round(n * 0.12)) if gems is None else gems
    mines = round(n * 0.12) if mines is None else mines
    walls = round(n * 0.10) if walls is None else walls
    stops = round(n * 0.14) if stops is None else stops
    if gems < 1 or gems + mines + walls + stops + 1 > n:
        raise PuzzleError("too many special cells for this board size")
    rng = random.Random(seed)
    lim = Limits(time_limit=3.0, max_nodes=300_000, max_stored=600_000)
    for _ in range(tries):
        cells = [BLANK] * n
        order = list(range(n))
        rng.shuffle(order)
        it = iter(order)
        for t, k in ((GEM, gems), (MINE, mines), (WALL, walls), (STOP, stops)):
            for _i in range(k):
                cells[next(it)] = t
        cells[next(it)] = START
        p = Puzzle(w, h, cells, start_is_stop)
        if p.unreachable_gems():
            continue
        if Solver(p, limits=lim).solve("astar", precheck=False).status == "SOLVED":
            return p
    raise PuzzleError("could not generate a solvable puzzle with these parameters")


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------
def system_info():
    lines = [f"OS        : {platform.platform()}",
             f"CPU       : {platform.processor() or platform.machine()} ({os.cpu_count()} logical cores)",
             f"Python    : {platform.python_implementation()} {platform.python_version()}"]
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        psize = os.sysconf("SC_PAGE_SIZE")
        lines.append(f"RAM       : {pages * psize / 2 ** 30:.1f} GiB")
    except (ValueError, OSError, AttributeError):
        try:
            import psutil  # type: ignore
            lines.append(f"RAM       : {psutil.virtual_memory().total / 2 ** 30:.1f} GiB")
        except Exception:
            pass
    return "\n".join(lines)


def format_table(results):
    head = ["Algorithm", "Status", "Moves", "Cells", "Expanded", "Generated", "PeakNodes", "Time(ms)", "Verified"]
    rows = []
    for r in results:
        rows.append([
            r.label, r.status,
            "-" if r.moves is None else str(len(r.moves)),
            "-" if r.cells is None else str(r.cells),
            f"{r.expanded:,}", f"{r.generated:,}", f"{r.peak:,}", f"{r.time_ms:,.1f}",
            "-" if r.verified is None else ("OK" if r.verified else "FAIL"),
        ])
    widths = [max(len(head[i]), *(len(row[i]) for row in rows)) for i in range(len(head))] if rows else [len(x) for x in head]
    fmt = "  ".join("{:<%d}" % widths[0] if i == 0 else "{:>%d}" % widths[i] for i in range(len(head)))
    out = [fmt.format(*head), "  ".join("-" * wd for wd in widths)]
    out += [fmt.format(*row) for row in rows]
    return "\n".join(out)


def run_algorithms(puzzle, algos, args):
    results = []
    lim = Limits(args.time_limit, args.max_nodes, args.max_stored)
    solver = Solver(puzzle, cost=args.cost, limits=lim)
    for a in algos:
        results.append(solver.solve(a, weight=args.weight, depth_limit=args.depth_limit,
                                    tt=args.tt, precheck=not args.no_precheck))
    return results


args_cost_holder = argparse.Namespace(cost="moves")


def pick_best(results):
    solved = [r for r in results if r.status == "SOLVED"]
    if not solved:
        return None
    if getattr(args_cost_holder, "cost", "moves") == "cells":
        return min(solved, key=lambda r: (r.cells, r.n_moves))
    return min(solved, key=lambda r: (r.n_moves, r.cells))


def print_outcome(puzzle, results, args, out=sys.stdout):
    P = lambda *a: print(*a, file=out)
    args_cost_holder.cost = args.cost
    P(format_table(results))
    P()
    for r in results:
        if r.status != "SOLVED":
            P(f"[{r.label}] {r.status}: {r.reason}")
        elif r.extra:
            P(f"[{r.label}] " + ", ".join(f"{k}={v}" for k, v in r.extra.items()))
    best = pick_best(results)
    statuses = {r.status for r in results}
    P()
    if best is not None:
        ok, msg = verify_solution(puzzle, best.moves)
        P(f"RESULT: SOLUTION FOUND by {best.label} - {best.n_moves} moves, {best.cells} cells travelled")
        P(f"Verification (independent checker): {msg}")
        P("Keys (numpad)  : " + (" ".join(best.keys()) if best.moves else "(no move needed)"))
        P("Directions     : " + (" ".join(DIR_NAMES[d] for d in best.moves) if best.moves else "-"))
        if args.trace and best.moves:
            P(); P(describe_moves(puzzle, best.moves))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(best.keys() + "\n")
            P(f"Solution written to {args.out}")
        return 0
    if results and all(r.reason.startswith("pre-analysis") for r in results):
        P("RESULT: NO SOLUTION - " + results[0].reason + " (use --no-precheck to force a full search).")
        return 1
    if "NO_SOLUTION" in statuses and "LIMIT" not in statuses:
        P("RESULT: NO SOLUTION - the whole search space was explored without reaching the goal.")
        return 1
    if "NO_SOLUTION" in statuses:
        P("RESULT: NO SOLUTION (proved by at least one complete algorithm); others stopped on resource limits.")
        return 1
    P("RESULT: STOPPED - resource limits were hit before a solution was found (raise --time-limit / --max-nodes / --max-stored).")
    return 2


def parse_algos(spec):
    if spec.strip().lower() == "all":
        return list(ALL_ALGOS)
    algos = [a.strip().lower() for a in spec.split(",") if a.strip()]
    for a in algos:
        if a not in ALL_ALGOS:
            raise PuzzleError(f"unknown algorithm '{a}'; choose from {', '.join(ALL_ALGOS)} or 'all'")
    return algos or ["bfs", "ids", "astar"]


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
def benchmark(args, algos):
    w, h = parse_size(args.size or "8x6")
    rows, per_algo = [], {a: [] for a in algos}
    print(f"Benchmark: {args.benchmark} random solvable puzzles, {w}x{h}, seeds {args.seed}..{args.seed + args.benchmark - 1}, "
          f"cost={args.cost}")
    for i in range(args.benchmark):
        seed = args.seed + i
        p = generate_puzzle(w, h, seed, start_is_stop=not args.no_start_stop)
        res = run_algorithms(p, algos, args)
        print(f"\n--- puzzle seed={seed}  gems={p.num_gems}  id={p.game_id()}")
        print(format_table(res))
        for r in res:
            per_algo[r.algo].append(r)
            rows.append([seed, p.num_gems, r.label, r.status, r.n_moves, r.cells, r.expanded, r.generated,
                         r.peak, round(r.time_ms, 3), r.verified])
    print("\n=== Summary (averages over solved instances) ===")
    head = ["Algorithm", "Solved", "AvgMoves", "AvgExpanded", "AvgPeakNodes", "AvgTime(ms)"]
    tab = []
    for a in algos:
        rs = per_algo[a]
        ok = [r for r in rs if r.status == "SOLVED"]
        avg = lambda f: (sum(f(r) for r in ok) / len(ok)) if ok else float("nan")
        tab.append([rs[0].label if rs else a, f"{len(ok)}/{len(rs)}", f"{avg(lambda r: r.n_moves):.2f}",
                    f"{avg(lambda r: r.expanded):,.0f}", f"{avg(lambda r: r.peak):,.0f}", f"{avg(lambda r: r.time_ms):,.1f}"])
    widths = [max(len(head[i]), *(len(r[i]) for r in tab)) for i in range(len(head))]
    fmt = "  ".join("{:<%d}" % widths[0] if i == 0 else "{:>%d}" % widths[i] for i in range(len(head)))
    print(fmt.format(*head))
    print("  ".join("-" * x for x in widths))
    for r in tab:
        print(fmt.format(*r))
    print("\nSystem:\n" + system_info())
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["seed", "gems", "algorithm", "status", "moves", "cells", "expanded", "generated",
                         "peak_nodes", "time_ms", "verified"])
            wr.writerows(rows)
        print(f"CSV written to {args.csv}")
    return 0


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------
SAMPLE_10x8 = (  # transcribed from the screenshot of the web version
    "mwbmssmmws"
    "bwsbsbbwwg"
    "gbbwbsbbbS"
    "smbggmmswm"
    "mwsgwmmgws"
    "bbwmswbbsb"
    "swwssbwgbg"
    "wmmsgmsgbb"
)


def selftest():
    print("Running self-test ...")
    fails = 0

    def check(cond, msg):
        nonlocal fails
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        fails += 0 if cond else 1

    # parsing variants
    desc = "bbbbbbbbbbbbbbbSbbbbgbbb"
    p1 = parse_puzzle(f"4x6:{desc}")
    save = ("SAVEFILE:41:Simon Tatham's Portable Puzzle Collection\nVERSION :1:1\nGAME    :7:Inertia\n"
            f"PARAMS  :3:4x6\nCPARAMS :3:4x6\nDESC    :{len(desc)}:{desc}\nNSTATES :1:1\n")
    p2 = parse_puzzle(save)
    p3 = parse_puzzle("\n".join(desc[i:i + 4] for i in range(0, 24, 4)).replace("S", "@"))
    check(p1.cells == p2.cells == p3.cells, "game-id, save-file and grid parsing agree")
    try:
        parse_puzzle("GAME    :7:Mosaics\nDESC    :3:bbb\n")
        check(False, "wrong-game save file rejected")
    except PuzzleError:
        check(True, "wrong-game save file rejected")

    # solver agreement
    p = parse_puzzle(f"10x8:{SAMPLE_10x8}")
    lim = Limits(time_limit=20, max_nodes=2_000_000, max_stored=2_000_000)
    s = Solver(p, limits=lim)
    r = {a: s.solve(a, depth_limit=30, tt=True) for a in ("bfs", "ucs", "ids", "astar", "ils", "dfs")}
    for a, x in r.items():
        print(f"    {x.label:<8} {x.status:<12} moves={x.n_moves} expanded={x.expanded}")
    if all(x.status == "SOLVED" for x in r.values()):
        check(all(x.verified for x in r.values()), "every returned solution passes the independent verifier")
        opt = {r[a].n_moves for a in ("bfs", "ucs", "ids", "astar", "ils")}
        check(len(opt) == 1, f"BFS/UCS/IDS/ILS/A* agree on the optimal length ({opt})")
    else:
        check(all(x.status != "SOLVED" for x in r.values()), "all algorithms agree the sample has no solution")

    # cells cost: UCS == A* == ILS
    sc = Solver(p, cost="cells", limits=lim)
    rc = {a: sc.solve(a, tt=True) for a in ("ucs", "astar", "ils")}
    if all(x.status == "SOLVED" for x in rc.values()):
        check(len({x.cells for x in rc.values()}) == 1, "UCS/ILS/A* agree on minimal cells-cost")

    # random puzzles
    ok = True
    for seed in range(5):
        q = generate_puzzle(7, 6, seed)
        sv = Solver(q, limits=lim)
        a, b, c = sv.solve("bfs"), sv.solve("astar"), sv.solve("ids", tt=True)
        ok &= (a.status == b.status == c.status == "SOLVED" and a.n_moves == b.n_moves == c.n_moves
               and a.verified and b.verified and c.verified)
    check(ok, "5 random puzzles: BFS/IDS/A* optimal & verified")

    # verifier catches bad sequences
    check(not verify_solution(p, [0, 0, 0])[0], "verifier rejects an invalid sequence")
    # unsolvable detection
    u = parse_puzzle("3x3:Sbbbbbbbg")
    check(Solver(u).solve("bfs").status in ("SOLVED", "NO_SOLUTION"), "tiny board handled")
    u2 = parse_puzzle("3x3:Swbwwbbbg")
    check(Solver(u2).solve("bfs").status == "NO_SOLUTION", "unsolvable board detected")
    print("SELF-TEST " + ("FAILED" if fails else "PASSED"))
    return 1 if fails else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        prog="inertia_solver.py",
        description="Inertia puzzle solver: uninformed search vs A*.  Input = Tatham save file / game id / text grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "algorithms: bfs ucs dfs dls ids ils astar | all\n"
            "moves are printed as numpad digits: 7=NW 8=N 9=NE 4=W 6=E 1=SW 2=S 3=SE\n"
            "(type them in the web game to watch the solution)\n\n"
            "examples:\n"
            "  python inertia_solver.py samples/sample_10x8.txt\n"
            "  python inertia_solver.py game.sav --algo bfs,ids,astar --time-limit 120 --trace\n"
            "  python inertia_solver.py --benchmark 10 --size 8x6 --csv results.csv\n"
            "  python inertia_solver.py game.sav --check 6236214\n"
            "  python inertia_solver.py --gui"
        ))
    ap.add_argument("file", nargs="?", help="puzzle file ('-' = stdin)")
    ap.add_argument("--size", help="board size WxH (only needed for bare DESC strings of unusual size)")
    ap.add_argument("--algo", default="bfs,ids,astar", help="comma list or 'all' (default: bfs,ids,astar)")
    ap.add_argument("--cost", choices=["moves", "cells"], default="moves",
                    help="path cost: number of moves (default) or cells travelled")
    ap.add_argument("--weight", type=float, default=1.0, help="A* weight (1 = optimal; >1 = faster, not optimal)")
    ap.add_argument("--depth-limit", type=int, default=25, help="depth limit for DLS (default 25)")
    ap.add_argument("--tt", action="store_true", help="add a transposition table to DLS/IDS/ILS (more memory, far fewer nodes)")
    ap.add_argument("--time-limit", type=float, default=60.0, help="seconds per algorithm (default 60)")
    ap.add_argument("--max-nodes", type=int, default=5_000_000, help="max expanded nodes per algorithm")
    ap.add_argument("--max-stored", type=int, default=3_000_000, help="max nodes kept in memory per algorithm")
    ap.add_argument("--no-start-stop", dest="no_start_stop", action="store_true",
                    help="treat the start square as a plain blank square instead of a stop square")
    ap.add_argument("--no-precheck", action="store_true", help="skip the quick unreachable-gem analysis")
    ap.add_argument("--show", action="store_true", help="print the board as ASCII art")
    ap.add_argument("--trace", action="store_true", help="print a step-by-step trace of the solution")
    ap.add_argument("--out", help="write the best solution (digits) to this file")
    ap.add_argument("--check", help="verify a move sequence (digits / N,S,E,W...) or @file against the puzzle")
    ap.add_argument("--generate", metavar="WxH", help="generate a random solvable puzzle (use --seed)")
    ap.add_argument("--save-puzzle", help="with --generate: write the puzzle to this file")
    ap.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    ap.add_argument("--benchmark", type=int, metavar="N", help="run the chosen algorithms on N random puzzles")
    ap.add_argument("--csv", help="with --benchmark: write all rows to a CSV file")
    ap.add_argument("--selftest", action="store_true", help="run built-in consistency tests")
    ap.add_argument("--gui", action="store_true", help="open the graphical interface (needs tkinter)")
    ap.add_argument("--sysinfo", action="store_true", help="print system information and continue")
    ap.add_argument("--version", action="version", version=f"inertia_solver {__version__}")
    return ap


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return _main(ap, args)
    except PuzzleError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"Error: file not found: {e.filename}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except BrokenPipeError:
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass
        return 0


def _main(ap, args):
    if args.selftest:
        return selftest()
    if args.gui:
        try:
            import inertia_pygame_gui as gui           # preferred: pygame UI (needs: pip install pygame)
        except ImportError:
            try:
                import inertia_gui as gui              # fallback: tkinter UI
            except ImportError as e:
                print(f"Cannot start the GUI: {e}\n(pip install pygame   or   Linux: sudo apt install python3-tk)",
                      file=sys.stderr)
                return 2
        return gui.run(args.file)
    if args.sysinfo:
        print(system_info() + "\n")

    algos = parse_algos(args.algo)
    sz = parse_size(args.size) if args.size and not args.benchmark else None

    if args.benchmark:
        if args.benchmark < 1:
            raise PuzzleError("--benchmark needs a positive number")
        return benchmark(args, algos)

    if args.generate:
        w, h = parse_size(args.generate)
        p = generate_puzzle(w, h, args.seed, start_is_stop=not args.no_start_stop)
        print(f"Generated {w}x{h} puzzle (seed {args.seed}):  {p.game_id()}")
        if args.save_puzzle:
            with open(args.save_puzzle, "w", encoding="utf-8") as f:
                f.write(p.game_id() + "\n")
            print(f"Saved to {args.save_puzzle}")
    else:
        if not args.file:
            ap.print_help()
            return 2
        text = sys.stdin.read() if args.file == "-" else read_text_file(args.file)
        p = parse_puzzle(text, size=sz, start_is_stop=not args.no_start_stop)

    print(f"Board {p.w}x{p.h}, {p.num_gems} gems, start at {p.xy(p.start)}, "
          f"start-square-is-stop={p.start_is_stop}, cost model={args.cost}")
    if args.show or args.generate:
        print(p.ascii())
        print("legend: @ start, * gem, X mine, o stop, # wall, . empty")
    print()

    if args.check is not None:
        t = args.check
        if t.startswith("@"):
            t = read_text_file(t[1:])
        moves = parse_moves(t)
        ok, msg = verify_solution(p, moves)
        print(("VALID   - " if ok else "INVALID - ") + msg)
        if args.trace or not ok:
            print(describe_moves(p, moves))
        return 0 if ok else 1

    return print_outcome(p, run_algorithms(p, algos, args), args)


if __name__ == "__main__":
    sys.exit(main())
