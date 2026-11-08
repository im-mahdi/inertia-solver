#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pygame front-end for the Inertia solver, drawn in the visual style of game.py
(3D tiles, blue gems, spiked red mines, dark-pink ball, rounded buttons).

  pip install pygame
  python inertia_pygame_gui.py [puzzle_file]      (or: python inertia_solver.py --gui)

Play     : numpad / arrow keys / q w e a d z x c / mouse click in a direction
           U or Backspace = Undo, Y = Redo, R = Restart, Enter = Solve
Edit     : switch to Edit mode, pick a brush, click / drag on the board
Agent    : pick an algorithm (BFS UCS DFS DLS IDS ILS A*) and press Solve;
           the solution is replayed on the board with time / node statistics
Files    : Open / Save buttons, or drag & drop a puzzle file on the window
"""
import math
import queue
import sys
import threading
import time

import pygame

from inertia_solver import (ALL_ALGOS, ALGO_TITLES, DIR_KEYS, DIR_NAMES, KEY_TO_DIR, Limits,
                            PlayState, Puzzle, PuzzleError, SAMPLE_10x8, Solver, generate_puzzle,
                            parse_puzzle, read_text_file, verify_solution)

# ------------------------------------------------------------------ colours (from game.py)
BG = (205, 228, 240)
BOARD = (177, 205, 220)
GRID = (135, 165, 182)
WHITE = (245, 250, 255)
DARK = (45, 65, 80)

PINK = (74, 6, 57)
PINK_LIGHT = (220, 100, 180)
PINK_SHADOW = (45, 0, 35)
BLUE = (58, 198, 239)
BLUE_LIGHT = (150, 235, 255)
BLUE_DARK = (15, 90, 125)
RED = (137, 5, 5)
RED_LIGHT = (210, 55, 55)
RED_DARK = (75, 0, 0)
TARGET = (173, 221, 156)
TARGET_LIGHT = (220, 245, 205)
TARGET_DARK = (105, 155, 95)

TILE = (225, 240, 245)
TILE_SHADOW = (160, 190, 205)
TRAIL = (250, 238, 190)
TRAIL_SHADOW = (200, 185, 130)
PANEL = (190, 215, 228)
OK_GREEN = (40, 120, 60)
BAD_RED = (160, 30, 30)

WIN_W, WIN_H = 1180, 820
BOARD_AREA = pygame.Rect(30, 85, 720, 540)   # region where the board is centred
PANEL_X = 785

KEY_TO_DIRECTION = {
    pygame.K_UP: 1, pygame.K_DOWN: 6, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    pygame.K_KP7: 0, pygame.K_KP8: 1, pygame.K_KP9: 2, pygame.K_KP4: 3,
    pygame.K_KP6: 4, pygame.K_KP1: 5, pygame.K_KP2: 6, pygame.K_KP3: 7,
    pygame.K_q: 0, pygame.K_w: 1, pygame.K_e: 2, pygame.K_a: 3,
    pygame.K_d: 4, pygame.K_z: 5, pygame.K_x: 6, pygame.K_c: 7,
}
OCT_TO_DIR = [4, 7, 6, 5, 3, 0, 1, 2]        # octant (0 = east, clockwise) -> direction index

BRUSHES = [("Empty", "b"), ("Gem", "g"), ("Mine", "m"), ("Stop", "s"), ("Wall", "w"), ("Start", "S")]
SPEEDS = [("Slow", 130), ("Medium", 65), ("Fast", 25), ("Turbo", 8)]
TIME_LIMITS = [10, 30, 60, 120]


def make_font(size, bold=True):
    return pygame.font.SysFont("arial,dejavusans,freesans,helvetica", size, bold=bold)


class App:
    def __init__(self, path=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Inertia - Gem Puzzle (search agent)")
        self.clock = pygame.time.Clock()
        self.font = make_font(25)
        self.mid_font = make_font(20)
        self.small = make_font(16)
        self.tiny = make_font(14, bold=False)
        self.mono = pygame.font.SysFont("couriernew,dejavusansmono,monospace", 14, bold=True)

        self.w, self.h = 10, 8
        self.cells = list(SAMPLE_10x8)
        self.puzzle = None
        self.ps = None
        self.redo_stack = []
        self.solution = None
        self.mode = "play"
        self.brush = "w"
        self.algo = "astar"
        self.cost = "moves"
        self.use_tt = False
        self.tl_idx = 1
        self.speed_idx = 1
        self.msg = "Collect all gems without hitting a mine!"
        self.msg_color = DARK
        self.stats = []                       # lines shown in the statistics box
        self.keys_line = ""
        self.anim = None                      # running slide animation
        self.replay = None                    # running solution replay
        self.solving = False
        self.q = queue.Queue()
        self.modal = None
        self.buttons = []                     # (rect, callback, enabled) rebuilt every frame
        self.cell = 62
        self.bx = self.by = 0

        if path:
            try:
                self.load_text(read_text_file(path))
                self.say(f"Loaded {path}")
            except (PuzzleError, OSError) as e:
                self.say(str(e), BAD_RED)
        self.rebuild_play()

    # ------------------------------------------------------------ helpers
    def say(self, text, color=DARK):
        self.msg, self.msg_color = text, color

    @property
    def busy(self):
        return self.anim is not None or self.replay is not None

    def layout(self):
        self.cell = max(18, min(66, BOARD_AREA.width // self.w, BOARD_AREA.height // self.h))
        bw, bh = self.w * self.cell, self.h * self.cell
        self.bx = BOARD_AREA.x + (BOARD_AREA.width - bw) // 2
        self.by = BOARD_AREA.y + (BOARD_AREA.height - bh) // 2

    def cell_xy(self, pos):
        return self.bx + (pos % self.w) * self.cell, self.by + (pos // self.w) * self.cell

    def load_text(self, text):
        p = parse_puzzle(text)
        self.w, self.h, self.cells = p.w, p.h, list(p.cells)
        self.solution = None
        self.stats, self.keys_line = [], ""
        self.mode = "play"
        self.stop_all()
        self.rebuild_play()

    def try_puzzle(self, quiet=False):
        try:
            return Puzzle(self.w, self.h, self.cells)
        except PuzzleError as e:
            if not quiet:
                self.say(str(e), BAD_RED)
            return None

    def rebuild_play(self):
        self.puzzle = self.try_puzzle(quiet=True)
        self.ps = PlayState(self.puzzle) if self.puzzle else None
        self.redo_stack = []

    def stop_all(self):
        self.anim = None
        self.replay = None

    # ------------------------------------------------------------ moves
    def do_move(self, d, from_replay=False):
        if self.ps is None or self.mode != "play" or self.anim is not None:
            return
        if self.busy and not from_replay:
            return
        if self.ps.dead:
            self.say("BOOM - you hit a mine. Press Undo or Restart.", BAD_RED)
            return
        if self.ps.won:
            return
        st, _end, _gm, _cnt, visited = self.puzzle.simulate(self.ps.pos, d)
        if st == "blocked":
            self.say(f"Blocked: cannot move {DIR_NAMES[d]}.", BAD_RED)
            return
        self.anim = {"d": d, "path": [self.ps.pos] + list(visited), "start": time.time(),
                     "from_replay": from_replay}

    def finish_move(self):
        a = self.anim
        self.anim = None
        st, _ = self.ps.press(a["d"])
        if not a["from_replay"]:
            self.redo_stack = []
        if st == "dead":
            self.say("BOOM! You hit a mine.", BAD_RED)
        elif self.ps.won:
            self.say(f"You won! All gems collected in {len(self.ps.moves)} moves.", OK_GREEN)
        else:
            left = self.puzzle.num_gems - bin(self.ps.mask).count("1")
            self.say(f"Moved {DIR_NAMES[a['d']]}. Gems left: {left}")

    def cmd_undo(self):
        if self.busy or self.ps is None or not self.ps.moves:
            return
        d = self.ps.moves[-1]
        if self.ps.undo():
            self.redo_stack.append(d)
            self.say("Undo!")

    def cmd_redo(self):
        if self.busy or self.ps is None or not self.redo_stack:
            return
        d = self.redo_stack.pop()
        self.do_move(d, from_replay=True)
        self.say("Redo!")

    def cmd_restart(self):
        self.stop_all()
        if self.ps:
            self.ps.reset()
            self.redo_stack = []
            self.say("Restarted. Collect all gems without hitting a mine!")

    # ------------------------------------------------------------ commands
    def set_mode(self, mode):
        if mode == self.mode:
            return
        self.stop_all()
        if mode == "play":
            if self.try_puzzle() is None:
                return
            self.mode = "play"
            self.rebuild_play()
            self.say("Play mode.")
        else:
            self.mode = "edit"
            self.say("Edit mode: pick a brush, click or drag on the board (exactly one Start).")

    def cmd_new(self):
        def ok(text):
            try:
                w, h = (int(t) for t in text.lower().replace("*", "x").split("x"))
                if not (2 <= w <= 40 and 2 <= h <= 40):
                    raise ValueError
            except ValueError:
                self.say("Please type a size like 10x8 (2..40).", BAD_RED)
                return
            self.w, self.h = w, h
            self.cells = ["b"] * (w * h)
            self.cells[0] = "S"
            self.solution = None
            self.stats, self.keys_line = [], ""
            self.stop_all()
            self.mode = "edit"
            self.brush = "g"
            self.rebuild_play()
            self.say("Blank board. Paint cells, then switch to Play.")
        self.modal = {"title": "New puzzle - board size WxH", "text": "8x6", "cb": ok}

    def cmd_random(self):
        def ok(text):
            try:
                parts = text.replace(",", " ").split()
                w, h = (int(t) for t in parts[0].lower().split("x"))
                seed = int(parts[1]) if len(parts) > 1 else 1
                p = generate_puzzle(w, h, seed)
            except (ValueError, IndexError, PuzzleError) as e:
                self.say(str(e) or "bad input, use e.g.  10x8 5", BAD_RED)
                return
            self.w, self.h, self.cells, self.solution = p.w, p.h, list(p.cells), None
            self.stats, self.keys_line = [], ""
            self.mode = "play"
            self.stop_all()
            self.rebuild_play()
            self.say(f"Random puzzle {p.game_id()[:46]}")
        self.modal = {"title": "Random puzzle - size and seed (e.g. 10x8 5)", "text": "10x8 1", "cb": ok}

    def cmd_open(self):
        path = None
        if sys.platform != "darwin":              # Tk + SDL clash on macOS -> typed path there
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                path = filedialog.askopenfilename(title="Open puzzle / Tatham save file")
                root.destroy()
                if not path:
                    return
            except Exception:
                path = None
        if path:
            self.open_path(path)
        else:
            self.modal = {"title": "Open file - type the path (or drag & drop a file)", "text": "",
                          "cb": self.open_path}

    def open_path(self, path):
        path = path.strip().strip('"')
        try:
            self.load_text(read_text_file(path))
            self.say(f"Loaded {path}")
        except (PuzzleError, OSError) as e:
            self.say(str(e), BAD_RED)

    def cmd_save(self):
        p = self.try_puzzle()
        if p is None:
            return
        path = None
        if sys.platform != "darwin":
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                path = filedialog.asksaveasfilename(title="Save puzzle", defaultextension=".txt")
                root.destroy()
                if not path:
                    return
            except Exception:
                path = None
        path = path or "inertia_saved.txt"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(p.game_id() + "\n")
            self.say(f"Saved {path}")
        except OSError as e:
            self.say(str(e), BAD_RED)

    def cmd_solve(self):
        if self.solving or self.busy:
            return
        p = self.try_puzzle()
        if p is None:
            return
        algo, cost, tt, tl = self.algo, self.cost, self.use_tt, TIME_LIMITS[self.tl_idx]
        self.solving = True
        self.solution = None
        self.stats = [f"Solving with {ALGO_TITLES[algo]} ..."]
        self.keys_line = ""

        def work():
            try:
                res = Solver(p, cost=cost, limits=Limits(time_limit=tl)).solve(algo, weight=1.0, depth_limit=25, tt=tt)
                self.q.put(("ok", p, res, algo))
            except Exception as e:                   # never let the thread die silently
                self.q.put(("err", str(e), None, algo))

        threading.Thread(target=work, daemon=True).start()

    def poll_solver(self):
        try:
            kind, a, res, algo = self.q.get_nowait()
        except queue.Empty:
            return
        self.solving = False
        if kind == "err":
            self.stats = [f"Solver error: {a}"]
            self.say("Solver error.", BAD_RED)
            return
        p = a
        lines = [f"{ALGO_TITLES[algo]}  ->  {res.status}",
                 f"({res.reason})" if res.reason else "",
                 f"expanded : {res.expanded:,}",
                 f"generated: {res.generated:,}",
                 f"peak mem : {res.peak:,} nodes",
                 f"time     : {res.time_ms:,.1f} ms"]
        if res.status == "SOLVED":
            ok, msg = verify_solution(p, res.moves)
            lines += [f"solution : {len(res.moves)} moves, {res.cells} cells", f"check    : {msg}"]
            self.keys_line = " ".join(DIR_KEYS[d] for d in res.moves)
            self.solution = list(res.moves)
            self.say("Solution found - replaying it on the board.", OK_GREEN)
            print("\n".join(l for l in lines if l))
            print("keys:", self.keys_line)
            self.stats = [l for l in lines if l]
            self.cmd_replay()
        else:
            self.say(f"No solution returned: {res.status}", BAD_RED)
            self.stats = [l for l in lines if l]
            print("\n".join(self.stats))

    def cmd_replay(self):
        if not self.solution:
            self.say("Solve the puzzle first.", BAD_RED)
            return
        if self.mode != "play":
            self.set_mode("play")
            if self.mode != "play":
                return
        self.stop_all()
        self.ps.reset()
        self.redo_stack = []
        self.replay = {"moves": list(self.solution), "i": 0, "wait": time.time() + 0.5}

    def update_replay(self):
        r = self.replay
        if r is None or self.anim is not None:
            return
        if r["i"] > 0 and r.get("applied") != r["i"]:
            r["applied"] = r["i"]
        if time.time() < r["wait"]:
            return
        if r["i"] >= len(r["moves"]):
            self.replay = None
            if self.ps.won:
                self.say(f"Replay finished - solved in {len(r['moves'])} moves.", OK_GREEN)
            return
        d = r["moves"][r["i"]]
        r["i"] += 1
        r["wait"] = time.time() + 0.25
        self.do_move(d, from_replay=True)
        if self.anim is None:                        # blocked / dead -> abort replay
            self.replay = None

    # ------------------------------------------------------------ animation
    def anim_state(self):
        """Returns (pixel_center, passed_cells) of the sliding ball or (None, set())."""
        a = self.anim
        if a is None:
            return None, set()
        delay = SPEEDS[self.speed_idx][1] / 1000.0
        t = (time.time() - a["start"]) / delay
        path = a["path"]
        last = len(path) - 1
        if t >= last:
            self.finish_move()
            return None, set()
        k = int(t)
        f = t - k
        x0, y0 = self.cell_xy(path[k])
        x1, y1 = self.cell_xy(path[k + 1])
        half = self.cell / 2 - 2
        cx = x0 + (x1 - x0) * f + half
        cy = y0 + (y1 - y0) * f + half
        return (cx, cy), set(path[:k + 2])

    # ------------------------------------------------------------ drawing: tiles
    def tile(self, x, y, color, shadow, light=WHITE):
        c = self.cell
        s = c / 62.0
        pygame.draw.rect(self.screen, shadow, (x + 3 * s, y + 4 * s, c - 3 * s, c - 3 * s), border_radius=9)
        pygame.draw.rect(self.screen, color, (x, y, c - 5 * s, c - 5 * s), border_radius=9)
        pygame.draw.line(self.screen, light, (x + 10 * s, y + 4 * s), (x + c - 18 * s, y + 4 * s), max(2, int(4 * s)))
        pygame.draw.line(self.screen, light, (x + 5 * s, y + 10 * s), (x + 5 * s, y + c - 18 * s), max(2, int(3 * s)))
        pygame.draw.rect(self.screen, GRID, (x, y, c - 5 * s, c - 5 * s), 2, border_radius=9)

    def center(self, x, y):
        s = self.cell / 62.0
        return x + 29 * s, y + 27 * s

    def draw_wall(self, x, y):
        c, s = self.cell, self.cell / 62.0
        pygame.draw.rect(self.screen, (75, 105, 125), (x + 3 * s, y + 5 * s, c - 5 * s, c - 5 * s), border_radius=8)
        pygame.draw.rect(self.screen, (105, 145, 165), (x, y, c - 5 * s, c - 9 * s), border_radius=8)
        pygame.draw.line(self.screen, (180, 210, 225), (x + 8 * s, y + 5 * s), (x + c - 15 * s, y + 5 * s), max(2, int(4 * s)))
        pygame.draw.rect(self.screen, GRID, (x, y, c - 5 * s, c - 9 * s), 2, border_radius=8)

    def draw_gem(self, x, y):
        s = self.cell / 62.0
        cx, cy = self.center(x, y)

        def P(dx, dy):
            return (cx + dx * s, cy + dy * s)
        pygame.draw.polygon(self.screen, BLUE_DARK, [P(0, -20), P(19, -4), P(0, 20), P(-19, -4)])
        body = [P(0, -23), P(20, -5), P(0, 17), P(-20, -5)]
        pygame.draw.polygon(self.screen, BLUE, body)
        pygame.draw.polygon(self.screen, BLUE_LIGHT, [P(0, -23), P(-20, -5), P(-5, -5)])
        pygame.draw.polygon(self.screen, (20, 135, 175), [P(0, -23), P(20, -5), P(5, -5)])
        pygame.draw.polygon(self.screen, (20, 100, 140), body, 2)
        pygame.draw.circle(self.screen, WHITE, P(-6, -8), max(2, int(3 * s)))

    def draw_mine(self, x, y):
        s = self.cell / 62.0
        cx, cy = self.center(x, y)
        pygame.draw.circle(self.screen, RED_DARK, (cx + 3 * s, cy + 4 * s), 19 * s)
        pygame.draw.circle(self.screen, RED, (cx, cy), 18 * s)
        pygame.draw.circle(self.screen, RED_LIGHT, (cx - 6 * s, cy - 7 * s), 7 * s)
        for ang in range(0, 360, 45):
            rad = math.radians(ang)
            a = (cx + 15 * s * math.cos(rad), cy + 15 * s * math.sin(rad))
            b = (cx + 22 * s * math.cos(rad), cy + 22 * s * math.sin(rad))
            pygame.draw.line(self.screen, RED_DARK, a, b, max(2, int(4 * s)))
        pygame.draw.circle(self.screen, DARK, (cx, cy), 18 * s, 2)

    def draw_stop(self, x, y):
        """Stop square: green 3D tile with a dashed ring (the 'broken circle' of Inertia)."""
        s = self.cell / 62.0
        self.tile(x, y, TARGET, TARGET_DARK, TARGET_LIGHT)
        cx, cy = self.center(x, y)
        r = 17 * s
        n = 16
        for k in range(n):
            a0 = 2 * math.pi * k / n
            a1 = a0 + math.pi / n * 0.95
            pts = [(cx + r * math.cos(a0 + (a1 - a0) * j / 3), cy + r * math.sin(a0 + (a1 - a0) * j / 3))
                   for j in range(4)]
            pygame.draw.lines(self.screen, (85, 135, 75), False, pts, max(2, int(4 * s)))
        pygame.draw.circle(self.screen, WHITE, (cx, cy), max(2, 3 * s))

    def draw_ball(self, cx, cy, dead=False):
        s = self.cell / 62.0
        body = (40, 40, 40) if dead else PINK
        pygame.draw.circle(self.screen, PINK_SHADOW, (cx + 3 * s, cy + 5 * s), 21 * s)
        pygame.draw.circle(self.screen, body, (cx, cy), 20 * s)
        pygame.draw.circle(self.screen, (110, 110, 110) if dead else PINK_LIGHT, (cx - 6 * s, cy - 7 * s), 8 * s)
        pygame.draw.circle(self.screen, DARK, (cx, cy), 20 * s, 2)

    # ------------------------------------------------------------ drawing: UI
    def button(self, text, rect, color, cb, enabled=True, active=False, font=None):
        rect = pygame.Rect(rect)
        font = font or self.small
        pygame.draw.rect(self.screen, (75, 105, 125), (rect.x + 3, rect.y + 5, rect.w, rect.h), border_radius=12)
        col = color if enabled else (200, 208, 214)
        pygame.draw.rect(self.screen, col, rect, border_radius=12)
        pygame.draw.line(self.screen, WHITE, (rect.x + 12, rect.y + 5), (rect.right - 12, rect.y + 5), 3)
        pygame.draw.rect(self.screen, (20, 100, 140) if active else DARK, rect, 4 if active else 2, border_radius=12)
        label = font.render(text, True, DARK if enabled else (140, 150, 158))
        self.screen.blit(label, (rect.x + (rect.w - label.get_width()) // 2,
                                 rect.y + (rect.h - label.get_height()) // 2))
        self.buttons.append((rect, cb, enabled))

    def section(self, title, y, h):
        rect = pygame.Rect(PANEL_X, y, 365, h)
        pygame.draw.rect(self.screen, (115, 150, 170), rect.move(0, 3), border_radius=14)
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=14)
        t = self.small.render(title, True, DARK)
        self.screen.blit(t, (rect.x + 14, rect.y + 7))
        return rect

    def draw(self):
        self.layout()
        self.buttons = []
        scr = self.screen
        scr.fill(BG)
        title = self.font.render("GEM PUZZLE  -  INERTIA", True, DARK)
        scr.blit(title, (30 + BOARD_AREA.width // 2 - title.get_width() // 2, 35))

        pos_px, passed = self.anim_state()

        # frame + cells
        cw, ch = self.w * self.cell, self.h * self.cell
        pygame.draw.rect(scr, (115, 150, 170), (self.bx - 12, self.by - 12, cw + 19, ch + 19), border_radius=15)
        pygame.draw.rect(scr, BOARD, (self.bx - 9, self.by - 9, cw + 13, ch + 13), border_radius=13)

        playing = self.mode == "play" and self.ps is not None and self.puzzle is not None
        grid = self.puzzle.grid if playing else [("s" if c == "S" else c) for c in self.cells]
        mask = self.ps.mask if playing else 0
        trail = self.ps.trail if playing else set()
        for i in range(self.w * self.h):
            x, y = self.cell_xy(i)
            c = grid[i]
            if c == "w":
                self.draw_wall(x, y)
            elif c == "s":
                self.draw_stop(x, y)
            else:
                if i in trail:
                    self.tile(x, y, TRAIL, TRAIL_SHADOW)
                else:
                    self.tile(x, y, TILE, TILE_SHADOW)
                if c == "m":
                    self.draw_mine(x, y)
                elif c == "g":
                    got = playing and i in self.puzzle.gem_index and (mask >> self.puzzle.gem_index[i]) & 1
                    if not got and i not in passed:
                        self.draw_gem(x, y)

        # ball
        if playing:
            if pos_px is not None:
                self.draw_ball(*pos_px)
            else:
                bx, by = self.cell_xy(self.ps.pos)
                half = self.cell / 2 - 2
                self.draw_ball(bx + half, by + half, dead=self.ps.dead)
        elif self.mode == "edit" and "S" in self.cells:
            bx, by = self.cell_xy(self.cells.index("S"))
            half = self.cell / 2 - 2
            self.draw_ball(bx + half, by + half)

        # messages / HUD
        if playing:
            left = self.puzzle.num_gems - bin(self.ps.mask).count("1")
            hud = f"Gems left: {left} / {self.puzzle.num_gems}      Moves: {len(self.ps.moves)}"
        else:
            hud = f"Edit mode  -  {self.w}x{self.h}  -  gems: {self.cells.count('g')}"
        t = self.font.render(hud, True, DARK)
        scr.blit(t, (30 + BOARD_AREA.width // 2 - t.get_width() // 2, 672))
        m = self.mid_font.render(self.msg, True, self.msg_color)
        if m.get_width() > 720:
            m = self.small.render(self.msg, True, self.msg_color)
        scr.blit(m, (30 + BOARD_AREA.width // 2 - m.get_width() // 2, 636))

        # bottom buttons
        can_play = playing and not self.busy
        self.button("Restart", (45, 735, 130, 52), (245, 205, 160), self.cmd_restart, playing)
        self.button("Undo", (200, 735, 130, 52), (180, 215, 240), self.cmd_undo, can_play and bool(self.ps.moves))
        self.button("Redo", (355, 735, 130, 52), (180, 225, 190), self.cmd_redo, can_play and bool(self.redo_stack))
        self.button("Replay", (510, 735, 130, 52), (225, 200, 240), self.cmd_replay, bool(self.solution) and not self.solving)
        sp = SPEEDS[self.speed_idx][0]
        self.button(f"Speed: {sp}", (665, 735, 100, 52), (235, 235, 200), self.cycle_speed, True, font=self.tiny)

        self.draw_panel()
        if self.modal:
            self.draw_modal()
        pygame.display.update()

    def cycle_speed(self):
        self.speed_idx = (self.speed_idx + 1) % len(SPEEDS)

    def draw_panel(self):
        # --- puzzle
        r = self.section("PUZZLE", 30, 112)
        for k, (t, cb) in enumerate([("Open", self.cmd_open), ("Save", self.cmd_save),
                                     ("New blank", self.cmd_new), ("Random", self.cmd_random)]):
            self.button(t, (r.x + 14 + (k % 2) * 172, r.y + 32 + (k // 2) * 38, 160, 32),
                        (245, 225, 180) if k < 2 else (200, 230, 200), cb, not self.solving)
        # --- mode
        r = self.section("MODE", 154, 138)
        self.button("Play", (r.x + 14, r.y + 30, 160, 30), (180, 215, 240), lambda: self.set_mode("play"),
                    True, self.mode == "play")
        self.button("Edit", (r.x + 186, r.y + 30, 160, 30), (180, 215, 240), lambda: self.set_mode("edit"),
                    not self.solving, self.mode == "edit")
        for k, (name, ch) in enumerate(BRUSHES):
            self.button(name, (r.x + 14 + (k % 3) * 116, r.y + 68 + (k // 3) * 34, 108, 28),
                        (235, 240, 245), lambda c=ch: self.set_brush(c),
                        self.mode == "edit", self.mode == "edit" and self.brush == ch, font=self.tiny)
        # --- agent
        r = self.section("AGENT", 304, 196)
        for k, a in enumerate(ALL_ALGOS):
            self.button(ALGO_TITLES[a], (r.x + 14 + (k % 4) * 87, r.y + 30 + (k // 4) * 34, 80, 28),
                        (255, 220, 190) if a == "astar" else (220, 230, 245),
                        lambda a=a: self.set_algo(a), not self.solving, self.algo == a, font=self.tiny)
        self.button(f"Cost: {self.cost}", (r.x + 14, r.y + 102, 112, 28), (235, 235, 200), self.toggle_cost,
                    not self.solving, font=self.tiny)
        self.button(f"TT: {'on' if self.use_tt else 'off'}", (r.x + 134, r.y + 102, 92, 28), (235, 235, 200),
                    self.toggle_tt, not self.solving, font=self.tiny)
        self.button(f"Limit: {TIME_LIMITS[self.tl_idx]}s", (r.x + 234, r.y + 102, 112, 28), (235, 235, 200),
                    self.cycle_limit, not self.solving, font=self.tiny)
        label = "Solving..." + "." * (int(time.time() * 3) % 4) if self.solving else f"Solve with {ALGO_TITLES[self.algo]}"
        self.button(label, (r.x + 14, r.y + 142, 332, 42), (190, 235, 170), self.cmd_solve,
                    not self.solving and not self.busy and self.mode in ("play", "edit"), font=self.mid_font)
        # --- statistics
        r = self.section("STATISTICS", 512, 278)
        y = r.y + 30
        for line in self.stats[:9]:
            col = OK_GREEN if "SOLVED" in line or line.strip().startswith("check") and "VALID" in line.upper() \
                and "INVALID" not in line.upper() else DARK
            self.screen.blit(self.mono.render(line[:44], True, col), (r.x + 14, y))
            y += 19
        if self.keys_line:
            y += 4
            self.screen.blit(self.small.render("keys (numpad):", True, DARK), (r.x + 14, y))
            y += 22
            words, row = self.keys_line.split(" "), ""
            lines = []
            for wd in words:
                if len(row) + len(wd) + 1 > 44:
                    lines.append(row)
                    row = wd
                else:
                    row = (row + " " + wd).strip()
            lines.append(row)
            for ln in lines[:(r.bottom - y - 8) // 18]:
                self.screen.blit(self.mono.render(ln, True, DARK), (r.x + 14, y))
                y += 18
        elif not self.stats:
            hint = ["Pick an algorithm and press Solve.", "", "Keys: numpad / arrows / QWE ADZXC",
                    "U undo   Y redo   R restart", "Enter solve"]
            for ln in hint:
                self.screen.blit(self.tiny.render(ln, True, DARK), (r.x + 14, y))
                y += 20

    def set_brush(self, c):
        self.brush = c

    def set_algo(self, a):
        self.algo = a

    def toggle_cost(self):
        self.cost = "cells" if self.cost == "moves" else "moves"

    def toggle_tt(self):
        self.use_tt = not self.use_tt

    def cycle_limit(self):
        self.tl_idx = (self.tl_idx + 1) % len(TIME_LIMITS)

    def draw_modal(self):
        ov = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        ov.fill((20, 35, 45, 150))
        self.screen.blit(ov, (0, 0))
        box = pygame.Rect(0, 0, 600, 170)
        box.center = (WIN_W // 2, WIN_H // 2)
        pygame.draw.rect(self.screen, (75, 105, 125), box.move(4, 6), border_radius=16)
        pygame.draw.rect(self.screen, BG, box, border_radius=16)
        pygame.draw.rect(self.screen, DARK, box, 2, border_radius=16)
        self.screen.blit(self.mid_font.render(self.modal["title"], True, DARK), (box.x + 24, box.y + 20))
        field = pygame.Rect(box.x + 24, box.y + 70, box.w - 48, 38)
        pygame.draw.rect(self.screen, WHITE, field, border_radius=8)
        pygame.draw.rect(self.screen, DARK, field, 2, border_radius=8)
        txt = self.modal["text"] + ("|" if int(time.time() * 2) % 2 == 0 else "")
        shown = self.mid_font.render(txt, True, DARK)
        while shown.get_width() > field.w - 16 and len(txt) > 1:
            txt = txt[1:]
            shown = self.mid_font.render(txt, True, DARK)
        self.screen.blit(shown, (field.x + 8, field.y + 7))
        self.screen.blit(self.tiny.render("Enter = OK     Esc = cancel", True, DARK), (box.x + 24, box.y + 125))

    # ------------------------------------------------------------ input
    def cell_at(self, pos):
        x, y = (pos[0] - self.bx) // self.cell, (pos[1] - self.by) // self.cell
        if 0 <= x < self.w and 0 <= y < self.h and pos[0] >= self.bx and pos[1] >= self.by:
            return int(x), int(y)
        return None

    def paint(self, pos):
        hit = self.cell_at(pos)
        if hit is None:
            return
        i = hit[1] * self.w + hit[0]
        b = self.brush
        if b == "S":
            self.cells = ["b" if ch == "S" else ch for ch in self.cells]
        elif self.cells[i] == "S":
            return                       # never erase the start by painting over it
        self.cells[i] = b
        self.solution = None

    def board_click(self, pos):
        if self.mode == "edit":
            self.paint(pos)
            return
        if self.ps is None or self.busy:
            return
        hit = self.cell_at(pos)
        if hit is None:
            return
        bx, by = self.ps.pos % self.w, self.ps.pos // self.w
        dx, dy = hit[0] - bx, hit[1] - by
        if dx == 0 and dy == 0:
            return
        octant = int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8
        self.do_move(OCT_TO_DIR[octant])

    def on_modal_key(self, event):
        m = self.modal
        if event.key == pygame.K_ESCAPE:
            self.modal = None
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.modal = None
            m["cb"](m["text"].strip())
        elif event.key == pygame.K_BACKSPACE:
            m["text"] = m["text"][:-1]
        elif event.unicode and event.unicode.isprintable():
            m["text"] += event.unicode

    def on_key(self, event):
        if self.modal:
            self.on_modal_key(event)
            return
        k = event.key
        if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.cmd_solve()
        elif k in (pygame.K_u, pygame.K_BACKSPACE):
            self.cmd_undo()
        elif k == pygame.K_y:
            self.cmd_redo()
        elif k == pygame.K_r:
            self.cmd_restart()
        elif k == pygame.K_ESCAPE:
            self.stop_all()
        else:
            d = KEY_TO_DIRECTION.get(k)
            if d is None and event.unicode in KEY_TO_DIR:
                d = KEY_TO_DIR[event.unicode]
            if d is not None:
                self.do_move(d)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            self.on_key(event)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.modal:
            for rect, cb, enabled in self.buttons:
                if rect.collidepoint(event.pos):
                    if enabled:
                        cb()
                    return
            self.board_click(event.pos)
        elif event.type == pygame.MOUSEMOTION and event.buttons[0] and self.mode == "edit" and not self.modal:
            self.paint(event.pos)
        elif event.type == pygame.DROPFILE:
            self.open_path(event.file)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    self.handle_event(event)
            self.poll_solver()
            self.update_replay()
            self.draw()
            self.clock.tick(60)
        pygame.quit()


def run(path=None):
    try:
        app = App(path)
    except pygame.error as e:
        print(f"Cannot open a window ({e}). Use the command-line solver instead.", file=sys.stderr)
        return 2
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else None))
