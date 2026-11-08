#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphical front-end for the Inertia solver (bonus part of the project).

  python inertia_gui.py [puzzle_file]        (or:  python inertia_solver.py --gui)

Features
  * play the puzzle yourself (numpad / arrow keys / q w e a d z x c / mouse click)
  * define a NEW puzzle: Edit mode, pick a brush and click / drag on the board
  * open a Tatham save file / game id / text grid, save, generate random puzzles
  * let the agent solve it with any algorithm (BFS UCS DFS DLS IDS ILS A*)
    and watch the solution being replayed, with time / node statistics
Needs tkinter (bundled with Python; Linux: sudo apt install python3-tk).
"""
import math
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from inertia_solver import (ALL_ALGOS, ALGO_TITLES, DIR_KEYS, DIR_NAMES, KEY_TO_DIR, Limits,
                            PlayState, Puzzle, PuzzleError, SAMPLE_10x8, Solver, generate_puzzle,
                            parse_puzzle, read_text_file, verify_solution)

COL_BG = "#e8dfc6"
COL_GRID = "#c9bf9e"
COL_TRAIL = "#f4e9ae"
COL_WALL = "#f3f1ea"
COL_WALL_DARK = "#8d8a80"
COL_GEM = "#5ff0f0"
COL_GEM_EDGE = "#1b8f8f"
COL_BALL = "#e02020"

KEYSYM_TO_DIR = {
    "Up": 1, "Down": 6, "Left": 3, "Right": 4,
    "KP_Home": 0, "KP_Up": 1, "KP_Prior": 2, "KP_Left": 3, "KP_Right": 4,
    "KP_End": 5, "KP_Down": 6, "KP_Next": 7,
    "q": 0, "w": 1, "e": 2, "a": 3, "d": 4, "z": 5, "x": 6, "c": 7,
}
OCT_TO_DIR = [4, 7, 6, 5, 3, 0, 1, 2]       # octant (0 = east, clockwise) -> direction index

BRUSHES = [("Empty", "b"), ("Gem", "g"), ("Mine", "m"), ("Stop", "s"), ("Wall", "w"), ("Start", "S")]


class App(tk.Tk):
    def __init__(self, path=None):
        super().__init__()
        self.title("Inertia - search agent")
        self.resizable(False, False)
        self.w, self.h = 10, 8
        self.cells = list(SAMPLE_10x8)
        self.puzzle = None
        self.ps = None
        self.anim_pos = None
        self.busy = False
        self.replay_token = 0
        self.solution = None
        self.q = queue.Queue()

        self.mode = tk.StringVar(value="play")
        self.brush = tk.StringVar(value="w")
        self.algo = tk.StringVar(value="astar")
        self.cost = tk.StringVar(value="moves")
        self.weight = tk.StringVar(value="1.0")
        self.tlimit = tk.StringVar(value="30")
        self.dlimit = tk.StringVar(value="25")
        self.use_tt = tk.BooleanVar(value=False)
        self.speed = tk.DoubleVar(value=60)

        self._build_ui()
        if path:
            try:
                self._load_text(read_text_file(path))
            except (PuzzleError, OSError) as e:
                messagebox.showerror("Cannot open file", str(e))
        self._rebuild_play()
        self.redraw()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = ttk.Frame(self, padding=8)
        root.grid(row=0, column=0)
        self.canvas = tk.Canvas(root, width=640, height=520, bg="#f6f1e3", highlightthickness=1,
                                highlightbackground="#bbb")
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="n")
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        side = ttk.Frame(root, padding=(10, 0, 0, 0))
        side.grid(row=0, column=1, sticky="n")

        f = ttk.LabelFrame(side, text="Puzzle", padding=6)
        f.pack(fill="x")
        for i, (t, c) in enumerate([("Open...", self.cmd_open), ("Save...", self.cmd_save),
                                    ("New blank...", self.cmd_new), ("Random...", self.cmd_random)]):
            ttk.Button(f, text=t, command=c, width=12).grid(row=i // 2, column=i % 2, padx=2, pady=2)

        f = ttk.LabelFrame(side, text="Mode", padding=6)
        f.pack(fill="x", pady=4)
        ttk.Radiobutton(f, text="Play", value="play", variable=self.mode, command=self._mode_changed).grid(row=0, column=0)
        ttk.Radiobutton(f, text="Edit (define puzzle)", value="edit", variable=self.mode,
                        command=self._mode_changed).grid(row=0, column=1)
        bf = ttk.Frame(f)
        bf.grid(row=1, column=0, columnspan=2, pady=(4, 0))
        for i, (t, c) in enumerate(BRUSHES):
            ttk.Radiobutton(bf, text=t, value=c, variable=self.brush).grid(row=i // 3, column=i % 3, sticky="w", padx=3)

        f = ttk.LabelFrame(side, text="Agent", padding=6)
        f.pack(fill="x")
        ttk.Label(f, text="Algorithm").grid(row=0, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.algo, values=ALL_ALGOS, state="readonly", width=8).grid(row=0, column=1)
        ttk.Label(f, text="Cost").grid(row=1, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.cost, values=["moves", "cells"], state="readonly", width=8).grid(row=1, column=1)
        ttk.Label(f, text="A* weight").grid(row=2, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.weight, width=9).grid(row=2, column=1)
        ttk.Label(f, text="Time limit (s)").grid(row=3, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.tlimit, width=9).grid(row=3, column=1)
        ttk.Label(f, text="DLS depth").grid(row=4, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.dlimit, width=9).grid(row=4, column=1)
        ttk.Checkbutton(f, text="transposition table (DLS/IDS/ILS)", variable=self.use_tt).grid(row=5, column=0, columnspan=2, sticky="w")
        self.btn_solve = ttk.Button(f, text="Solve", command=self.cmd_solve)
        self.btn_solve.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        f = ttk.LabelFrame(side, text="Playback", padding=6)
        f.pack(fill="x", pady=4)
        ttk.Button(f, text="Replay solution", command=self.cmd_replay).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(f, text="Stop", command=self.cmd_stop).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(f, text="Undo", command=self.cmd_undo).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(f, text="Restart", command=self.cmd_restart).grid(row=1, column=1, padx=2, pady=2)
        ttk.Label(f, text="speed (ms/cell)").grid(row=2, column=0, sticky="w")
        ttk.Scale(f, from_=5, to=300, variable=self.speed, orient="horizontal", length=110).grid(row=2, column=1)

        self.log = tk.Text(root, width=62, height=9, state="disabled", font=("Courier", 9), wrap="word")
        self.log.grid(row=2, column=0, columnspan=2, pady=(6, 0), sticky="ew")
        self.status = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.status, font=("TkDefaultFont", 10, "bold")).grid(row=3, column=0, columnspan=2, sticky="w")

        self.bind("<Key>", self._on_key)
        self.focus_set()

    def _say(self, text, clear=False):
        self.log.configure(state="normal")
        if clear:
            self.log.delete("1.0", "end")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ------------------------------------------------------- puzzle plumbing
    def _load_text(self, text):
        p = parse_puzzle(text)
        self.w, self.h, self.cells = p.w, p.h, list(p.cells)
        self.solution = None

    def _try_puzzle(self, quiet=False):
        try:
            return Puzzle(self.w, self.h, self.cells)
        except PuzzleError as e:
            if not quiet:
                messagebox.showwarning("Invalid puzzle", str(e))
            return None

    def _rebuild_play(self):
        self.puzzle = self._try_puzzle(quiet=True)
        self.ps = PlayState(self.puzzle) if self.puzzle else None
        self.anim_pos = None
        self._update_status()

    def _mode_changed(self):
        self.cmd_stop()
        if self.mode.get() == "play":
            p = self._try_puzzle()
            if p is None:
                self.mode.set("edit")
                return
        self._rebuild_play()
        self.redraw()

    def _update_status(self):
        if self.mode.get() == "edit":
            self.status.set("Edit mode: choose a brush and click/drag on the board (exactly one Start is required).")
        elif self.ps is None:
            self.status.set("Puzzle is not valid yet - switch to Edit mode.")
        elif self.ps.dead:
            self.status.set("BOOM - you hit a mine.  Press Undo or Restart.")
        elif self.ps.won:
            self.status.set(f"Solved! All gems collected in {len(self.ps.moves)} moves.")
        else:
            left = self.puzzle.num_gems - bin(self.ps.mask).count("1")
            self.status.set(f"Moves: {len(self.ps.moves)}   gems left: {left}")

    # ------------------------------------------------------------- drawing
    def _cs(self):
        return max(22, min(64, 620 // max(self.w, self.h)))

    def redraw(self):
        c = self.canvas
        c.delete("all")
        cs = self._cs()
        pad = 10
        c.configure(width=self.w * cs + 2 * pad, height=self.h * cs + 2 * pad)
        playing = self.mode.get() == "play" and self.ps is not None
        collected = self.ps.mask if playing else 0
        trail = self.ps.trail if playing else set()
        for y in range(self.h):
            for x in range(self.w):
                i = y * self.w + x
                x0, y0 = pad + x * cs, pad + y * cs
                x1, y1 = x0 + cs, y0 + cs
                ch = self.cells[i]
                c.create_rectangle(x0, y0, x1, y1, fill=COL_TRAIL if i in trail else COL_BG, outline=COL_GRID)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if ch == "w":
                    c.create_rectangle(x0 + 3, y0 + 3, x1 - 3, y1 - 3, fill=COL_WALL, outline=COL_WALL_DARK, width=2)
                elif ch == "m":
                    r = cs * 0.24
                    for k in range(8):
                        a = k * math.pi / 4
                        c.create_line(cx + math.cos(a) * r, cy + math.sin(a) * r,
                                      cx + math.cos(a) * r * 1.6, cy + math.sin(a) * r * 1.6, width=3)
                    c.create_oval(cx - r, cy - r, cx + r, cy + r, fill="black")
                elif ch in ("s", "S"):
                    r = cs * 0.34
                    c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#333", dash=(3, 3), width=1)
                elif ch == "g":
                    show = True
                    if playing and self.puzzle and i in self.puzzle.gem_index:
                        show = not (collected >> self.puzzle.gem_index[i]) & 1
                    if show:
                        r = cs * 0.3
                        c.create_polygon(cx, cy - r, cx + r, cy, cx, cy + r, cx - r, cy, fill=COL_GEM, outline=COL_GEM_EDGE, width=2)
        # start marker (ball) ------------------------------------------------
        if playing:
            pos = self.anim_pos if self.anim_pos is not None else self.ps.pos
            dead = self.ps.dead and self.anim_pos is None
        else:
            pos = self.cells.index("S") if "S" in self.cells else None
            dead = False
        if pos is not None:
            x, y = pos % self.w, pos // self.w
            cx, cy = pad + x * cs + cs / 2, pad + y * cs + cs / 2
            r = cs * 0.32
            c.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#222" if dead else COL_BALL, outline="#600", width=2)
            c.create_oval(cx - r * 0.5, cy - r * 0.6, cx - r * 0.1, cy - r * 0.2, fill="#ff9a9a", outline="")
        self._update_status()

    # ------------------------------------------------------------- input
    def _cell_at(self, event):
        cs, pad = self._cs(), 10
        x, y = (event.x - pad) // cs, (event.y - pad) // cs
        if 0 <= x < self.w and 0 <= y < self.h:
            return int(x), int(y)
        return None

    def _paint(self, event):
        hit = self._cell_at(event)
        if hit is None:
            return
        x, y = hit
        i = y * self.w + x
        b = self.brush.get()
        if b == "S":
            self.cells = ["b" if ch == "S" else ch for ch in self.cells]
        elif self.cells[i] == "S":
            return                    # never delete the start by painting over it; place Start elsewhere
        self.cells[i] = b
        self.solution = None
        self.redraw()

    def _on_click(self, event):
        self.focus_set()
        if self.busy and self.mode.get() == "play":
            return
        if self.mode.get() == "edit":
            self._paint(event)
            return
        if self.ps is None:
            return
        hit = self._cell_at(event)
        if hit is None:
            return
        bx, by = self.ps.pos % self.w, self.ps.pos // self.w
        dx, dy = hit[0] - bx, hit[1] - by
        if dx == 0 and dy == 0:
            return
        octant = int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8
        self.do_move(OCT_TO_DIR[octant])

    def _on_drag(self, event):
        if self.mode.get() == "edit":
            self._paint(event)

    def _on_key(self, event):
        if self.mode.get() != "play" or isinstance(self.focus_get(), (ttk.Entry, tk.Entry)):
            return
        d = KEYSYM_TO_DIR.get(event.keysym)
        if d is None and event.char in KEY_TO_DIR:
            d = KEY_TO_DIR[event.char]
        if d is not None:
            self.do_move(d)
        elif event.keysym in ("u", "BackSpace"):
            self.cmd_undo()

    # --------------------------------------------------------------- moves
    def do_move(self, d, done=None, force=False):
        if self.ps is None or (self.busy and not force):
            return
        if self.ps.dead or self.ps.won:
            self._update_status()
            if done:
                done()
            return
        st, _end, _gm, _cnt, visited = self.puzzle.simulate(self.ps.pos, d)
        if st == "blocked":
            self.bell()
            if done:
                done()
            return
        self.busy = True
        delay = max(5, int(self.speed.get()))

        def step(k):
            if k >= len(visited):
                self.anim_pos = None
                self.ps.press(d)
                self.busy = False
                self.redraw()
                if done:
                    done()
                return
            self.anim_pos = visited[k]
            self.redraw()
            self.after(delay, step, k + 1)

        step(0)

    def cmd_undo(self):
        if self.busy:
            return
        if self.ps and self.ps.undo():
            self.redraw()

    def cmd_restart(self):
        self.cmd_stop()
        if self.ps:
            self.ps.reset()
            self.redraw()

    def cmd_stop(self):
        self.replay_token += 1
        self.busy = False
        self.anim_pos = None

    # ------------------------------------------------------------- commands
    def cmd_open(self):
        path = filedialog.askopenfilename(title="Open puzzle / Tatham save file")
        if not path:
            return
        try:
            self._load_text(read_text_file(path))
        except (PuzzleError, OSError) as e:
            messagebox.showerror("Cannot open file", str(e))
            return
        self.mode.set("play")
        self.cmd_stop()
        self._rebuild_play()
        self.redraw()
        self._say(f"Loaded {path}", clear=True)

    def cmd_save(self):
        p = self._try_puzzle()
        if p is None:
            return
        path = filedialog.asksaveasfilename(title="Save puzzle", defaultextension=".txt")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(p.game_id() + "\n")
            self._say(f"Saved {path}")

    def cmd_new(self):
        s = simpledialog.askstring("New puzzle", "Board size WxH (e.g. 10x8):", initialvalue="8x6", parent=self)
        if not s:
            return
        try:
            w, h = (int(t) for t in s.lower().replace("*", "x").split("x"))
            if not (2 <= w <= 40 and 2 <= h <= 40):
                raise ValueError
        except ValueError:
            messagebox.showerror("Bad size", "Please enter something like 10x8 (2..40).")
            return
        self.w, self.h = w, h
        self.cells = ["b"] * (w * h)
        self.cells[0] = "S"
        self.solution = None
        self.cmd_stop()
        self.mode.set("edit")
        self.brush.set("g")
        self._rebuild_play()
        self.redraw()
        self._say("Blank board created. Paint cells, then switch to Play (needs a start and, to be interesting, some gems).", clear=True)

    def cmd_random(self):
        s = simpledialog.askstring("Random puzzle", "Size WxH and seed, e.g.  10x8 5", initialvalue="10x8 1", parent=self)
        if not s:
            return
        try:
            parts = s.replace(",", " ").split()
            w, h = (int(t) for t in parts[0].lower().split("x"))
            seed = int(parts[1]) if len(parts) > 1 else 1
            p = generate_puzzle(w, h, seed)
        except (ValueError, IndexError, PuzzleError) as e:
            messagebox.showerror("Cannot generate", str(e) or "bad input")
            return
        self.w, self.h, self.cells, self.solution = p.w, p.h, list(p.cells), None
        self.mode.set("play")
        self.cmd_stop()
        self._rebuild_play()
        self.redraw()
        self._say(f"Random puzzle: {p.game_id()}", clear=True)

    def cmd_solve(self):
        if self.busy:
            return
        p = self._try_puzzle()
        if p is None:
            return
        try:
            weight = float(self.weight.get())
            tl = float(self.tlimit.get())
            dl = int(self.dlimit.get())
            if weight < 1 or tl <= 0 or dl < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Bad parameter", "weight >= 1, time limit > 0, depth >= 0 are required.")
            return
        algo, cost, tt = self.algo.get(), self.cost.get(), self.use_tt.get()
        self.btn_solve.state(["disabled"])
        self.status.set(f"Solving with {ALGO_TITLES[algo]} ...")
        self._say(f"--- {ALGO_TITLES[algo]}  cost={cost}", clear=True)

        def work():
            try:
                res = Solver(p, cost=cost, limits=Limits(time_limit=tl)).solve(algo, weight=weight, depth_limit=dl, tt=tt)
                self.q.put(("ok", p, res))
            except Exception as e:                      # never let the thread die silently
                self.q.put(("err", str(e), None))

        threading.Thread(target=work, daemon=True).start()
        self.after(120, self._poll)

    def _poll(self):
        try:
            kind, a, res = self.q.get_nowait()
        except queue.Empty:
            self.after(120, self._poll)
            return
        self.btn_solve.state(["!disabled"])
        if kind == "err":
            messagebox.showerror("Solver error", a)
            self._update_status()
            return
        p = a
        self._say(f"status    : {res.status}  ({res.reason})")
        self._say(f"expanded  : {res.expanded:,}   generated: {res.generated:,}   peak nodes in memory: {res.peak:,}")
        self._say(f"time      : {res.time_ms:,.1f} ms")
        if res.status == "SOLVED":
            ok, msg = verify_solution(p, res.moves)
            self._say(f"solution  : {len(res.moves)} moves, {res.cells} cells")
            self._say("keys      : " + (" ".join(DIR_KEYS[d] for d in res.moves) or "-"))
            self._say("check     : " + msg)
            self.solution = list(res.moves)
            self.status.set("Solution found - press 'Replay solution'.")
        else:
            self.solution = None
            self.status.set(f"No solution returned: {res.status}")

    def cmd_replay(self):
        if not self.solution:
            messagebox.showinfo("Replay", "Solve the puzzle first.")
            return
        if self.mode.get() != "play":
            self.mode.set("play")
            self._mode_changed()
        if self.ps is None:
            return
        self.cmd_stop()
        self.ps.reset()
        self.redraw()
        token = self.replay_token
        moves = list(self.solution)

        def nxt(i):
            if token != self.replay_token or i >= len(moves):
                return
            self.after(250, lambda: self.do_move(moves[i], done=lambda: nxt(i + 1), force=True)
                       if token == self.replay_token else None)

        nxt(0)


def run(path=None):
    try:
        app = App(path)
    except tk.TclError as e:
        print(f"Cannot open a window ({e}). Use the command-line solver instead.", file=sys.stderr)
        return 2
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else None))
