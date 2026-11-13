"""FluoroMerge GUI.

Drop a folder (or any channel images) onto the window. Files load into a
review grid, grouped into the sets FluoroMerge detected automatically. From
there you can merge every detected set in one click, or click any tiles to
pick which images go together and merge just those. Space opens the focused
image to change its color (LUT) and display range on a histogram; space again
returns to the grid. "Merge automatically" on the start screen skips the
review and works like earlier versions.
"""
from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageTk

from .core import (
    IMAGE_MATCH_NOTE, LUTS, Adjustment, lut_name, Profile, RunOptions, ScanResult, SourceImage, _load_array,
    append_merge_log, collect_tiffs, composite, default_output_dir,
    describe_members, group_sets, image_stats, load_profiles, manual_stem,
    run_merge, save_composite, scan_file, unique_path,
)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAVE_DND = True
except ImportError:
    HAVE_DND = False

PREVIEW_MAX = 1024     # longest side of the in-memory preview of each image
THUMB = 150            # longest side of grid thumbnails
HIST_BINS = 256
CLIP_WARN = 1.0        # % of pixels at/above the white point that counts as "a lot"
CLIP_COLOR = (255, 140, 0)
GAP = 14

BG = "#1b1c1f"
PANEL = "#25272b"
TILE = "#2f3136"
TILE_FOCUS = "#474a52"
FG = "#e8e8ea"
MUTED = "#9a9ca3"
ACCENT = "#3d8bfd"
UNKNOWN_COLOR = (255, 255, 255)
HIST_BG = "#141518"
WARN = "#ff9f1a"


def profile_dirs() -> list[Path]:
    """Bundled profiles (inside the PyInstaller bundle) plus a user-editable
    'profiles' folder next to the executable / repo root."""
    dirs = []
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys._MEIPASS) / "profiles")          # bundled
        dirs.append(Path(sys.executable).parent / "profiles")  # next to the exe
    else:
        dirs.append(Path(__file__).resolve().parent.parent / "profiles")
    dirs.append(Path.home() / ".fluoromerge" / "profiles")
    # de-duplicate while preserving order
    seen, unique = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def hex_color(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def shrink(px: np.ndarray, max_side: int) -> np.ndarray:
    """Area-averaged downsample so the longest side is at most max_side."""
    h, w = px.shape
    scale = max_side / max(h, w)
    if scale >= 1:
        return px.astype(np.float32)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    im = Image.fromarray(px.astype(np.float32), mode="F")
    return np.asarray(im.resize(size, Image.Resampling.BOX), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Workspace model
# --------------------------------------------------------------------------- #

@dataclass
class Loaded:
    """Pixels cached for one file: a display-size preview, a thumbnail, the
    file's own auto-contrast window, and its full-resolution shape."""
    preview: np.ndarray
    thumb: np.ndarray
    auto: tuple[float, float]
    shape: tuple[int, int]
    hist: np.ndarray                 # pixel counts in HIST_BINS bins ...
    hist_range: tuple[float, float]  # ... spanning the file's min..max


class Item:
    """One image in the workspace plus how the user wants it displayed:
    a color and a display window (black point, white point). `window` is
    None until the user moves it, meaning "use `base`" — the batch-wide window
    for the channel, or this image's own auto window."""

    def __init__(self, img: SourceImage, data: Loaded) -> None:
        self.img = img
        self.data = data
        self.base = data.auto
        self.color = UNKNOWN_COLOR
        self.window: tuple[float, float] | None = None
        self.edited = False
        self.channel_override: str | None = None
        self.tile: Tile | None = None

    @property
    def path(self) -> Path:
        return self.img.path

    @property
    def channel(self) -> str | None:
        return self.channel_override or self.img.channel

    def current_window(self) -> tuple[float, float]:
        return self.window or self.base

    def adjustment(self) -> Adjustment:
        black, white = self.current_window()
        return Adjustment(self.color, black, white)

    def clipped_percent(self) -> float:
        """Share of pixels at or above the white point (shown as full color)."""
        counts = self.data.hist
        lo, hi = self.data.hist_range
        centers = lo + (np.arange(len(counts)) + 0.5) * (hi - lo) / len(counts)
        total = counts.sum()
        white = self.current_window()[1]
        return 100.0 * counts[centers >= white].sum() / total if total else 0.0

    def copy_settings_from(self, other: Item) -> None:
        self.color, self.window = other.color, other.current_window()
        self.edited = True


# --------------------------------------------------------------------------- #
# Widgets
# --------------------------------------------------------------------------- #

class Scrollable(tk.Frame):
    """Vertically scrolling frame; children go in `.inner`."""

    def __init__(self, parent, bg: str) -> None:
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self.window = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._update)
        self.canvas.bind("<Configure>", lambda e: (self.canvas.itemconfigure(
            self.window, width=e.width), self._update()))

    def _update(self, _event=None) -> None:
        h = self.inner.winfo_reqheight() if not self.inner.place_slaves() \
            else self.inner.winfo_height()
        self.canvas.configure(scrollregion=(0, 0, self.inner.winfo_width(), h))
        needed = h > self.canvas.winfo_height() + 1
        if needed and not self.vsb.winfo_ismapped():
            self.vsb.pack(side="right", fill="y", before=self.canvas)
        elif not needed and self.vsb.winfo_ismapped():
            self.vsb.pack_forget()
            self.canvas.yview_moveto(0)

    def scroll(self, units: int) -> None:
        if self.inner.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(units, "units")

    def see(self, widget: tk.Widget) -> None:
        self.update_idletasks()
        total = max(1, self.inner.winfo_height())
        view = self.canvas.winfo_height()
        y = widget.winfo_rooty() - self.inner.winfo_rooty()
        h = widget.winfo_height()
        top = self.canvas.canvasy(0)
        if y < top:
            self.canvas.yview_moveto(max(0, y - GAP) / total)
        elif y + h > top + view:
            self.canvas.yview_moveto((y + h + GAP - view) / total)


class Tile(tk.Frame):
    """Thumbnail + channel / objective label. Click toggles selection;
    double-click (or space when focused) opens it."""

    def __init__(self, parent, app: App, item: Item) -> None:
        super().__init__(parent, bg=TILE, highlightthickness=3,
                         highlightbackground=PANEL, highlightcolor=PANEL)
        self.app, self.item = app, item
        self.photo: ImageTk.PhotoImage | None = None
        self.image_label = tk.Label(self, bg="black", bd=0)
        self.image_label.pack(padx=5, pady=(5, 3))
        row = tk.Frame(self, bg=TILE)
        row.pack(fill="x", padx=6)
        self.dot = tk.Canvas(row, width=12, height=12, bg=TILE, highlightthickness=0)
        self.dot.pack(side="left", padx=(0, 4))
        self.title = tk.Label(row, bg=TILE, fg=FG, font=app.font_bold, anchor="w")
        self.title.pack(side="left", fill="x")
        self.subtitle = tk.Label(self, bg=TILE, fg=MUTED, font=app.font_small,
                                 anchor="w", width=1)
        self.subtitle.pack(fill="x", padx=6, pady=(0, 5))
        for w in (self, self.image_label, row, self.dot, self.title, self.subtitle):
            w.bind("<Button-1>", self.on_click)
            w.bind("<Double-Button-1>", self.on_double)
            if HAVE_DND:
                app.register_drop(w)
        self.refresh()

    def on_click(self, _event) -> None:
        self.app.toggle_select(self.item)
        self.app.set_focus(self.item)

    def on_double(self, _event) -> None:
        self.app.toggle_select(self.item)   # undo the first click's toggle
        self.app.open_single(self.item)

    def refresh(self) -> None:
        item = self.item
        rgb = composite([(item.data.thumb, item.adjustment())])
        self.photo = ImageTk.PhotoImage(rgb)
        self.image_label.configure(image=self.photo, width=rgb.width, height=rgb.height)
        self.refresh_label()

    def refresh_label(self) -> None:
        item = self.item
        self.dot.delete("all")
        self.dot.create_oval(1, 1, 11, 11, fill=hex_color(item.color), outline="")
        channel = item.channel or "Unknown channel"
        if item.channel and item.img.channel_source == "filename-order" \
                and not item.channel_override:
            channel += " ?"      # guessed from file order
        px = item.img.pixel_size_um
        scale = item.img.objective or self.app.profile_objective() or (
            f"{px:.3g} µm/px" if px else None)
        parts = [channel, scale]
        self.title.configure(text="  ·  ".join(p for p in parts if p))
        self.subtitle.configure(text=item.path.name)

    def set_state(self, selected: bool, focused: bool) -> None:
        border = ACCENT if selected else (TILE_FOCUS if focused else PANEL)
        bg = TILE_FOCUS if focused else TILE
        self.configure(highlightbackground=border, highlightcolor=border, bg=bg)
        for w in (self.dot, self.title, self.subtitle, self.title.master):
            w.configure(bg=bg)


class Card(tk.Frame):
    """A titled group of tiles: one detected set, or the unmatched files."""

    def __init__(self, parent, app: App, title: str, note: str, items: list[Item],
                 wrap: bool) -> None:
        super().__init__(parent, bg=PANEL, padx=8, pady=6)
        self.app, self.items, self.wrap = app, items, wrap
        head = tk.Frame(self, bg=PANEL)
        head.pack(fill="x", pady=(0, 4))
        tk.Label(head, text=title, bg=PANEL, fg=FG, font=app.font_bold).pack(side="left")
        if note:
            # green: the images themselves show one field; amber: worth a check
            confident = note.startswith(IMAGE_MATCH_NOTE) and "· no" not in note
            tk.Label(head, text=f"  {note}", bg=PANEL,
                     fg="#7cc47c" if confident else "#e0b050",
                     font=app.font_small).pack(side="left")
        pick = tk.Label(head, text="select all", bg=PANEL, fg=ACCENT,
                        font=app.font_small, cursor="hand2")
        pick.pack(side="right", padx=(12, 0))
        pick.bind("<Button-1>", lambda e: app.select_only(items))
        self.body = tk.Frame(self, bg=PANEL)
        self.body.pack()
        self.tiles = []
        for item in items:
            item.tile = Tile(self.body, app, item)
            self.tiles.append(item.tile)
        if HAVE_DND:
            for w in (self, head, self.body):
                app.register_drop(w)
        self.columns = 0
        self.arrange(len(items))

    def arrange(self, columns: int) -> None:
        columns = max(1, min(columns, len(self.tiles)))
        if columns == self.columns:
            return
        self.columns = columns
        for i, t in enumerate(self.tiles):
            t.grid(row=i // columns, column=i % columns, padx=3, pady=3)

    def fit(self, max_width: int) -> None:
        if self.wrap and self.tiles:
            self.update_idletasks()
            tile_w = self.tiles[0].winfo_reqwidth() + 6
            self.arrange((max_width - 16) // tile_w)


class Histogram(tk.Canvas):
    """Pixel-intensity histogram (log scale) with draggable black and white
    point handles. Pixels at or below the black point show as black, pixels
    at or above the white point as full color. Drag between the handles to
    move both at once."""

    PAD = 9

    def __init__(self, parent, item: Item, height: int, on_change) -> None:
        super().__init__(parent, height=height, bg=HIST_BG, highlightthickness=0,
                         cursor="sb_h_double_arrow")
        self.item, self.on_change = item, on_change
        self._drag: str | None = None
        self._axis: tuple[float, float] | None = None
        self._grab_offset = 0.0
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_motion)
        self.bind("<ButtonRelease-1>", self.on_release)

    def axis(self) -> tuple[float, float]:
        if self._axis is not None:          # frozen while dragging
            return self._axis
        lo, hi = self.item.data.hist_range
        black, white = self.item.current_window()
        lo, hi = min(lo, black), max(hi, white)
        hi = hi if hi > lo else lo + 1
        # headroom past the brightest pixel so the white point can be raised
        # (dimming the image) even when it already sits at the top
        return lo, hi + 0.15 * (hi - lo)

    def x_of(self, v: float) -> float:
        lo, hi = self.axis()
        return self.PAD + (v - lo) / (hi - lo) * (self.winfo_width() - 2 * self.PAD)

    def v_of(self, x: float) -> float:
        lo, hi = self.axis()
        span = max(1, self.winfo_width() - 2 * self.PAD)
        return lo + (x - self.PAD) / span * (hi - lo)

    def draw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20:
            return
        item = self.item
        counts = item.data.hist
        hlo, hhi = item.data.hist_range
        black, white = item.current_window()
        heights = np.log1p(counts)
        heights = heights / heights.max() if heights.max() > 0 else heights
        bright = hex_color(item.color)
        r, g, b = item.color
        dim = hex_color((int(r * 0.3 + 40), int(g * 0.3 + 40), int(b * 0.3 + 40)))
        top, bottom = 6, h - 10
        bin_w = (hhi - hlo) / len(counts)
        prev_x = None
        for i, frac in enumerate(heights):
            v = hlo + (i + 0.5) * bin_w
            x0 = int(self.x_of(hlo + i * bin_w))
            x1 = max(x0 + 1, int(self.x_of(hlo + (i + 1) * bin_w)))
            if prev_x is not None and x0 < prev_x:
                x0 = prev_x
            prev_x = x1
            if frac <= 0 or x1 <= x0:
                continue
            y = bottom - frac * (bottom - top)
            self.create_rectangle(x0, y, x1, bottom, width=0,
                                  fill=bright if black <= v <= white else dim)
        xb, xw = self.x_of(black), self.x_of(white)
        # transfer line: how raw intensity maps to displayed brightness
        self.create_line(xb, bottom, xw, top, fill="#d0d0d4", dash=(3, 3))
        for x, fill in ((xb, "#000000"), (xw, "#ffffff")):
            self.create_line(x, top - 4, x, bottom, fill="#d0d0d4")
            self.create_polygon(x - 6, h - 1, x + 6, h - 1, x, bottom,
                                fill=fill, outline="#d0d0d4")

    def on_press(self, e) -> None:
        self._axis = self.axis()
        xb, xw = self.x_of(self.item.current_window()[0]), self.x_of(
            self.item.current_window()[1])
        db, dw = abs(e.x - xb), abs(e.x - xw)
        if min(db, dw) > 10 and xb < e.x < xw:
            self._drag = "both"
            self._grab_offset = self.v_of(e.x) - self.item.current_window()[0]
        else:
            self._drag = "black" if db < dw or (db == dw and e.x < xb) else "white"
        self.on_motion(e)

    def on_motion(self, e) -> None:
        if self._drag is None:
            return
        lo, hi = self._axis
        gap = (hi - lo) / 200
        v = min(max(self.v_of(e.x), lo), hi)
        black, white = self.item.current_window()
        if self._drag == "black":
            black = min(v, white - gap)
        elif self._drag == "white":
            white = max(v, black + gap)
        else:
            width = white - black
            black = min(max(self.v_of(e.x) - self._grab_offset, lo), hi - width)
            white = black + width
        self.item.window = (black, white)
        self.item.edited = True
        self.draw()
        self.on_change()

    def on_release(self, _e) -> None:
        self._drag, self._axis = None, None
        self.draw()


class AdjustPanel(tk.Frame):
    """One channel's display controls: color swatches, a histogram with
    black / white point handles, and Auto / Reset / use-for-all actions."""

    def __init__(self, parent, app: App, item: Item, on_change, merge_mode: bool) -> None:
        super().__init__(parent, bg=PANEL, padx=10, pady=6 if merge_mode else 8)
        self.app, self.item, self.on_change = app, item, on_change
        self.visible = tk.BooleanVar(value=True)

        head = tk.Frame(self, bg=PANEL)
        head.pack(fill="x")
        if merge_mode:
            ttk.Checkbutton(head, variable=self.visible, takefocus=0,
                            command=self.changed_visibility).pack(side="left")
        self.dot = tk.Canvas(head, width=12, height=12, bg=PANEL, highlightthickness=0)
        self.dot.pack(side="left", padx=(0, 5))
        tk.Label(head, text=item.channel or "Unknown channel", bg=PANEL, fg=FG,
                 font=app.font_bold).pack(side="left")
        if merge_mode:
            tk.Label(head, text=item.path.name, bg=PANEL, fg=MUTED, anchor="w",
                     font=app.font_small, width=1).pack(side="left", fill="x",
                                                        expand=True, padx=(8, 0))

        swatches = tk.Frame(self, bg=PANEL)
        swatches.pack(fill="x", pady=(6, 4))
        self.swatches: dict[tuple, tk.Canvas] = {}
        size = 18 if merge_mode else 22
        for rgb in LUTS.values():
            c = tk.Canvas(swatches, width=size, height=size, bg=PANEL,
                          highlightthickness=0, cursor="hand2")
            c.pack(side="left", padx=(0, 4))
            c.bind("<Button-1>", lambda e, rgb=rgb: self.set_color(rgb))
            self.swatches[rgb] = c
        self.color_label = tk.Label(swatches, bg=PANEL, fg=MUTED, font=app.font_small)
        if not merge_mode:
            self.color_label.pack(side="left", padx=(4, 0))

        self.hist = Histogram(self, item, 56 if merge_mode else 110, self._edited)
        self.hist.pack(fill="x")

        info = tk.Frame(self, bg=PANEL)
        info.pack(fill="x", pady=(3, 0))
        self.range_label = tk.Label(info, bg=PANEL, fg=MUTED, font=app.font_small)
        self.range_label.pack(side="left")
        self.clip_label = tk.Label(info, bg=PANEL, fg=MUTED, font=app.font_small)
        self.clip_label.pack(side="right")

        # compact layout in the merge preview: actions share the swatch row
        links = swatches if merge_mode else tk.Frame(self, bg=PANEL)
        if not merge_mode:
            links.pack(fill="x", pady=(4, 0))
        actions = [("Auto", self.auto), ("Reset", self.reset)]
        if item.channel:
            actions.append((f"{'All' if merge_mode else 'Use for all'} {item.channel}",
                            self.apply_to_channel))
        if merge_mode:
            actions.reverse()     # packed from the right
        side = "right" if merge_mode else "left"
        for n, (text, command) in enumerate(actions):
            if n:
                tk.Label(links, text="·", bg=PANEL, fg=MUTED,
                         font=app.font_small).pack(side=side, padx=3)
            link = tk.Label(links, text=text, bg=PANEL, fg=ACCENT, cursor="hand2",
                            font=app.font_small)
            link.pack(side=side)
            link.bind("<Button-1>", lambda e, c=command: c())
        self.sync()

    def sync(self) -> None:
        item = self.item
        for rgb, c in self.swatches.items():
            c.delete("all")
            on = tuple(item.color) == rgb
            n = int(c.cget("width"))
            c.create_rectangle(2, 2, n - 2, n - 2, fill=hex_color(rgb),
                               outline=FG if on else PANEL, width=2 if on else 1)
        self.dot.delete("all")
        self.dot.create_oval(1, 1, 11, 11, fill=hex_color(item.color), outline="")
        self.color_label.configure(text=lut_name(item.color))
        self.hist.draw()
        self.sync_readout()

    def sync_readout(self) -> None:
        black, white = self.item.current_window()
        self.range_label.configure(text=f"Display range {black:.0f} – {white:.0f}")
        clipped = self.item.clipped_percent()
        self.clip_label.configure(
            text=f"{clipped:.1f}% maxed out",
            fg=WARN if clipped > CLIP_WARN else MUTED)

    def _edited(self) -> None:
        self.sync_readout()
        self.on_change([self.item])

    def set_color(self, rgb) -> None:
        self.item.color = rgb
        self.item.edited = True
        self.sync()
        self.on_change([self.item])

    def auto(self) -> None:
        self.item.window = self.item.data.auto
        self.item.edited = True
        self.sync()
        self.on_change([self.item])

    def reset(self) -> None:
        self.app.reset_item(self.item)
        self.sync()
        self.on_change([self.item])

    def apply_to_channel(self) -> None:
        changed = self.app.apply_to_channel(self.item)
        self.app.log(f"Applied {self.item.path.name}'s color and display range to "
                     f"{len(changed)} other {self.item.channel} image(s).")
        self.on_change(changed + [self.item])

    def changed_visibility(self) -> None:
        self.on_change([])


class Viewer(tk.Frame):
    """Full-size view of one image, or of a merge preview, with adjustment
    controls on the right. Space / Esc returns to the grid."""

    def __init__(self, parent, app: App, items: list[Item], merge_mode: bool) -> None:
        super().__init__(parent, bg=BG)
        self.app, self.items, self.merge_mode = app, items, merge_mode
        self.photo: ImageTk.PhotoImage | None = None
        self._pending = None

        side = tk.Frame(self, bg=BG, width=340)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.canvas.bind("<Configure>", lambda e: self.schedule())
        if HAVE_DND:
            app.register_drop(self.canvas)

        # bottom of sidebar: actions + hints
        foot = tk.Frame(side, bg=BG)
        foot.pack(side="bottom", fill="x", pady=(6, 0))
        ttk.Checkbutton(foot, text="Highlight pixels at full brightness  (H)",
                        takefocus=0, variable=app.show_clipped,
                        command=self.schedule).pack(anchor="w", pady=(0, 6))
        if merge_mode:
            ttk.Button(foot, text="Save merge  ⏎", takefocus=0, style="Accent.TButton",
                       command=self.save).pack(fill="x")
            hint = "Space / Esc: back to grid    Untick a layer to leave it out"
        else:
            self.select_var = tk.BooleanVar(value=item_selected(app, items[0]))
            ttk.Checkbutton(foot, text="Selected for merging  (S)", takefocus=0,
                            variable=self.select_var,
                            command=lambda: app.toggle_select(self.items[0])
                            ).pack(anchor="w")
            hint = "Space / Esc: back to grid    ← →: previous / next"
        ttk.Button(foot, text="Back to grid", takefocus=0,
                   command=app.close_viewer).pack(fill="x", pady=(6, 0))
        tk.Label(foot, text=hint, bg=BG, fg=MUTED, font=app.font_small,
                 wraplength=320, justify="left").pack(anchor="w", pady=(6, 0))

        scroll = Scrollable(side, BG)
        scroll.pack(side="top", fill="both", expand=True)
        body = scroll.inner

        if merge_mode:
            title = f"Merge of {len(items)} images"
            sub = " + ".join(i.channel or "?" for i in items)
        else:
            title = items[0].path.name
            sub = f"in {items[0].path.parent.name}"
        tk.Label(body, text=title, bg=BG, fg=FG, font=app.font_title, anchor="w",
                 wraplength=310, justify="left").pack(fill="x")
        tk.Label(body, text=sub, bg=BG, fg=MUTED, font=app.font_small, anchor="w",
                 wraplength=310, justify="left").pack(fill="x", pady=(0, 8))

        if not merge_mode:
            self._info_rows(body, items[0])

        self.panels: list[AdjustPanel] = []
        for item in items:
            panel = AdjustPanel(body, app, item, self.changed, merge_mode)
            panel.pack(fill="x", pady=(0, 6))
            self.panels.append(panel)
        self.schedule()

    def _info_rows(self, body, item: Item) -> None:
        app = self.app
        grid = tk.Frame(body, bg=BG)
        grid.pack(fill="x", pady=(0, 8))
        tk.Label(grid, text="Channel", bg=BG, fg=MUTED, font=app.font_small).grid(
            row=0, column=0, sticky="w")
        choices = app.profile.channel_names
        var = tk.StringVar(value=item.channel or "")
        box = ttk.Combobox(grid, textvariable=var, values=choices, state="readonly",
                           width=12)
        box.grid(row=0, column=1, sticky="w", padx=(8, 0))
        box.bind("<<ComboboxSelected>>",
                 lambda e: app.set_channel(item, var.get()))
        how = {"metadata": "from metadata", "filename": "from file name",
               "color": "from image color", "filename-order": "guessed from file order",
               "unknown": "not recognized"}.get(item.img.channel_source, "")
        if item.channel_override:
            how = "set by you"
        rows = [("", how),
                ("Objective", item.img.objective or app.profile_objective() or "unknown"),
                ("Emission", f"{item.img.emission_nm:.0f} nm" if item.img.emission_nm
                 else "—"),
                ("Size", f"{item.data.shape[1]} × {item.data.shape[0]} px"),
                ("Pixel size", f"{(item.img.pixel_size_um or app.profile.pixel_size_um):.4g} µm"
                 if (item.img.pixel_size_um or app.profile.pixel_size_um) else "unknown")]
        for r, (k, v) in enumerate(rows, start=1):
            tk.Label(grid, text=k, bg=BG, fg=MUTED, font=app.font_small).grid(
                row=r, column=0, sticky="w")
            tk.Label(grid, text=v, bg=BG, fg=FG, font=app.font_small).grid(
                row=r, column=1, sticky="w", padx=(8, 0))

    def changed(self, items: list[Item]) -> None:
        self.app.thumbs_dirty.update(items)
        for p in self.panels:
            if p.item in items:
                p.sync()
        self.schedule()

    def schedule(self) -> None:
        if self._pending is None:
            self._pending = self.after(25, self.render)

    def layers(self) -> list[tuple[np.ndarray, Adjustment]]:
        return [(p.item.data.preview, p.item.adjustment())
                for p in self.panels if p.visible.get()]

    def render(self) -> None:
        self._pending = None
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        self.canvas.delete("all")
        layers = self.layers()
        if not layers:
            self.canvas.create_text(cw // 2, ch // 2, text="All layers hidden",
                                    fill=MUTED)
            return
        rgb = composite(layers)
        if self.app.show_clipped.get():
            clipped = np.zeros(layers[0][0].shape, dtype=bool)
            for px, adj in layers:
                clipped |= px >= adj.white
            if clipped.any():
                arr = np.array(rgb)
                arr[clipped] = CLIP_COLOR
                rgb = Image.fromarray(arr)
        scale = min(cw / rgb.width, ch / rgb.height)
        size = (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale)))
        rgb = rgb.resize(size, Image.Resampling.LANCZOS if scale < 1
                         else Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(rgb)
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo)

    def save(self) -> None:
        members = [p.item for p in self.panels if p.visible.get()]
        if not members:
            self.app.log("Nothing to save — every layer is hidden.")
            return
        self.app.save_manual(members)

    def sync_selection(self) -> None:
        if not self.merge_mode:
            self.select_var.set(item_selected(self.app, self.items[0]))


def item_selected(app: App, item: Item) -> bool:
    return item in app.selected


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

class App:
    def __init__(self) -> None:
        root = TkinterDnD.Tk() if HAVE_DND else tk.Tk()
        self.root = root
        root.title("FluoroMerge")
        root.geometry("1180x800")
        root.minsize(760, 560)
        root.configure(bg=BG)
        self._style()

        self.profiles: list[Profile] = load_profiles(profile_dirs())
        if not self.profiles:
            raise SystemExit(
                "No microscope profiles found. Put at least one .toml profile in "
                "a 'profiles' folder next to the app."
            )

        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.inputs: list[Path] = []
        self.cache: dict[Path, Loaded] = {}
        self.items: list[Item] = []
        self.result: ScanResult | None = None
        self.cards: list[Card] = []
        self.selected: list[Item] = []     # in click order
        self.focused: Item | None = None
        self.viewer: Viewer | None = None
        self.thumbs_dirty: set[Item] = set()
        self.show_clipped = tk.BooleanVar(value=False)
        self._layout_pending = None

        self._build_toolbar()
        self.body = tk.Frame(root, bg=BG)
        self.body.pack(fill="both", expand=True, padx=12)
        self._build_home()
        self._build_grid()
        self._build_bottom()
        self.show_home()

        root.bind_all("<Key>", self.on_key)
        root.bind_all("<MouseWheel>", self.on_wheel)
        root.bind_all("<Button-4>", lambda e: self.on_wheel(e, -1))
        root.bind_all("<Button-5>", lambda e: self.on_wheel(e, 1))
        root.bind_all("<ButtonRelease-1>", self._refocus, add="+")
        if HAVE_DND:
            self.register_drop(root)
        root.after(50, self.drain_events)

    # ---- setup ----

    def _style(self) -> None:
        import tkinter.font as tkfont

        base = tkfont.nametofont("TkDefaultFont")
        size = base.cget("size") or 12
        family = base.cget("family")
        self.font_small = (family, max(size - 2, 9))
        self.font_bold = (family, size, "bold")
        self.font_title = (family, size + 3, "bold")
        self.font_big = (family, size + 6)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=FG, fieldbackground=TILE,
                        bordercolor="#3a3d43", lightcolor=TILE, darkcolor=TILE,
                        troughcolor=TILE, focuscolor=ACCENT, selectbackground=ACCENT,
                        selectforeground="white", insertcolor=FG, arrowcolor=FG)
        style.configure("TButton", background="#383b41", padding=(10, 4))
        style.map("TButton", background=[("active", "#464a51"), ("disabled", TILE)],
                  foreground=[("disabled", MUTED)])
        style.configure("Accent.TButton", background=ACCENT, foreground="white")
        style.map("Accent.TButton", background=[("active", "#5a9dff"),
                                                ("disabled", TILE)])
        style.map("TCombobox", fieldbackground=[("readonly", TILE)],
                  foreground=[("readonly", FG)], background=[("readonly", "#383b41")])
        style.configure("TCheckbutton", background=BG)
        style.map("TCheckbutton", background=[("active", BG)],
                  indicatorcolor=[("selected", ACCENT), ("!selected", TILE)])
        style.configure("TRadiobutton", background=BG)
        style.map("TRadiobutton", background=[("active", BG)],
                  indicatorcolor=[("selected", ACCENT), ("!selected", TILE)])
        style.configure("Horizontal.TScale", background="#8a8d95", troughcolor=TILE,
                        sliderlength=16, sliderthickness=16, gripcount=0)
        style.configure("Vertical.TScrollbar", background="#383b41", troughcolor=BG,
                        arrowcolor=MUTED)
        self.root.option_add("*TCombobox*Listbox.background", TILE)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(12, 10, 12, 8))
        bar.pack(fill="x")
        ttk.Label(bar, text="Microscope").pack(side="left")
        self.profile_var = tk.StringVar(value=self.profiles[0].name)
        box = ttk.Combobox(bar, textvariable=self.profile_var, state="readonly",
                           width=26, values=[p.name for p in self.profiles])
        box.pack(side="left", padx=(6, 16))
        box.bind("<<ComboboxSelected>>", self.on_profile)

        ttk.Label(bar, text="Contrast").pack(side="left")
        self.contrast_var = tk.StringVar(value="fixed")
        for text, value in (("Same across batch", "fixed"), ("Auto per image", "auto")):
            ttk.Radiobutton(bar, text=text, value=value, variable=self.contrast_var,
                            takefocus=0, command=self.on_contrast).pack(
                side="left", padx=(6, 0))

        self.format_var = tk.StringVar(value="tif")
        ttk.Combobox(bar, textvariable=self.format_var, state="readonly", width=4,
                     values=["tif", "png"]).pack(side="right")
        ttk.Label(bar, text="Save as").pack(side="right", padx=(16, 6))
        self.scalebar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Scale bar", variable=self.scalebar_var,
                        takefocus=0).pack(side="right")

    def _build_home(self) -> None:
        self.home = tk.Frame(self.body, bg=BG)
        drop_text = ("Drop images or a folder here" if HAVE_DND
                     else "Choose images or a folder to begin")
        self.drop = tk.Label(self.home, text=drop_text, bg=PANEL, fg=FG,
                             font=self.font_big, height=8, bd=0, highlightthickness=0)
        self.drop.pack(fill="x", pady=(30, 14))
        if HAVE_DND:
            self.register_drop(self.drop)
            self.register_drop(self.home)
        buttons = tk.Frame(self.home, bg=BG)
        buttons.pack()
        ttk.Button(buttons, text="Choose folder…", takefocus=0,
                   command=self.pick_folder).pack(side="left")
        ttk.Button(buttons, text="Choose images…", takefocus=0,
                   command=self.pick_files).pack(side="left", padx=(8, 0))

        self.auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.home, takefocus=0, variable=self.auto_var,
                        text="Merge automatically without reviewing").pack(pady=(18, 2))
        tk.Label(
            self.home, bg=BG, fg=MUTED, font=self.font_small, justify="center",
            text=("Off: images open in a grid where you can check the detected sets, "
                  "pick any images to merge,\nand press Space on an image to adjust "
                  "its color and brightness.   On: every detected set is merged "
                  "straight away."),
        ).pack()

    def _build_grid(self) -> None:
        self.grid_frame = tk.Frame(self.body, bg=BG)
        actions = tk.Frame(self.grid_frame, bg=BG)
        actions.pack(fill="x", pady=(0, 8))
        self.merge_all_btn = ttk.Button(actions, text="Merge all detected sets",
                                        style="Accent.TButton", takefocus=0,
                                        command=self.merge_all)
        self.merge_all_btn.pack(side="left")
        ttk.Button(actions, text="Add images…", takefocus=0,
                   command=self.pick_files).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Start over", takefocus=0,
                   command=self.clear_workspace).pack(side="left", padx=(8, 0))
        self.summary_label = tk.Label(actions, bg=BG, fg=MUTED, font=self.font_small)
        self.summary_label.pack(side="left", padx=(14, 0))
        tk.Label(actions, bg=BG, fg=MUTED, font=self.font_small,
                 text="Click to select · Space to open · ⌘/Ctrl-A all · Esc clear"
                 ).pack(side="right")

        self.scroll = Scrollable(self.grid_frame, BG)
        self.scroll.pack(fill="both", expand=True)
        self.scroll.canvas.bind("<Configure>", self._on_grid_resize, add="+")
        if HAVE_DND:
            self.register_drop(self.scroll.canvas)
            self.register_drop(self.scroll.inner)

        sel = tk.Frame(self.grid_frame, bg=PANEL, padx=10, pady=8)
        sel.pack(fill="x", pady=(8, 0))
        self.selection_label = tk.Label(sel, bg=PANEL, fg=FG, anchor="w")
        self.selection_label.pack(side="left", fill="x", expand=True)
        ttk.Button(sel, text="Clear", takefocus=0,
                   command=lambda: self.select_only([])).pack(side="right")
        self.merge_btn = ttk.Button(sel, text="Merge selected  (M)", takefocus=0,
                                    style="Accent.TButton",
                                    command=lambda: self.save_manual(self.selected_ordered()))
        self.merge_btn.pack(side="right", padx=(0, 8))
        self.preview_btn = ttk.Button(sel, text="Preview & adjust merge  (P)",
                                      takefocus=0, command=self.open_merge)
        self.preview_btn.pack(side="right", padx=(0, 8))

    def _build_bottom(self) -> None:
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", padx=12, pady=(8, 10))
        self.status = tk.Label(bottom, bg=BG, fg=MUTED, anchor="w",
                               font=self.font_small)
        self.status.pack(fill="x")
        self.log_box = tk.Text(bottom, height=5, state="disabled", wrap="word",
                               bg=PANEL, fg=FG, relief="flat", bd=0,
                               highlightthickness=0, font=self.font_small,
                               padx=8, pady=6)
        self.log_box.pack(fill="x")

    def register_drop(self, widget: tk.Widget) -> None:
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", self.on_drop)

    # ---- screens ----

    def show_home(self) -> None:
        self.grid_frame.pack_forget()
        self.home.pack(fill="both", expand=True)

    def show_grid(self) -> None:
        self.home.pack_forget()
        self.grid_frame.pack(fill="both", expand=True)

    @property
    def profile(self) -> Profile:
        return next(p for p in self.profiles if p.name == self.profile_var.get())

    def profile_objective(self) -> str | None:
        import re

        m = re.search(r"(\d{1,3})\s?[xX]\b", self.profile.name)
        return f"{m.group(1)}x" if m else None

    # ---- input ----

    def on_drop(self, event) -> str:
        paths = [Path(p) for p in self.root.tk.splitlist(event.data)]
        self.add_inputs(paths)
        return event.action

    def pick_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Choose a folder of channel images")
        if chosen:
            self.add_inputs([Path(chosen)])

    def pick_files(self) -> None:
        chosen = filedialog.askopenfilenames(
            title="Choose channel images",
            filetypes=[("TIFF images", "*.tif *.tiff")],
        )
        if chosen:
            self.add_inputs([Path(p) for p in chosen])

    def add_inputs(self, paths: list[Path]) -> None:
        if self.busy:
            self.log("Still working — please wait a moment.")
            return
        if not collect_tiffs(paths):
            self.log("No .tif/.tiff files in what was dropped.")
            return
        if self.auto_var.get() and not self.items:
            self.run_auto(paths)
            return
        for p in paths:
            if p not in self.inputs:
                self.inputs.append(p)
        self.rescan()

    def run_auto(self, paths: list[Path]) -> None:
        """One-step batch merge, exactly as earlier versions did it."""
        self.clear_log()
        options = RunOptions(contrast=self.contrast_var.get(),
                             scale_bar=self.scalebar_var.get(),
                             output_format=self.format_var.get())
        profile = self.profile
        self.busy = True

        def work() -> None:
            try:
                run_merge(paths, profile, options,
                          log=lambda m: self.events.put(("log", m)))
            except Exception as exc:
                self.events.put(("log", f"Error: {exc}"))
            finally:
                self.events.put(("idle", None))

        threading.Thread(target=work, daemon=True).start()

    # ---- loading (background thread) ----

    def rescan(self) -> None:
        """Scan + group every input with the current profile, loading pixels
        for files not seen before. Runs off the UI thread."""
        self.busy = True
        profile, inputs, cache = self.profile, list(self.inputs), dict(self.cache)
        self.set_status("Scanning…")

        def work() -> None:
            try:
                files = collect_tiffs(inputs)
                seen, unique = set(), []
                for f in files:
                    if f.resolve() not in seen:
                        seen.add(f.resolve())
                        unique.append(f)
                images = []
                for n, f in enumerate(unique, 1):
                    self.events.put(("status", f"Reading {n} of {len(unique)}: {f.name}"))
                    try:
                        images.append(scan_file(f))
                    except Exception as exc:
                        self.events.put(("log", f"⚠ Can't read {f.name}: {exc}"))
                result = group_sets(images, profile,
                                    progress=lambda m: self.events.put(("status", m)))
                loaded = {}
                for n, img in enumerate(images, 1):
                    if img.path in cache:
                        continue
                    self.events.put(("status",
                                     f"Loading pixels {n} of {len(images)}: {img.path.name}"))
                    px = _load_array(img.path)
                    preview = shrink(px, PREVIEW_MAX)
                    lo, hi = float(px.min()), float(px.max())
                    hi = hi if hi > lo else lo + 1
                    hist, _ = np.histogram(px, bins=HIST_BINS, range=(lo, hi))
                    loaded[img.path] = Loaded(preview=preview,
                                              thumb=shrink(preview, THUMB),
                                              auto=image_stats(px, profile),
                                              shape=px.shape[:2],
                                              hist=hist, hist_range=(lo, hi))
                self.events.put(("loaded", (images, result, loaded)))
            except Exception as exc:
                self.events.put(("log", f"Error: {exc}"))
                self.events.put(("idle", None))

        threading.Thread(target=work, daemon=True).start()

    def on_loaded(self, images: list[SourceImage], result: ScanResult,
                  loaded: dict[Path, Loaded]) -> None:
        self.cache.update(loaded)
        old = {i.path: i for i in self.items}
        self.items = []
        for img in images:
            item = old.get(img.path)
            if item is None:
                item = Item(img, self.cache[img.path])
            else:
                item.img = img
            self.items.append(item)
        self.result = result
        by_path = {i.path: i for i in self.items}
        self.selected = [by_path[i.path] for i in self.selected if i.path in by_path]
        if self.focused is not None:
            self.focused = by_path.get(self.focused.path)
        self.recompute_bases()
        self.build_grid()
        self.show_grid()
        for w in result.warnings:
            self.log(f"⚠ {w}")
        n_sets = len(result.sets)
        self.set_status(f"{len(self.items)} image(s) · {n_sets} set(s) detected"
                        f" · {len(result.orphans)} not in a set")
        self.busy = False

    # ---- display settings ----

    def recompute_bases(self) -> None:
        """Default display window per image: the batch-wide window for its
        channel ("same across batch") or its own auto window."""
        profile = self.profile
        windows: dict[str, tuple[float, float]] = {}
        if self.contrast_var.get() == "fixed":
            for ch in profile.channel_names:
                autos = [i.data.auto for i in self.items if i.channel == ch]
                if autos:
                    windows[ch] = (min(a[0] for a in autos), max(a[1] for a in autos))
        for item in self.items:
            item.base = windows.get(item.channel, item.data.auto)
            if not item.edited:
                item.color = self.default_color(item)

    def default_color(self, item: Item) -> tuple[int, int, int]:
        try:
            return self.profile.color_of(item.channel) if item.channel else UNKNOWN_COLOR
        except KeyError:
            return UNKNOWN_COLOR

    def reset_item(self, item: Item) -> None:
        item.window = None
        item.color = self.default_color(item)
        item.edited = False

    def apply_to_channel(self, source: Item) -> list[Item]:
        changed = [i for i in self.items if i is not source and i.channel == source.channel]
        for i in changed:
            i.copy_settings_from(source)
        return changed

    def set_channel(self, item: Item, channel: str) -> None:
        item.channel_override = channel
        if not item.edited:
            item.color = self.default_color(item)
        self.recompute_bases()
        self.thumbs_dirty.update(self.items)
        self.log(f"{item.path.name}: channel set to {channel}. Use the grid to pick "
                 f"its set by hand.")
        if self.viewer is not None:
            self.open_single(item)    # rebuild the side panel

    def on_contrast(self) -> None:
        if self.items:
            self.recompute_bases()
            self.refresh_tiles(self.items)
            if self.viewer is not None:
                self.viewer.changed([])

    def on_profile(self, _event=None) -> None:
        self.root.focus_set()
        if self.items and not self.busy:
            self.log(f"Re-reading images with profile '{self.profile.name}'…")
            for i in self.items:
                i.channel_override = None
            self.close_viewer()
            self.rescan()

    # ---- grid ----

    def build_grid(self) -> None:
        for c in self.cards:
            c.destroy()
        self.cards = []
        for item in self.items:
            item.tile = None
        by_path = {i.path: i for i in self.items}
        order = {name: n for n, name in enumerate(self.profile.channel_names)}
        inner = self.scroll.inner
        for s in self.result.sets:
            members = sorted((by_path[img.path] for img in s.members.values()),
                             key=lambda i: order.get(i.img.channel, 99))
            self.cards.append(Card(inner, self,
                                   f"{s.prefix.rstrip('_- ')}  ·  set {s.number}",
                                   s.note, members, wrap=False))
        in_sets = {img.path for s in self.result.sets for img in s.members.values()}
        rest = [i for i in self.items if i.path not in in_sets]
        if rest:
            self.cards.append(Card(inner, self, f"Not in a complete set ({len(rest)})",
                                   "", rest, wrap=True))
        self.summary_label.configure(
            text=f"{len(self.result.sets)} set(s) detected")
        self.merge_all_btn.state(["!disabled"] if self.result.sets else ["disabled"])
        if self.focused is None and self.items:
            self.focused = self.flat_order()[0]
        self.refresh_states()
        self.layout_cards()

    def flat_order(self) -> list[Item]:
        return [t.item for c in self.cards for t in c.tiles]

    def _on_grid_resize(self, _event) -> None:
        if self._layout_pending is not None:
            self.root.after_cancel(self._layout_pending)
        self._layout_pending = self.root.after(60, self.layout_cards)

    def layout_cards(self) -> None:
        """Flow cards left-to-right, wrapping to the window width."""
        self._layout_pending = None
        width = max(self.scroll.canvas.winfo_width(), 300)
        x = y = line_h = 0
        for card in self.cards:
            card.fit(width)
            card.update_idletasks()
            w, h = card.winfo_reqwidth(), card.winfo_reqheight()
            if x > 0 and x + w > width:
                x, y, line_h = 0, y + line_h + GAP, 0
            card.place(x=x, y=y)
            x += w + GAP
            line_h = max(line_h, h)
        self.scroll.inner.configure(height=y + line_h + GAP)

    def refresh_tiles(self, items) -> None:
        for i in items:
            if i.tile is not None:
                i.tile.refresh()

    def refresh_states(self) -> None:
        for item in self.items:
            if item.tile is not None:
                item.tile.set_state(item in self.selected, item is self.focused)
        n = len(self.selected)
        if n == 0:
            text = "Nothing selected — click images to choose which ones to merge together."
        else:
            chans = [i.channel or "?" for i in self.selected_ordered()]
            text = f"{n} selected:  " + " + ".join(chans)
            dupes = {c for c in chans if chans.count(c) > 1 and c != "?"}
            if dupes:
                text += f"    (more than one {', '.join(sorted(dupes))})"
        self.selection_label.configure(text=text)
        state = ["!disabled"] if n >= 1 else ["disabled"]
        self.preview_btn.state(state)
        self.merge_btn.state(["!disabled"] if n >= 2 else ["disabled"])
        if self.viewer is not None:
            self.viewer.sync_selection()

    def selected_ordered(self) -> list[Item]:
        order = {name: n for n, name in enumerate(self.profile.channel_names)}
        return sorted(self.selected, key=lambda i: order.get(i.channel, 99))

    def toggle_select(self, item: Item) -> None:
        if item in self.selected:
            self.selected.remove(item)
        else:
            self.selected.append(item)
        self.refresh_states()

    def select_only(self, items: list[Item]) -> None:
        self.selected = list(items)
        if items:
            self.focused = items[0]
        self.refresh_states()

    def set_focus(self, item: Item | None) -> None:
        self.focused = item
        self.refresh_states()
        if item is not None and item.tile is not None and self.viewer is None:
            self.scroll.see(item.tile)

    def move_focus(self, direction: str) -> None:
        order = self.flat_order()
        if not order:
            return
        cur = self.focused if self.focused in order else order[0]
        if direction in ("Left", "Right"):
            i = order.index(cur) + (1 if direction == "Right" else -1)
            self.set_focus(order[max(0, min(i, len(order) - 1))])
            return
        # Up / Down: nearest tile in the next visual row
        t = cur.tile
        cx, cy = t.winfo_rootx() + t.winfo_width() / 2, t.winfo_rooty()
        sign = 1 if direction == "Down" else -1
        rows = [i for i in order
                if sign * (i.tile.winfo_rooty() - cy) > t.winfo_height() / 2]
        if not rows:
            return
        nearest_y = min(abs(i.tile.winfo_rooty() - cy) for i in rows)
        line = [i for i in rows if abs(abs(i.tile.winfo_rooty() - cy) - nearest_y) < 10]
        best = min(line, key=lambda i: abs(i.tile.winfo_rootx()
                                           + i.tile.winfo_width() / 2 - cx))
        self.set_focus(best)

    def clear_workspace(self) -> None:
        if self.busy:
            return
        self.close_viewer()
        for c in self.cards:
            c.destroy()
        self.cards, self.items, self.selected = [], [], []
        self.inputs, self.cache, self.result, self.focused = [], {}, None, None
        self.set_status("")
        self.show_home()

    # ---- viewer ----

    def open_single(self, item: Item) -> None:
        self._open_viewer([item], merge_mode=False)
        self.focused = item

    def open_merge(self) -> None:
        items = self.selected_ordered()
        if not items:
            return
        if not self._same_size(items):
            return
        self._open_viewer(items, merge_mode=True)

    def _open_viewer(self, items: list[Item], merge_mode: bool) -> None:
        if self.viewer is not None:
            self.viewer.destroy()
        else:
            self.grid_frame.pack_forget()
        self.viewer = Viewer(self.body, self, items, merge_mode)
        self.viewer.pack(fill="both", expand=True)
        self.take_focus()

    def close_viewer(self) -> None:
        if self.viewer is None:
            return
        self.viewer.destroy()
        self.viewer = None
        self.grid_frame.pack(fill="both", expand=True)
        self.take_focus()
        self.refresh_tiles(self.thumbs_dirty)
        self.thumbs_dirty.clear()
        self.refresh_states()
        if self.focused is not None and self.focused.tile is not None:
            self.root.after(30, lambda: self.focused and self.focused.tile
                            and self.scroll.see(self.focused.tile))

    def step_viewer(self, delta: int) -> None:
        order = self.flat_order()
        if self.viewer is None or self.viewer.merge_mode or not order:
            return
        i = order.index(self.viewer.items[0]) + delta
        if 0 <= i < len(order):
            self.open_single(order[i])

    # ---- keyboard / mouse ----

    def on_key(self, event) -> str | None:
        try:
            focus = self.root.focus_get()
        except (KeyError, tk.TclError):     # focus is in a combobox popdown
            return None
        if isinstance(focus, (ttk.Combobox, ttk.Entry, tk.Entry, tk.Text, tk.Listbox)):
            return None
        if not self.items:
            return None
        key = event.keysym
        # Control, plus ⌘ on macOS (elsewhere 0x0008 can be Alt or NumLock)
        mod = event.state & (0x000C if sys.platform == "darwin" else 0x0004)
        if key in ("a", "A") and mod:
            self.select_only(self.flat_order())
            return "break"
        if mod:
            return None
        if self.viewer is not None:
            if key in ("space", "Escape"):
                self.close_viewer()
            elif key in ("Left", "Right"):
                self.step_viewer(1 if key == "Right" else -1)
            elif key in ("h", "H"):
                self.show_clipped.set(not self.show_clipped.get())
                self.viewer.schedule()
            elif key in ("s", "S") and not self.viewer.merge_mode:
                self.toggle_select(self.viewer.items[0])
            elif key == "Return" and self.viewer.merge_mode:
                self.viewer.save()
            else:
                return None
            return "break"
        if key == "space":
            if self.focused is not None:
                self.open_single(self.focused)
        elif key in ("Left", "Right", "Up", "Down"):
            self.move_focus(key)
        elif key == "Return" and self.focused is not None:
            self.toggle_select(self.focused)
        elif key == "Escape":
            self.select_only([])
        elif key in ("p", "P"):
            self.open_merge()
        elif key in ("m", "M"):
            self.save_manual(self.selected_ordered())
        else:
            return None
        return "break"

    def on_wheel(self, event, units: int | None = None) -> None:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None and not isinstance(widget, Scrollable):
            widget = widget.master
        if widget is None:
            return
        if units is None:
            d = event.delta
            units = -d if sys.platform == "darwin" else -int(d / 120) or (-1 if d > 0 else 1)
        widget.scroll(units)

    def _refocus(self, event) -> None:
        if isinstance(event.widget, (tk.Widget, tk.Tk)) and not isinstance(
                event.widget, (ttk.Combobox, ttk.Entry, tk.Entry, tk.Text, tk.Listbox)):
            self.take_focus()

    def take_focus(self) -> None:
        """Keep keyboard focus on the main window so Space / Esc / arrows
        always reach on_key — a click on a slider, or a widget being
        destroyed while it held focus, must not swallow the next key."""
        def grab() -> None:
            try:
                self.root.focus_set()
            except tk.TclError:
                pass
        grab()
        self.root.after_idle(grab)

    # ---- saving ----

    def _same_size(self, items: list[Item]) -> bool:
        shapes = {i.data.shape for i in items}
        if len(shapes) > 1:
            self.log("Can't merge images of different sizes: " + ", ".join(
                f"{i.path.name} ({i.data.shape[1]}×{i.data.shape[0]})" for i in items))
            return False
        return True

    def output_dir(self) -> Path:
        return default_output_dir(self.inputs)

    def save_manual(self, items: list[Item]) -> None:
        if len(items) < 1 or not self._same_size(items):
            return
        if self.busy:
            self.log("Still working — please wait a moment.")
            return
        members = [(i.img, i.adjustment()) for i in items]
        scale_bar, fmt = self.scalebar_var.get(), self.format_var.get()
        fallback = self.profile.pixel_size_um
        self.busy = True
        self.set_status("Saving merge…")

        def work() -> None:
            try:
                out_dir = self.output_dir()
                path = unique_path(out_dir, manual_stem([i.path for i in items]), fmt)
                _, drew = save_composite(members, path, scale_bar, fallback)
                append_merge_log(out_dir, f"{path.name}  <-  {describe_members(members)}")
                self.events.put(("log", f"✓ Saved {path.name}  →  {out_dir}"))
                if scale_bar and not drew:
                    self.events.put(("log", "⚠ No pixel size in metadata or profile — "
                                            "scale bar skipped."))
                self.events.put(("status", f"Saved {path}"))
            except Exception as exc:
                self.events.put(("log", f"Error: {exc}"))
            finally:
                self.events.put(("idle", None))

        threading.Thread(target=work, daemon=True).start()

    def merge_all(self) -> None:
        """Merge every detected set, using whatever display adjustments were
        made in the grid (defaults otherwise)."""
        if self.busy or not self.result or not self.result.sets:
            return
        by_path = {i.path: i for i in self.items}
        jobs = []
        order = {name: n for n, name in enumerate(self.profile.channel_names)}
        for s in self.result.sets:
            members = sorted((by_path[img.path] for img in s.members.values()),
                             key=lambda i: order.get(i.channel, 99))
            jobs.append((s, [(i.img, i.adjustment()) for i in members]))
        orphans = list(self.result.orphans)
        warnings = list(self.result.warnings)
        scale_bar, fmt = self.scalebar_var.get(), self.format_var.get()
        profile, contrast = self.profile, self.contrast_var.get()
        self.busy = True
        self.merge_all_btn.state(["disabled"])

        def work() -> None:
            try:
                out_dir = self.output_dir()
                manifest, missing_px = [], False
                for n, (s, members) in enumerate(jobs, 1):
                    self.events.put(("status", f"Merging set {n} of {len(jobs)}…"))
                    name = f"{s.prefix}_set{s.number:02d}_merge.{fmt}"
                    _, drew = save_composite(members, out_dir / name, scale_bar,
                                             profile.pixel_size_um)
                    missing_px |= scale_bar and not drew
                    note = f"   [{s.note}]" if s.note else ""
                    manifest.append(f"{name}  <-  {describe_members(members)}{note}")
                    self.events.put(("log", f"✓ {name}"))
                with open(out_dir / "merge_summary.txt", "w", encoding="utf-8") as fh:
                    fh.write(f"FluoroMerge summary\nProfile: {profile.name}\n")
                    fh.write(f"Contrast: {contrast} (display range shown per file)\n")
                    fh.write(f"\nComposites ({len(manifest)}):\n")
                    fh.write("".join(f"  {line}\n" for line in manifest))
                    if orphans:
                        fh.write(f"\nNot merged ({len(orphans)}):\n")
                        for o in orphans:
                            fh.write(f"  {o.path.name} ({o.channel or 'unrecognized'})\n")
                    if warnings:
                        fh.write("\nWarnings:\n")
                        fh.write("".join(f"  {w}\n" for w in warnings))
                if missing_px:
                    self.events.put(("log", "⚠ No pixel size in metadata or profile — "
                                            "scale bars skipped."))
                self.events.put(("log", f"{len(jobs)} composite(s) written to {out_dir}"))
                self.events.put(("status", f"{len(jobs)} composite(s) written to {out_dir}"))
            except Exception as exc:
                self.events.put(("log", f"Error: {exc}"))
            finally:
                self.events.put(("idle", None))

        threading.Thread(target=work, daemon=True).start()

    # ---- events / logging ----

    def drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "status":
                    self.set_status(payload)
                elif kind == "loaded":
                    self.on_loaded(*payload)
                elif kind == "idle":
                    self.busy = False
                    if self.result and self.result.sets:
                        self.merge_all_btn.state(["!disabled"])
        except queue.Empty:
            pass
        self.root.after(50, self.drain_events)

    def log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def run(self) -> None:
        self.root.mainloop()


def launch() -> None:
    App().run()
