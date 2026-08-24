"""FluoroMerge GUI: drop a folder (or a set of channel images) to merge."""
from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from .core import Profile, RunOptions, load_profiles, run_merge

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAVE_DND = True
except ImportError:
    HAVE_DND = False


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


class App:
    def __init__(self) -> None:
        root = TkinterDnD.Tk() if HAVE_DND else tk.Tk()
        self.root = root
        root.title("FluoroMerge")
        root.geometry("640x560")
        root.minsize(520, 460)

        self.profiles: list[Profile] = load_profiles(profile_dirs())
        if not self.profiles:
            raise SystemExit(
                "No microscope profiles found. Put at least one .toml profile in "
                "a 'profiles' folder next to the app."
            )

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.running = False

        main = ttk.Frame(root, padding=12)
        main.pack(fill="both", expand=True)

        # --- options row ---
        opts = ttk.Frame(main)
        opts.pack(fill="x", pady=(0, 8))

        ttk.Label(opts, text="Microscope profile:").grid(row=0, column=0, sticky="w")
        self.profile_var = tk.StringVar(value=self.profiles[0].name)
        ttk.Combobox(
            opts, textvariable=self.profile_var, state="readonly", width=34,
            values=[p.name for p in self.profiles],
        ).grid(row=0, column=1, columnspan=3, sticky="w", padx=(6, 0))

        ttk.Label(opts, text="Contrast:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.contrast_var = tk.StringVar(value="fixed")
        ttk.Radiobutton(
            opts, text="Consistent across batch", value="fixed", variable=self.contrast_var,
        ).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Radiobutton(
            opts, text="Auto per image", value="auto", variable=self.contrast_var,
        ).grid(row=1, column=2, sticky="w", pady=(6, 0))

        self.scalebar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Add scale bar", variable=self.scalebar_var).grid(
            row=2, column=1, sticky="w", padx=(6, 0), pady=(6, 0)
        )
        ttk.Label(opts, text="Save as:").grid(row=2, column=2, sticky="e", pady=(6, 0))
        self.format_var = tk.StringVar(value="tif")
        ttk.Combobox(
            opts, textvariable=self.format_var, state="readonly", width=5,
            values=["tif", "png"],
        ).grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(6, 0))

        # --- drop zone ---
        drop_text = (
            "Drop an image folder here\n(or a set of channel images)"
            if HAVE_DND
            else "Use the buttons below to choose images"
        )
        self.drop = tk.Label(
            main, text=drop_text, relief="ridge", bd=2, height=6,
            font=("TkDefaultFont", 13),
        )
        self.drop.pack(fill="x", pady=(0, 8))
        if HAVE_DND:
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self.on_drop)

        buttons = ttk.Frame(main)
        buttons.pack(fill="x", pady=(0, 8))
        ttk.Button(buttons, text="Choose folder…", command=self.pick_folder).pack(
            side="left"
        )
        ttk.Button(buttons, text="Choose images…", command=self.pick_files).pack(
            side="left", padx=(8, 0)
        )

        # --- log ---
        self.log_box = tk.Text(main, height=14, state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True)

        root.after(100, self.drain_log)

    # ---- input handlers ----

    def on_drop(self, event) -> None:
        paths = [Path(p) for p in self.root.tk.splitlist(event.data)]
        self.start(paths)

    def pick_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Choose the folder of channel images")
        if chosen:
            self.start([Path(chosen)])

    def pick_files(self) -> None:
        chosen = filedialog.askopenfilenames(
            title="Choose the channel images for one field of view",
            filetypes=[("TIFF images", "*.tif *.tiff")],
        )
        if chosen:
            self.start([Path(p) for p in chosen])

    # ---- run ----

    def start(self, paths: list[Path]) -> None:
        if self.running:
            self.log("A merge is already running — please wait.")
            return
        profile = next(p for p in self.profiles if p.name == self.profile_var.get())
        options = RunOptions(
            contrast=self.contrast_var.get(),
            scale_bar=self.scalebar_var.get(),
            output_format=self.format_var.get(),
        )
        self.running = True
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        def work() -> None:
            try:
                run_merge(paths, profile, options, log=self.log_queue.put)
            except Exception as exc:
                self.log_queue.put(f"Error: {exc}")
            finally:
                self.running = False

        threading.Thread(target=work, daemon=True).start()

    # ---- logging ----

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def drain_log(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log_box.configure(state="normal")
                self.log_box.insert("end", line + "\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self.drain_log)

    def run(self) -> None:
        self.root.mainloop()


def launch() -> None:
    App().run()
