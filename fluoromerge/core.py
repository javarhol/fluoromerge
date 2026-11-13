"""FluoroMerge core: group single-channel fluorescence TIFFs into color composites.

Channel identity is read from OME-XML metadata (EmissionWavelength) embedded in
TIFF tag 270 whenever possible; a filename-order fallback from the active
microscope profile covers files without usable metadata. Files named with a
channel word (e.g. "..._s1-2_tritc.tif") are recognized from that word and
grouped by the rest of the name, with capture time used to rescue sets whose
names were typed inconsistently. Pixel size for scale bars comes from OME
PhysicalSizeX, with a per-profile fallback.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TIFF_EXTENSIONS = {".tif", ".tiff"}
IMAGEDESCRIPTION_TAG = 270

# Filename words that name a fluorescence channel, mapped to a nominal emission
# peak (nm) so they resolve against any profile's emission ranges.
DYE_EMISSION_NM = {
    "dapi": 461, "hoechst": 461, "bfp": 450,
    "fitc": 519, "gfp": 510, "egfp": 510, "af488": 519, "alexa488": 519,
    "tritc": 576, "af555": 580, "alexa555": 580, "af568": 603, "alexa568": 603,
    "rfp": 600, "mcherry": 610, "texasred": 615,
}

# Longest time span (seconds) of files that are grouped into a set by capture
# time alone, when their names don't agree.
RESCUE_MAX_SPAN_S = 10 * 60


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #

@dataclass
class ChannelDef:
    name: str
    emission_range: tuple[float, float]
    color: tuple[int, int, int]
    aliases: tuple[str, ...] = ()   # extra filename words naming this channel


@dataclass
class Profile:
    name: str
    channels: list[ChannelDef]
    pixel_size_um: float | None = None  # fallback when metadata has none
    black_percentile: float = 0.5
    white_percentile: float = 99.9
    path: Path | None = None

    @property
    def channel_names(self) -> list[str]:
        return [c.name for c in self.channels]

    def classify_wavelength(self, em_nm: float) -> str | None:
        for c in self.channels:
            lo, hi = c.emission_range
            if lo <= em_nm <= hi:
                return c.name
        return None

    def classify_hue(self, hue_deg: float) -> str:
        """Nearest channel by circular distance between a capture's dominant hue
        and the channel's display-color hue (for color-camera captures)."""
        import colorsys

        def dist(c: ChannelDef) -> float:
            r, g, b = (v / 255.0 for v in c.color)
            channel_hue = colorsys.rgb_to_hsv(r, g, b)[0] * 360.0
            d = abs(hue_deg - channel_hue) % 360.0
            return min(d, 360.0 - d)

        return min(self.channels, key=dist).name

    def classify_name(self, word: str) -> str | None:
        """Channel named by a filename word: a profile channel name or alias,
        else a common dye/fluorophore name mapped through its emission peak."""
        w = word.lower()
        for c in self.channels:
            if w == c.name.lower() or w in (a.lower() for a in c.aliases):
                return c.name
        if w in DYE_EMISSION_NM:
            return self.classify_wavelength(DYE_EMISSION_NM[w])
        return None

    def color_of(self, channel: str) -> tuple[int, int, int]:
        for c in self.channels:
            if c.name == channel:
                return c.color
        raise KeyError(channel)


def load_profile(path: Path) -> Profile:
    import tomllib

    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    channels = [
        ChannelDef(
            name=ch["name"],
            emission_range=(float(ch["emission_range"][0]), float(ch["emission_range"][1])),
            color=tuple(int(v) for v in ch["color"]),
            aliases=tuple(str(a) for a in ch.get("aliases", [])),
        )
        for ch in data["channels"]
    ]
    contrast = data.get("contrast", {})
    return Profile(
        name=data.get("name", path.stem),
        channels=channels,
        pixel_size_um=data.get("pixel_size_um"),
        black_percentile=float(contrast.get("black_percentile", 0.5)),
        white_percentile=float(contrast.get("white_percentile", 99.9)),
        path=path,
    )


def load_profiles(profile_dirs: list[Path]) -> list[Profile]:
    profiles = []
    for d in profile_dirs:
        if d.is_dir():
            for p in sorted(d.glob("*.toml")):
                try:
                    profiles.append(load_profile(p))
                except Exception as exc:  # a broken user profile must not kill the app
                    print(f"warning: could not load profile {p}: {exc}")
    return profiles


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #

@dataclass
class SourceImage:
    path: Path
    prefix: str          # filename minus trailing _NN index
    index: int           # trailing numeric index (files without one get -1)
    emission_nm: float | None
    pixel_size_um: float | None
    is_color: bool = False           # RGB capture from a color camera
    hue: float | None = None         # dominant hue of bright pixels (color captures)
    hue_concentration: float = 0.0   # 0..1; low = multicolor (already merged?)
    channel: str | None = None       # assigned during classification
    channel_source: str = "unknown"  # "metadata" | "filename" | "color" | "filename-order"
    mtime: float = 0.0               # file modification time (≈ capture time)
    name_base: str | None = None     # stem minus a trailing word, e.g. "..._s1-2"
    name_word: str | None = None     # that trailing word, e.g. "tritc"
    objective: str | None = None     # e.g. "10x", from metadata or filename


def read_description(path: Path) -> str:
    """TIFF ImageDescription (tag 270) as text; "" when absent or unreadable."""
    try:
        with Image.open(path) as im:
            desc = im.tag_v2.get(IMAGEDESCRIPTION_TAG, "") if hasattr(im, "tag_v2") else ""
    except Exception:
        return ""
    if not isinstance(desc, str):
        try:
            desc = desc.decode("utf-8", "ignore")
        except Exception:
            return ""
    return desc


_OBJECTIVE_RE = re.compile(r"(?<![A-Za-z0-9.])(\d{1,3})\s?[xX](?![A-Za-z])")


def detect_objective(path: Path, desc: str | None = None) -> str | None:
    """Objective magnification, e.g. "10x": OME NominalMagnification first,
    then a "10x"/"40X" token in the file or folder name."""
    desc = read_description(path) if desc is None else desc
    m = re.search(r'NominalMagnification="([\d.]+)"', desc)
    if m:
        return f"{float(m.group(1)):g}x"
    for text in (path.stem, path.parent.name):
        m = _OBJECTIVE_RE.search(text)
        if m:
            return f"{int(m.group(1))}x"
    return None


def read_ome_fields(path: Path, desc: str | None = None) -> tuple[float | None, float | None]:
    """Return (emission_nm, pixel_size_um) from OME-XML in tag 270, if present."""
    desc = read_description(path) if desc is None else desc
    if not desc:
        return None, None

    def unit_of(attr: str) -> str:
        m = re.search(attr + r'="([^"]+)"', desc)
        # Real exports sometimes store µ with a mangled encoding ("Âµm");
        # strip stray  bytes so the unit still compares equal to "µm".
        return m.group(1).replace("Â", "").strip() if m else ""

    em = None
    m = re.search(r'EmissionWavelength="([\d.]+)"', desc)
    if m:
        em = float(m.group(1))
        if unit_of("EmissionWavelengthUnit") in ("µm", "um"):
            em *= 1000.0

    px = None
    m = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', desc)
    if m:
        px = float(m.group(1))
        unit = unit_of("PhysicalSizeXUnit")
        if unit == "nm":
            px /= 1000.0
        elif unit == "mm":
            px *= 1000.0
    return em, px


MIN_HUE_CONCENTRATION = 0.85  # below this a color capture looks multicolor


def color_capture_hue(path: Path) -> tuple[float, float]:
    """Dominant hue of the brightest pixels of an RGB capture, plus its circular
    concentration (1.0 = a single pure hue, low values = a multicolor image,
    e.g. an overlay that was already merged by the microscope software)."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.float64)[::4, ::4]
    intensity = a.max(axis=2)
    bright = a[intensity >= np.percentile(intensity, 98)]
    chroma = bright / (bright.sum(axis=1, keepdims=True) + 1e-9)
    r, g, b = chroma[:, 0], chroma[:, 1], chroma[:, 2]
    hue = np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)
    sin_m, cos_m = np.sin(hue).mean(), np.cos(hue).mean()
    concentration = float(np.hypot(sin_m, cos_m))
    mean_hue = float(np.degrees(np.arctan2(sin_m, cos_m)) % 360)
    return mean_hue, concentration


def scan_file(path: Path) -> SourceImage:
    if path.stem.isdigit():
        # Purely numeric names (0001.tif, 0002.tif, ...) are one sequence;
        # use the folder name as the set prefix.
        prefix, index = path.parent.name, int(path.stem)
    else:
        m = re.match(r"(.+?)[_-](\d+)$", path.stem)
        prefix, index = (m.group(1), int(m.group(2))) if m else (path.stem, -1)
    # A trailing non-numeric word may name the channel ("..._s1-2_tritc");
    # whether it does is decided against the profile during classification.
    wm = re.match(r"(.+?)[_-]+([A-Za-z][A-Za-z0-9]*)$", path.stem)
    desc = read_description(path)
    em, px = read_ome_fields(path, desc)
    with Image.open(path) as im:
        is_color = im.mode not in ("I;16", "I;16B", "I;16L", "I", "L", "F", "1")
    img = SourceImage(path=path, prefix=prefix, index=index,
                      emission_nm=em, pixel_size_um=px, is_color=is_color,
                      mtime=path.stat().st_mtime,
                      name_base=wm.group(1) if wm else None,
                      name_word=wm.group(2) if wm else None,
                      objective=detect_objective(path, desc))
    if is_color and em is None and img.name_word is None:
        img.hue, img.hue_concentration = color_capture_hue(path)
    return img


def name_set_key(base: str) -> str:
    """Set key for a channel-named file: separators normalized and the last
    purely numeric token (the per-channel shot number) dropped, so
    "s1-1_dapi", "s1-2_tritc", "s1_3_fitc" all key to "s1", and
    "s1-1-par_dapi", "s1-2-par_tritc" key to "s1_par"."""
    tokens = [t for t in re.split(r"[-_]+", base) if t]
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i].isdigit():
            del tokens[i]
            break
    return "_".join(tokens)


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #

@dataclass
class MergeSet:
    prefix: str
    number: int                       # 1-based within prefix
    members: dict[str, SourceImage]   # channel name -> image
    note: str = ""                    # e.g. how the set was assembled

    @property
    def pixel_size_um(self) -> float | None:
        for img in self.members.values():
            if img.pixel_size_um:
                return img.pixel_size_um
        return None


@dataclass
class ScanResult:
    sets: list[MergeSet] = field(default_factory=list)
    orphans: list[SourceImage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def collect_tiffs(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(
                f for f in sorted(p.iterdir())
                if f.suffix.lower() in TIFF_EXTENSIONS and not f.name.startswith(".")
            )
        elif p.suffix.lower() in TIFF_EXTENSIONS:
            files.append(p)
    return files


def classify_images(images: list[SourceImage], profile: Profile, result: ScanResult) -> None:
    """Assign a channel to each image, from metadata or filename-order fallback."""
    with_meta = [i for i in images if i.emission_nm is not None]
    for img in with_meta:
        img.channel = profile.classify_wavelength(img.emission_nm)
        img.channel_source = "metadata"
        if img.channel is None:
            result.warnings.append(
                f"{img.path.name}: emission {img.emission_nm:.0f} nm matches no "
                f"channel in profile '{profile.name}'"
            )

    for img in images:
        if img.emission_nm is not None or not img.name_word:
            continue
        channel = profile.classify_name(img.name_word)
        if channel is None:
            continue
        img.channel = channel
        img.channel_source = "filename"
        img.prefix, img.index = name_set_key(img.name_base), -1
        if img.is_color:
            # Cross-check the name against the photo's color; don't override it.
            img.hue, img.hue_concentration = color_capture_hue(img.path)
            if img.hue_concentration >= MIN_HUE_CONCENTRATION:
                by_color = profile.classify_hue(img.hue)
                if by_color != channel:
                    result.warnings.append(
                        f"{img.path.name}: named {channel} but its color looks like "
                        f"{by_color}; using the filename"
                    )

    color_captures = [i for i in images if i.emission_nm is None and i.is_color
                      and i.channel_source != "filename"]
    for img in color_captures:
        if img.hue is None:   # skipped in scan_file: its name ended in a word
            img.hue, img.hue_concentration = color_capture_hue(img.path)
        if img.hue_concentration < MIN_HUE_CONCENTRATION:
            result.warnings.append(
                f"{img.path.name}: multicolor content (hue concentration "
                f"{img.hue_concentration:.2f}) — looks already merged; skipped"
            )
            continue
        img.channel = profile.classify_hue(img.hue)
        img.channel_source = "color"

    no_meta = [i for i in images if i.emission_nm is None and not i.is_color
               and i.channel_source != "filename"]
    if no_meta:
        # Fall back to filename order: within each prefix, assign channels
        # cyclically in the profile's channel order (DAPI, FITC, TRITC, ...).
        order = profile.channel_names
        by_prefix: dict[str, list[SourceImage]] = {}
        for img in no_meta:
            by_prefix.setdefault(img.prefix, []).append(img)
        for prefix, group in by_prefix.items():
            group.sort(key=lambda i: i.index)
            for pos, img in enumerate(group):
                img.channel = order[pos % len(order)]
                img.channel_source = "filename-order"
            result.warnings.append(
                f"{prefix}: {len(group)} file(s) had no channel metadata; assumed "
                f"filename order {' → '.join(order)}"
            )


def group_sets(images: list[SourceImage], profile: Profile) -> ScanResult:
    """Walk each prefix in shot order (numeric index, then capture time),
    completing a set once every profile channel is present. A repeated channel
    orphans the older capture (e.g. an abandoned first DAPI shot); leftovers at
    the end are orphans too. Channel-named leftovers are then given a second
    chance by capture time (see rescue_by_time)."""
    result = ScanResult()
    classify_images(images, profile, result)
    wanted = set(profile.channel_names)
    by_prefix: dict[str, list[SourceImage]] = {}
    for img in images:
        if img.channel in wanted:
            by_prefix.setdefault(img.prefix, []).append(img)
        else:
            result.orphans.append(img)

    for prefix in sorted(by_prefix):
        pending: dict[str, SourceImage] = {}
        for img in sorted(by_prefix[prefix], key=lambda i: (i.index, i.mtime, i.path.name)):
            if img.channel in pending:
                result.orphans.append(pending.pop(img.channel))
            pending[img.channel] = img
            if set(pending) == wanted:
                result.sets.append(MergeSet(prefix=prefix, number=0, members=pending))
                pending = {}
        result.orphans.extend(pending.values())

    rescue_by_time(images, profile, result)

    # Number sets within each prefix in shot order.
    def order(s: MergeSet) -> tuple:
        m = s.members.values()
        return (s.prefix, min(i.index for i in m), min(i.mtime for i in m))
    result.sets.sort(key=order)
    counts: dict[str, int] = {}
    for s in result.sets:
        counts[s.prefix] = counts.get(s.prefix, 0) + 1
        s.number = counts[s.prefix]
    return result


def _shared_sample(keys: list[str]) -> bool:
    """True when set keys agree on everything but (at most) their last token
    of the shortest key, e.g. "s38_s4_1" / "s38_s4_1_2" (a retake with a
    mistyped number) but not "s38_s4" / "s39_s1"."""
    split = [k.split("_") for k in keys]
    need = min(len(t) for t in split) - 1
    return need >= 1 and all(t[:need] == split[0][:need] for t in split)


def rescue_by_time(images: list[SourceImage], profile: Profile, result: ScanResult) -> None:
    """Channel-named orphans whose names disagree (typos, "s1_1" vs "s1-1",
    retakes renamed by hand) are still one set when they were captured
    back-to-back: one file per channel, consecutive in capture order within
    their folder with nothing else in between, within RESCUE_MAX_SPAN_S, and
    from the same sample. Every such set is reported so it can be checked."""
    wanted = set(profile.channel_names)
    k = len(wanted)
    leftover = {id(o) for o in result.orphans
                if o.channel in wanted and o.channel_source == "filename"}
    if len(leftover) < k:
        return

    rescued: set[int] = set()
    folders: dict[Path, list[SourceImage]] = {}
    for img in images:
        folders.setdefault(img.path.parent, []).append(img)
    for files in folders.values():
        timeline = sorted(files, key=lambda i: (i.mtime, i.path.name))
        i = 0
        while i + k <= len(timeline):
            run = timeline[i:i + k]
            if (all(id(r) in leftover for r in run)
                    and {r.channel for r in run} == wanted
                    and run[-1].mtime - run[0].mtime <= RESCUE_MAX_SPAN_S
                    and _shared_sample([r.prefix for r in run])):
                members = {r.channel: r for r in run}
                names = ", ".join(r.path.name for r in run)
                result.sets.append(MergeSet(
                    prefix=run[0].prefix, number=0, members=members,
                    note="grouped by capture time",
                ))
                result.warnings.append(
                    f"names disagree, grouped by capture time — please check: {names}"
                )
                rescued.update(id(r) for r in run)
                i += k
            else:
                i += 1
    result.orphans = [o for o in result.orphans if id(o) not in rescued]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _load_array(path: Path) -> np.ndarray:
    """Channel intensity as a 2D array. Color-camera captures (RGB photos of a
    single fluorescence channel) reduce to the max across R/G/B, which keeps
    full signal for any emission color including yellow-green."""
    with Image.open(path) as im:
        arr = np.array(im, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, :, :3].max(axis=2)
    return arr


def compute_fixed_windows(sets: list[MergeSet], profile: Profile) -> dict[str, tuple[float, float]]:
    """One (black, white) window per channel across the whole batch, so
    brightness stays comparable between images."""
    windows: dict[str, tuple[float, float]] = {}
    for channel in profile.channel_names:
        lows, highs = [], []
        for s in sets:
            px = _load_array(s.members[channel].path)
            lows.append(np.percentile(px, profile.black_percentile))
            highs.append(np.percentile(px, profile.white_percentile))
        lo, hi = min(lows), max(highs)
        windows[channel] = (lo, hi if hi > lo else lo + 1)
    return windows


def _scaled(px: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    lo, hi = window
    return np.clip((px - lo) / (hi - lo), 0.0, 1.0)


# Display colors (lookup tables) offered for a channel.
LUTS: dict[str, tuple[int, int, int]] = {
    "Red": (255, 0, 0), "Green": (0, 255, 0), "Blue": (0, 0, 255),
    "Cyan": (0, 255, 255), "Magenta": (255, 0, 255), "Yellow": (255, 255, 0),
    "Gray": (255, 255, 255),
}


def lut_name(color: tuple[int, int, int]) -> str:
    for name, c in LUTS.items():
        if tuple(color) == c:
            return name
    return "rgb({}, {}, {})".format(*color)


@dataclass
class Adjustment:
    """How one channel image is displayed in a composite: raw intensity
    `black` maps to dark, `white` to full `color`."""
    color: tuple[int, int, int]
    black: float
    white: float

    def describe(self) -> str:
        return f"{lut_name(self.color)} {self.black:.0f}-{self.white:.0f}"


def image_stats(px: np.ndarray, profile: Profile) -> tuple[float, float]:
    """Auto-contrast window (black, white) from the profile's percentiles."""
    lo = float(np.percentile(px, profile.black_percentile))
    hi = float(np.percentile(px, profile.white_percentile))
    return lo, (hi if hi > lo else lo + 1)


def colorize(px: np.ndarray, adj: Adjustment) -> np.ndarray:
    """One channel as a float RGB layer (0..1)."""
    scaled = _scaled(px, (adj.black, max(adj.white, adj.black + 1e-6)))
    color = np.asarray(adj.color, dtype=np.float32) / 255.0
    return scaled[:, :, None].astype(np.float32) * color[None, None, :]


def composite(layers: list[tuple[np.ndarray, Adjustment]]) -> Image.Image:
    """Additive color merge of any number of adjusted channel images
    (all the same shape)."""
    rgb: np.ndarray | None = None
    for px, adj in layers:
        layer = colorize(px, adj)
        rgb = layer if rgb is None else rgb + layer
    out = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(out)


def render_set(
    s: MergeSet,
    profile: Profile,
    windows: dict[str, tuple[float, float]] | None,
) -> Image.Image:
    """Additive color merge. `windows=None` means per-image auto contrast."""
    rgb: np.ndarray | None = None
    for channel, img in s.members.items():
        px = _load_array(img.path)
        if windows is not None:
            scaled = _scaled(px, windows[channel])
        else:
            lo = np.percentile(px, profile.black_percentile)
            hi = np.percentile(px, profile.white_percentile)
            scaled = _scaled(px, (lo, max(hi, lo + 1)))
        color = np.array(profile.color_of(channel), dtype=np.float64) / 255.0
        layer = scaled[:, :, None] * color[None, None, :]
        rgb = layer if rgb is None else rgb + layer
    out = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(out)


# --------------------------------------------------------------------------- #
# Scale bar
# --------------------------------------------------------------------------- #

_NICE_LENGTHS_UM = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000]


def pick_bar_length_um(image_width_px: int, um_per_px: float) -> float:
    target = image_width_px * 0.15 * um_per_px
    return min(_NICE_LENGTHS_UM, key=lambda v: abs(v - target))


def _find_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf",                      # bundled on most Linux
        "arialbd.ttf", "arial.ttf",                 # Windows
        "/System/Library/Fonts/Helvetica.ttc",      # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw_scale_bar(img: Image.Image, um_per_px: float) -> None:
    """White bar with a thin black halo, bottom-right corner, labeled in µm."""
    w, h = img.size
    length_um = pick_bar_length_um(w, um_per_px)
    bar_px = int(round(length_um / um_per_px))
    bar_h = max(4, h // 150)
    margin = max(20, w // 40)
    x1, y1 = w - margin, h - margin
    x0, y0 = x1 - bar_px, y1 - bar_h

    label = f"{length_um:g} µm"
    font = _find_font(max(16, h // 35))
    draw = ImageDraw.Draw(img)
    tw, th = draw.textbbox((0, 0), label, font=font)[2:]
    tx = x0 + (bar_px - tw) / 2
    ty = y0 - th - bar_h

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            draw.rectangle([x0 + dx, y0 + dy, x1 + dx, y1 + dy], fill=(0, 0, 0))
            if dx or dy:
                draw.text((tx + dx, ty + dy), label, font=font, fill=(0, 0, 0))
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
    draw.text((tx, ty), label, font=font, fill=(255, 255, 255))


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

@dataclass
class RunOptions:
    output_dir: Path | None = None      # default: <input folder>/composites
    contrast: str = "fixed"             # "fixed" | "auto"
    scale_bar: bool = True
    output_format: str = "tif"          # "tif" | "png"


def default_output_dir(inputs: list[Path]) -> Path:
    base = inputs[0] if inputs[0].is_dir() else inputs[0].parent
    candidate = base / "composites"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_test"
        probe.touch()
        probe.unlink()
        return candidate
    except OSError:
        fallback = Path.home() / "Desktop" / f"{base.name}_composites"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def unique_path(out_dir: Path, stem: str, ext: str) -> Path:
    path = out_dir / f"{stem}.{ext}"
    n = 2
    while path.exists():
        path = out_dir / f"{stem}_{n}.{ext}"
        n += 1
    return path


def manual_stem(paths: list[Path]) -> str:
    """Output name for a hand-picked set: the files' shared name prefix."""
    stems = [p.stem for p in paths]
    common = os.path.commonprefix(stems)
    if common not in stems:
        # don't end mid-token ("sample_a_0" from "sample_a_01" + "sample_a_05")
        common = re.sub(r"[^_\-\s.]*$", "", common)
    common = common.rstrip("_- .")
    return (common or stems[0]) + "_merge"


def save_composite(
    members: list[tuple[SourceImage, Adjustment]],
    out_path: Path,
    scale_bar: bool,
    fallback_pixel_size_um: float | None,
) -> tuple[Path, bool]:
    """Render full-resolution channel images with their adjustments and save.
    Returns (path, scale_bar_drawn)."""
    arrays = [(_load_array(img.path), adj) for img, adj in members]
    shapes = {a.shape for a, _ in arrays}
    if len(shapes) > 1:
        names = ", ".join(f"{img.path.name} {a.shape[1]}x{a.shape[0]}"
                          for (img, _), (a, _) in zip(members, arrays))
        raise ValueError(f"images are different sizes and can't be overlaid: {names}")
    image = composite(arrays)
    um_per_px = next((img.pixel_size_um for img, _ in members if img.pixel_size_um),
                     fallback_pixel_size_um)
    drew = False
    if scale_bar and um_per_px:
        draw_scale_bar(image, um_per_px)
        drew = True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path, drew


def describe_members(members: list[tuple[SourceImage, Adjustment]]) -> str:
    return "  ".join(
        f"{img.channel or '?'}:{img.path.name} [{adj.describe()}]" for img, adj in members
    )


def append_merge_log(out_dir: Path, line: str) -> None:
    """Record of every merge saved from the review grid, with the display
    settings used, so figures can be reproduced."""
    from datetime import datetime

    with open(out_dir / "merge_log.txt", "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now():%Y-%m-%d %H:%M}  {line}\n")


def run_merge(
    inputs: list[Path],
    profile: Profile,
    options: RunOptions,
    log=print,
) -> ScanResult:
    inputs = [Path(p) for p in inputs]
    files = collect_tiffs(inputs)
    if not files:
        raise ValueError("No .tif/.tiff files found in the selected input.")
    log(f"Scanning {len(files)} TIFF file(s) with profile '{profile.name}'...")

    images = [scan_file(f) for f in files]
    result = group_sets(images, profile)
    for w in result.warnings:
        log(f"⚠ {w}")
    if not result.sets:
        log("No complete channel sets found — nothing to merge.")
        return result

    out_dir = options.output_dir or default_output_dir(inputs)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = None
    if options.contrast == "fixed":
        log("Computing shared contrast windows across the batch...")
        windows = compute_fixed_windows(result.sets, profile)
        for ch, (lo, hi) in windows.items():
            log(f"  {ch}: {lo:.0f}–{hi:.0f}")

    manifest: list[str] = []
    missing_px_warned = False
    for s in result.sets:
        composite = render_set(s, profile, windows)
        um_per_px = s.pixel_size_um or profile.pixel_size_um
        if options.scale_bar:
            if um_per_px:
                draw_scale_bar(composite, um_per_px)
            elif not missing_px_warned:
                log("⚠ No pixel size in metadata or profile — scale bars skipped.")
                missing_px_warned = True

        ext = "png" if options.output_format == "png" else "tif"
        out_name = f"{s.prefix}_set{s.number:02d}_merge.{ext}"
        composite.save(out_dir / out_name)
        sources = "  ".join(
            f"{ch}:{img.path.name}" for ch, img in sorted(s.members.items())
        )
        note = f"   [{s.note}]" if s.note else ""
        manifest.append(f"{out_name}  <-  {sources}{note}")
        log(f"✓ {out_name}")

    summary = out_dir / "merge_summary.txt"
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write(f"FluoroMerge summary\nProfile: {profile.name}\n")
        fh.write(f"Contrast: {options.contrast}\n")
        if windows:
            for ch, (lo, hi) in windows.items():
                fh.write(f"  {ch} window: {lo:.0f}-{hi:.0f}\n")
        fh.write(f"\nComposites ({len(result.sets)}):\n")
        fh.write("".join(f"  {line}\n" for line in manifest))
        if result.orphans:
            fh.write(f"\nSkipped orphans ({len(result.orphans)}):\n")
            for o in result.orphans:
                fh.write(f"  {o.path.name} ({o.channel or 'unrecognized'})\n")
        if result.warnings:
            fh.write("\nWarnings:\n")
            fh.write("".join(f"  {w}\n" for w in result.warnings))

    log(f"\n{len(result.sets)} composite(s) written to {out_dir}")
    if result.orphans:
        log(f"Skipped {len(result.orphans)} orphan file(s) — see merge_summary.txt:")
        for o in result.orphans:
            log(f"  {o.path.name} ({o.channel or 'unrecognized'})")
    return result
