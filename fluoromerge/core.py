"""FluoroMerge core: group single-channel fluorescence TIFFs into color composites.

Channel identity is read from OME-XML metadata (EmissionWavelength) embedded in
TIFF tag 270 whenever possible; a filename-order fallback from the active
microscope profile covers files without usable metadata. Pixel size for scale
bars comes from OME PhysicalSizeX, with a per-profile fallback.
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


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #

@dataclass
class ChannelDef:
    name: str
    emission_range: tuple[float, float]
    color: tuple[int, int, int]


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
    channel: str | None = None       # assigned during classification
    channel_source: str = "unknown"  # "metadata" | "filename-order"


def read_ome_fields(path: Path) -> tuple[float | None, float | None]:
    """Return (emission_nm, pixel_size_um) from OME-XML in tag 270, if present."""
    try:
        with Image.open(path) as im:
            desc = im.tag_v2.get(IMAGEDESCRIPTION_TAG, "") if hasattr(im, "tag_v2") else ""
    except Exception:
        return None, None
    if not isinstance(desc, str):
        try:
            desc = desc.decode("utf-8", "ignore")
        except Exception:
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


def scan_file(path: Path) -> SourceImage:
    m = re.match(r"(.+?)[_-](\d+)$", path.stem)
    prefix, index = (m.group(1), int(m.group(2))) if m else (path.stem, -1)
    em, px = read_ome_fields(path)
    return SourceImage(path=path, prefix=prefix, index=index, emission_nm=em, pixel_size_um=px)


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #

@dataclass
class MergeSet:
    prefix: str
    number: int                       # 1-based within prefix
    members: dict[str, SourceImage]   # channel name -> image

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

    no_meta = [i for i in images if i.emission_nm is None]
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
    """Walk each prefix in numeric order, completing a set once every profile
    channel is present. A repeated channel orphans the older capture (e.g. an
    abandoned first DAPI shot); leftovers at the end are orphans too."""
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
        count = 0
        for img in sorted(by_prefix[prefix], key=lambda i: i.index):
            if img.channel in pending:
                result.orphans.append(pending.pop(img.channel))
            pending[img.channel] = img
            if set(pending) == wanted:
                count += 1
                result.sets.append(MergeSet(prefix=prefix, number=count, members=pending))
                pending = {}
        result.orphans.extend(pending.values())
    return result


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _load_array(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im, dtype=np.float64)


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
        manifest.append(f"{out_name}  <-  {sources}")
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
