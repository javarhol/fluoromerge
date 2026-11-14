"""Generate synthetic OME-TIFF fixtures that replicate the structures seen in
real microscope exports: complete triplets, an out-of-order set, a duplicated
channel (abandoned first shot), a trailing orphan, files with no metadata,
color-camera captures, files named by channel word ("..._s1-2_tritc"), and a
messy tissue-section session whose shot order defeats file-order grouping
(messy_tissue/, for testing grouping by image content).

Usage: python tests/make_fixtures.py [output_dir]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2

W, H = 696, 520
RNG = np.random.default_rng(7)

OME_TEMPLATE = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2015-01">'
    '<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" '
    'Type="uint16" SizeX="{w}" SizeY="{h}" SizeC="1" SizeZ="1" SizeT="1" '
    'PhysicalSizeX="0.645" PhysicalSizeXUnit="µm" '
    'PhysicalSizeY="0.645" PhysicalSizeYUnit="µm">'
    '<Channel ID="Channel:0:0" EmissionWavelength="{em}" '
    'EmissionWavelengthUnit="nm" SamplesPerPixel="1"/>'
    '<TiffData/></Pixels></Image></OME>'
)


def blobs(n: int, radius: float, intensity: float, seed_shift: int) -> np.ndarray:
    """A field of gaussian blobs — stand-ins for nuclei / cell bodies."""
    rng = np.random.default_rng(100 + seed_shift)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W))
    for _ in range(n):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        img += intensity * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2))
    return img


def field(em: int, field_seed: int) -> np.ndarray:
    """One channel of one field of view. Channels of the same field share blob
    positions (same field_seed) so cross-channel correlation is realistic."""
    base = {455: blobs(120, 6, 3000, field_seed),          # nuclei
            518: blobs(40, 18, 900, field_seed) + 150,     # cytoplasmic
            615: blobs(60, 3, 1800, field_seed + 1)}[em]   # puncta (offset pattern)
    noise = RNG.normal(90, 25, (H, W))
    return np.clip(base + noise, 0, 65535).astype(np.uint16)


def save(path: Path, em: int, field_seed: int, with_meta: bool = True) -> None:
    img = Image.fromarray(field(em, field_seed))
    if with_meta:
        ifd = ImageFileDirectory_v2()
        ifd[270] = OME_TEMPLATE.format(w=W, h=H, em=em)
        img.save(path, tiffinfo=ifd)
    else:
        img.save(path)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "fixtures"
    out.mkdir(parents=True, exist_ok=True)

    # sample_a: two clean in-order sets (like most of the real data)
    for i, (em, seed) in enumerate(
        [(455, 1), (518, 1), (615, 1), (455, 2), (518, 2), (615, 2)], start=1
    ):
        save(out / f"test_sample_a_{i:02d}.tif", em, seed)

    # sample_b: out-of-order set (TRITC first), then an abandoned DAPI followed
    # by a complete set, then a trailing orphan DAPI
    save(out / "test_sample_b_01.tif", 615, 3)
    save(out / "test_sample_b_02.tif", 455, 3)
    save(out / "test_sample_b_03.tif", 518, 3)
    save(out / "test_sample_b_04.tif", 455, 4)   # abandoned: superseded by 05
    save(out / "test_sample_b_05.tif", 455, 5)
    save(out / "test_sample_b_06.tif", 518, 5)
    save(out / "test_sample_b_07.tif", 615, 5)
    save(out / "test_sample_b_08.tif", 455, 6)   # trailing orphan

    # sample_c: no OME metadata at all -> filename-order fallback
    for i, (em, seed) in enumerate([(455, 7), (518, 7), (615, 7)], start=1):
        save(out / f"test_sample_c_{i:02d}.tif", em, seed, with_meta=False)

    # color-camera folder: RGB captures with purely numeric names, one channel
    # per photo tinted in its emission color, plus an already-merged overlay
    # that the classifier must reject
    color_dir = out / "color_camera"
    color_dir.mkdir(exist_ok=True)
    tints = {455: (0.25, 0.45, 1.0), 518: (0.75, 1.0, 0.15), 615: (1.0, 0.12, 0.1)}
    layers = {}
    for i, em in enumerate([615, 455, 518], start=1):   # deliberately out of order
        mono = field(em, 8).astype(np.float64)
        mono = 255 * mono / mono.max()
        rgb = (mono[:, :, None] * np.array(tints[em])[None, None, :]).clip(0, 255)
        layers[em] = rgb
        Image.fromarray(rgb.astype(np.uint8)).save(color_dir / f"{i:04d}.tif")
    # the overlay needs comparable numbers of bright pixels in every hue, like
    # a real merged image, so build it from three equal blob fields
    overlay = np.zeros((H, W, 3))
    for tint, seed in [((0.25, 0.45, 1.0), 20), ((0.75, 1.0, 0.15), 21), ((1.0, 0.12, 0.1), 22)]:
        mono = blobs(60, 8, 255, seed)
        mono = 255 * mono / mono.max()
        overlay += mono[:, :, None] * np.array(tint)[None, None, :]
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
        color_dir / "overlay_already_merged.tif"
    )

    # channel-named folder: the channel is a word at the end of the filename
    # (hand-typed at the scope), with the inconsistencies seen in real data.
    # File times stand in for capture times; the last three names disagree
    # and must be grouped by capture time.
    named_dir = out / "channel_named"
    named_dir.mkdir(exist_ok=True)
    t0 = 1_780_000_000
    named = [
        ("slide_s1-1_dapi", 455, 9, 0), ("slide_s1-2_tritc", 615, 9, 40),
        ("slide_s1-3_fitc", 518, 9, 80),
        ("slide_s1-1-par_dapi", 455, 10, 300), ("slide_s1-2-par_tritc", 615, 10, 340),
        ("slide_s1-3-par_fitc", 518, 10, 380),
        ("slide_s2_1_dapi", 455, 11, 900), ("slide_s2_2_tritc", 615, 11, 940),
        ("slide_s2_3_fitc", 518, 11, 980),
        ("slide_s2-1-1_dapi", 455, 12, 1500), ("slide_s2-1-2-2_tritc", 615, 12, 1540),
        ("slide_s2--1-3-3_fitc", 518, 12, 1580),
    ]
    for stem, em, seed, dt in named:
        path = named_dir / f"{stem}.tif"
        save(path, em, seed, with_meta=False)
        os.utime(path, (t0 + dt, t0 + dt))

    make_messy_tissue(out / "messy_tissue")

    print(f"fixtures written to {out}")


# --------------------------------------------------------------------------- #
# Messy tissue-section folder (modeled on real 40x/10x gut sections)
# --------------------------------------------------------------------------- #

def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur via FFT (wraps at the edges, fine for texture)."""
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.fftfreq(a.shape[1])[None, :]
    k = np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(a) * k))


def tissue_field(seed: int) -> dict[int, np.ndarray]:
    """Three channels of one tissue field. Like real sections they look
    different per stain — nuclei / cytoplasm excluding nuclei / edges +
    puncta — but share the tissue outline, gaps and lumens."""
    rng = np.random.default_rng(1000 + seed)
    yy, xx = np.mgrid[0:H, 0:W]
    shape = _smooth(rng.normal(size=(H, W)), 40)
    shape = (shape - shape.mean()) / shape.std()
    tissue = 1 / (1 + np.exp(-(shape + 0.3) * 6))           # tissue with gaps
    for _ in range(4):                                        # crypt lumens
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        rx, ry = rng.uniform(25, 60), rng.uniform(20, 50)
        tissue *= 1 - np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) ** 2)
    nuclei = np.zeros((H, W))
    for _ in range(260):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        nuclei += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rng.uniform(3, 5) ** 2))
    nuclei = np.clip(nuclei, 0, 1) * tissue
    texture = 0.6 + 0.4 * _smooth(rng.normal(size=(H, W)), 6) / 0.05
    edges = np.hypot(*np.gradient(_smooth(tissue, 2)))
    puncta = (rng.random((H, W)) > 0.9993).astype(float)
    puncta = _smooth(puncta, 1.5) * 400
    noise = lambda: rng.normal(110, 20, (H, W))              # noqa: E731
    channels = {
        455: 2800 * nuclei + 250 * tissue + noise(),
        518: 900 * tissue * np.clip(texture, 0.2, 2) * (1 - 0.85 * nuclei) + noise(),
        615: 9000 * edges + 300 * tissue + 900 * puncta + noise(),
    }
    return {em: np.clip(a, 0, 65535).astype(np.uint16) for em, a in channels.items()}


TISSUE_OME = OME_TEMPLATE.replace(
    '<Image ID="Image:0">',
    '<Instrument ID="Instrument:0"><Objective ID="Objective:0" '
    'NominalMagnification="{mag}"/></Instrument><Image ID="Image:0">',
).replace('PhysicalSizeX="0.645"', 'PhysicalSizeX="{px}"').replace(
    'PhysicalSizeY="0.645"', 'PhysicalSizeY="{px}"')


def make_messy_tissue(folder: Path) -> None:
    """Shot order that defeats file-order grouping, like real sessions:

        01 DAPI  field A   (lone shot, no partners)
        02-04    field B   TRITC, FITC, DAPI
        05-07    field C   TRITC, FITC, DAPI
        08-09    field D   DAPI, FITC only (no TRITC)
        10 FITC  field E   (stray shot)
        11-13    field F   FITC, TRITC, DAPI at 10x (different pixel size)

    Expected sets: {02,03,04} {05,06,07} {08,09} {11,12,13}; 01 and 10 alone.
    Grouping by file order alone gets B, C and D wrong."""
    folder.mkdir(parents=True, exist_ok=True)
    plan = [("A", 455), ("B", 615), ("B", 518), ("B", 455), ("C", 615), ("C", 518),
            ("C", 455), ("D", 455), ("D", 518), ("E", 518), ("F", 518), ("F", 615),
            ("F", 455)]
    fields = {}
    t0 = 1_790_000_000
    for n, (name, em) in enumerate(plan, start=1):
        if name not in fields:
            fields[name] = tissue_field(ord(name))
        mag, px = (10, 0.645) if name == "F" else (40, 0.16125)
        path = folder / f"M1_sl05__{n:02d}.tif"
        ifd = ImageFileDirectory_v2()
        ifd[270] = TISSUE_OME.format(w=W, h=H, em=em, mag=mag, px=px)
        Image.fromarray(fields[name][em]).save(path, tiffinfo=ifd)
        t = t0 + n * 90
        os.utime(path, (t, t))


if __name__ == "__main__":
    main()
