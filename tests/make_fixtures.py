"""Generate synthetic OME-TIFF fixtures that replicate the structures seen in
real microscope exports: complete triplets, an out-of-order set, a duplicated
channel (abandoned first shot), a trailing orphan, and files with no metadata.

Usage: python tests/make_fixtures.py [output_dir]
"""
from __future__ import annotations

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

    print(f"fixtures written to {out}")


if __name__ == "__main__":
    main()
