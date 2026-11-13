"""Match channel images of the same field of view by what they show.

Different stains of one field look nothing alike pixel-for-pixel (nuclei vs
cytoplasm vs puncta), but they share structure: tissue outlines, gaps, folds,
debris, and the dark holes a cytoplasmic stain leaves where nuclei are — all
at the same pixel positions, because every channel comes from the same camera.
Normalized mutual information on a small, illumination-flattened copy of each
image picks that up without assuming bright-matches-bright (the standard
measure for aligning different imaging modalities).

On real tissue sections this separated same-field from different-field pairs
by about 3x: same-field pairs scored >= 0.065, different fields <= 0.022.
SAME_FIELD sits between the two.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

SIDE = 192          # longest side of the comparison copy
SAME_FIELD = 0.04   # normalized MI above which two images are the same field
NEIGHBORS = 6       # compare each image with this many next captures in its folder
MAX_SHIFT = 1       # +-px (at SIDE) to allow for small channel-to-channel offsets
BINS = 24


@dataclass
class Signature:
    shape: tuple[int, int]     # full-resolution size
    data: np.ndarray           # SIDE-sized, log-scaled, illumination removed


def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window with reflected edges."""
    p = np.pad(a, r, mode="reflect")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def signature(px: np.ndarray) -> Signature:
    """Comparison copy of one channel image (2D intensity array)."""
    a = np.log1p(np.maximum(px - np.percentile(px, 1), 0)).astype(np.float32)
    h, w = a.shape
    scale = SIDE / max(h, w)
    small = Image.fromarray(a, mode="F").resize(
        (max(8, round(w * scale)), max(8, round(h * scale))), Image.Resampling.BOX)
    s = np.asarray(small, dtype=np.float64)
    # remove uneven illumination / slow background (~3 box passes ≈ gaussian)
    bg = s
    for _ in range(3):
        bg = _box_blur(bg, 11)
    return Signature(shape=(h, w), data=s - bg)


def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized mutual information (0 = unrelated)."""
    def q(x):
        span = np.ptp(x)
        return np.clip(((x - x.min()) / (span + 1e-12) * BINS).astype(int), 0, BINS - 1)

    joint = np.bincount((q(a) * BINS + q(b)).ravel(), minlength=BINS * BINS)
    p = joint / joint.sum()
    pa = p.reshape(BINS, BINS).sum(axis=1)
    pb = p.reshape(BINS, BINS).sum(axis=0)

    def ent(v):
        v = v[v > 0]
        return -(v * np.log(v)).sum()

    hab = ent(p)
    return float((ent(pa) + ent(pb)) / hab - 1) if hab > 0 else 0.0


def similarity(x: Signature, y: Signature) -> float:
    """Best normalized MI over small shifts; 0 for images of different sizes."""
    if x.shape != y.shape:
        return 0.0
    a, b = x.data, y.data
    best = 0.0
    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            A = a[max(dy, 0):a.shape[0] + min(dy, 0), max(dx, 0):a.shape[1] + min(dx, 0)]
            B = b[max(-dy, 0):b.shape[0] + min(-dy, 0), max(-dx, 0):b.shape[1] + min(-dx, 0)]
            best = max(best, _nmi(A, B))
    return best


def _compatible_pixel_size(a, b) -> bool:
    if a.pixel_size_um and b.pixel_size_um:
        return abs(a.pixel_size_um - b.pixel_size_um) <= 0.02 * max(a.pixel_size_um,
                                                                     b.pixel_size_um)
    return True


def match_fields(images: list, load: Callable[[Path], np.ndarray],
                 progress: Callable[[str], None] | None = None,
                 ) -> tuple[list[list], dict[tuple[int, int], float]]:
    """Group channel images that show the same field.

    `images` are SourceImage objects with a channel assigned. Each is compared
    with its next NEIGHBORS captures in the same folder (different channel,
    same size and pixel size). Groups grow from the strongest pairs down, never
    holding two images of one channel, and only while every pair in the group
    clears SAME_FIELD. Returns the groups (2+ images each) and the scores,
    keyed by (id(a), id(b))."""
    folders: dict[Path, list] = {}
    for img in images:
        if img.channel:
            folders.setdefault(img.path.parent, []).append(img)

    pairs = []
    for files in folders.values():
        files.sort(key=lambda i: (i.mtime, i.path.name))
        for i, a in enumerate(files):
            for b in files[i + 1:i + 1 + NEIGHBORS]:
                if a.channel != b.channel and _compatible_pixel_size(a, b):
                    pairs.append((a, b))

    needed = {id(x): x for pair in pairs for x in pair}
    sigs: dict[int, Signature] = {}
    for n, img in enumerate(needed.values(), 1):
        if progress:
            progress(f"Comparing images {n} of {len(needed)}: {img.path.name}")
        sigs[id(img)] = signature(load(img.path))

    scores: dict[tuple[int, int], float] = {}
    for a, b in pairs:
        s = similarity(sigs[id(a)], sigs[id(b)])
        scores[(id(a), id(b))] = scores[(id(b), id(a))] = s

    def score(a, b) -> float:
        return scores.get((id(a), id(b)), 0.0)

    group: dict[int, list] = {id(x): [x] for x in needed.values()}
    for s, a, b in sorted(((score(a, b), a, b) for a, b in pairs),
                          key=lambda t: t[0], reverse=True):
        if s < SAME_FIELD:
            break
        ga, gb = group[id(a)], group[id(b)]
        if ga is gb:
            continue
        if {x.channel for x in ga} & {x.channel for x in gb}:
            continue
        if min(score(x, y) for x, y in itertools.product(ga, gb)) < SAME_FIELD:
            continue
        merged = ga + gb
        for x in merged:
            group[id(x)] = merged

    unique = {id(g): g for g in group.values()}.values()
    return [g for g in unique if len(g) > 1], scores
