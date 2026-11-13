# FluoroMerge

Merge single-channel fluorescence microscopy TIFFs (e.g. DAPI / FITC / TRITC)
into color composite images, with automatic scale bars. Built for the common
workflow where the microscope saves each channel as its own TIFF and the files
just need to be combined — no ImageJ knowledge required.

## Using the app (undergrads start here)

1. Download the build for your computer (Windows / Linux / macOS) from the
   **Releases** page and unzip it.
2. Open **FluoroMerge**.
3. Pick your microscope profile from the dropdown.
4. Drag your images or image folder onto the window (or use *Choose folder…*).
5. The images open in a **review grid**. Each tile shows the channel
   (DAPI, FITC, …) and objective, and the sets FluoroMerge detected are grouped
   together in boxes. Then either:
   - click **Merge all detected sets**, or
   - click the images that belong together (any 2 or more, even from
     different boxes) and press **Merge selected**.
6. Composites appear in a `composites` folder next to your images.

Tick *Merge automatically without reviewing* on the start screen to skip the
grid and merge every detected set straight away, as earlier versions did.

### Adjusting an image before merging

Press **Space** on an image (or double-click it) to open it full size. There
you can pick its color (LUT) and set its **display range** on the histogram:
pixels at or below the black handle show as black, pixels at or above the
white handle show as full color. Drag the white handle left to brighten, the
black handle right to darken the background, or the space between them to
shift both. *Auto* stretches the image on its own, *Reset* goes back to the
starting range, and *Use for all DAPI* copies the color and range to every
image of that channel. Press Space again to go back to the grid.

The "% maxed out" readout turns orange when more than 1% of pixels are at
full brightness — detail there is lost. Tick *Highlight pixels at full
brightness* (H) to see them in orange on the image.

**Preview & adjust merge** (P) shows the selected images overlaid, with a
histogram for each layer, before you save. Untick a layer to leave it out.

Adjustments are linear and change only how the image is displayed and saved
as a composite, never the original files.

Keys in the grid: arrows move, Enter selects, Space opens, P previews,
M merges, ⌘/Ctrl-A selects all, Esc clears the selection. In the viewer,
← / → step through images, S selects and H toggles the full-brightness highlight.

Every merge records the files and the display settings it used (color and
display range per channel) in `merge_summary.txt` (merge all) or
`merge_log.txt` (merges you picked), so figures can be reproduced.

### How it decides which file is which channel

Each TIFF's embedded OME metadata records the emission wavelength it was
captured with, so files are matched to channels **by metadata, not filename
order** — sets captured in an unusual order still merge correctly.

Color-camera microscopes (e.g. Optika) that save each channel as an RGB photo
with no metadata are also supported: the channel is recognized from the
dominant color of the bright pixels, and multicolor images (overlays the
microscope software already merged) are detected and skipped. Grayscale files
with no metadata fall back to filename order (DAPI, FITC, TRITC by default);
the log reports whenever a fallback was used.

Files named with a channel word at the end — `..._s1-1_dapi.tif`,
`..._s1-2_tritc.tif`, `..._s1-3_fitc.tif` — are recognized from that word
(DAPI, Hoechst, FITC, GFP, TRITC, mCherry, … or the profile's channel names)
and grouped by the rest of the name, ignoring the shot number and `-`/`_`
differences. When hand-typed names disagree (a typo, a retake renamed
`s4-1-2-2`), files captured back-to-back — one per channel, nothing else in
between, within 10 minutes, same sample — are still merged, and the summary
flags each such set as *grouped by capture time* so you can double-check it.
Capture time comes from the file's modified date, so copy folders in a way
that keeps it (Finder and Explorer do).

Incomplete sets (e.g. a DAPI image with no matching FITC/TRITC) are **never
guessed at** — they are skipped and reported in the summary so you can
re-image or investigate.

### Contrast modes

- **Consistent across batch** (default): one brightness window per channel for
  the whole run, so signal intensity is visually comparable between images —
  use this when comparing conditions.
- **Auto per image**: each image is stretched independently — prettier for
  standalone figures, but intensities are not comparable between images.

### Scale bars

The bar length is read from the pixel size embedded in each image's metadata
(`PhysicalSizeX`). If your files don't carry one, add `pixel_size_um` to your
microscope profile (measure once with a stage micrometer for each objective).

## Microscope profiles

Profiles are plain-text `.toml` files in the `profiles` folder next to the app
(or in `~/.fluoromerge/profiles`). Each defines the channel names, the
emission-wavelength ranges used to recognize them, their display colors, and
an optional fallback pixel size. Copy `generic_dapi_fitc_tritc.toml` and edit
it to add a new scope; one profile per scope + objective works well
(e.g. `bx51_10x.toml`, `bx51_20x.toml`).

## Running from source / command line

```bash
pip install .
fluoromerge                       # launches the GUI
fluoromerge /path/to/images       # headless batch merge
fluoromerge img1.tif img2.tif img3.tif --contrast auto --format png
```

## Building the executables

Builds are automated with GitHub Actions (`.github/workflows/build.yml`):
push a tag like `v0.1.0` (or run the workflow manually from the Actions tab)
and download the Windows/Linux/macOS artifacts. To build locally instead:

```bash
pip install . pyinstaller
pyinstaller --windowed --name FluoroMerge --collect-all tkinterdnd2 \
    --add-data "profiles:profiles" run_app.py     # use ";" on Windows
```

## Development

```bash
python tests/make_fixtures.py         # generate synthetic OME-TIFF test data
python -m fluoromerge tests/fixtures  # end-to-end run on the fixtures
```

The fixtures replicate real-world quirks: out-of-order channel sets,
duplicated channels from abandoned shots, trailing orphans, files with no
metadata, color-camera captures, and inconsistently hand-typed channel-named
files (`tests/fixtures/channel_named`).
