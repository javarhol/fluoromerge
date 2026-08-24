"""Entry point: `fluoromerge` launches the GUI; passing inputs runs headless."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import RunOptions, load_profiles, run_merge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fluoromerge",
        description="Merge single-channel fluorescence TIFFs into color composites.",
    )
    parser.add_argument("inputs", nargs="*", type=Path,
                        help="Folder(s) or channel image files. Empty = launch GUI.")
    parser.add_argument("--profile", default=None,
                        help="Profile name (as shown in the GUI) or path to a .toml file")
    parser.add_argument("--out", type=Path, default=None, help="Output folder")
    parser.add_argument("--contrast", choices=["fixed", "auto"], default="fixed")
    parser.add_argument("--format", choices=["tif", "png"], default="tif")
    parser.add_argument("--no-scale-bar", action="store_true")
    args = parser.parse_args(argv)

    if not args.inputs:
        from .gui import launch
        launch()
        return 0

    from .gui import profile_dirs
    if args.profile and args.profile.endswith(".toml"):
        from .core import load_profile
        profile = load_profile(Path(args.profile))
    else:
        profiles = load_profiles(profile_dirs())
        if not profiles:
            print("No profiles found.", file=sys.stderr)
            return 1
        if args.profile:
            matches = [p for p in profiles if p.name == args.profile]
            if not matches:
                print(f"Profile '{args.profile}' not found. Available: "
                      f"{', '.join(p.name for p in profiles)}", file=sys.stderr)
                return 1
            profile = matches[0]
        else:
            profile = profiles[0]

    options = RunOptions(
        output_dir=args.out,
        contrast=args.contrast,
        scale_bar=not args.no_scale_bar,
        output_format=args.format,
    )
    result = run_merge(args.inputs, profile, options)
    return 0 if result.sets else 2


if __name__ == "__main__":
    raise SystemExit(main())
