"""Copy bundled tutorial notebooks to a user-specified directory.

Notebooks are shipped inside the ``tabkernels`` wheel under
``tabkernels/_notebooks``. After ``pip install tabkernels`` users can run::

    tabkernels-notebooks --copy ~/work
    tabkernels-notebooks --list

to extract the 26 tutorial notebooks that accompany the monograph.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from importlib import resources
from pathlib import Path


def _notebooks_root() -> Path:
    """Return the directory containing the bundled .ipynb files."""
    # importlib.resources.files() works for the package data directory.
    return Path(str(resources.files("tabkernels").joinpath("_notebooks")))


def list_notebooks() -> list[Path]:
    root = _notebooks_root()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.suffix == ".ipynb")


def copy_to(dest: Path, *, overwrite: bool = False) -> list[Path]:
    """Copy every bundled notebook into ``dest``.

    Returns the list of destination paths written.
    """
    notebooks = list_notebooks()
    if not notebooks:
        raise FileNotFoundError(
            "No bundled notebooks found. The package may have been installed "
            "without notebook data; install from a recent wheel or sdist."
        )
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for src in notebooks:
        target = dest / src.name
        if target.exists() and not overwrite:
            print(f"skip (exists): {target}", file=sys.stderr)
            continue
        shutil.copy2(src, target)
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tabkernels-notebooks",
        description="Copy the bundled tabkernels tutorial notebooks to a directory.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--copy",
        metavar="DEST",
        help="Copy all bundled notebooks into DEST (created if missing).",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List bundled notebook filenames and exit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing notebooks at the destination.",
    )
    args = parser.parse_args(argv)

    if args.list:
        nbs = list_notebooks()
        if not nbs:
            print("No bundled notebooks found.", file=sys.stderr)
            return 1
        for nb in nbs:
            print(nb.name)
        return 0

    dest = Path(args.copy).expanduser().resolve()
    written = copy_to(dest, overwrite=args.overwrite)
    print(f"Copied {len(written)} notebook(s) to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
