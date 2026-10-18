"""Extract downloaded Devanagari OCR image tarballs into the combined images directory.

Tarballs are unpacked flat into <data-dir>/devanagari_ocr/, which is
exactly where the annotations point, so all extracted batches share one
namespace. Every completed batch leaves a marker file behind, so
re-running only extracts what is missing; --force re-extracts selected
batches anyway. Requesting a batch that was never downloaded prints a
warning and exits with code 1 while still extracting the rest.

Usage:
    python3 extract.py                    # every tarball already downloaded
    python3 extract.py --batches 1-3      # extract only these batches
    python3 extract.py --dry-run          # show the plan, change nothing
    python3 extract.py --force           # re-extract, ignoring markers
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

# The batch-spec syntax and tarball naming rules live with the downloader,
# which owns the description of the repository layout.
from download_dataset import batch_filename, parse_batch_spec

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "input" / "devanagari_ocr_dataset"


def marker_path(data_dir: Path, tar_name: str) -> Path:
    """Marker file that records one tarball as fully extracted."""
    stem = Path(tar_name).name.removesuffix(".tar.gz")
    return data_dir / "devanagari_ocr" / f".extracted_{stem}"


def extract_batch_tarball(
    tar_path: Path, images_dir: Path, progress: bool = True
) -> tuple[int, int]:
    """Unpack one tarball flat into images_dir; return (written, overwritten).

    Members are written under their own basenames, matching the annotation
    paths ("devanagari_ocr/<filename>" relative to the data directory).
    Existing files are counted and overwritten: an interrupted extraction
    leaves no marker, so the whole batch is simply unpacked again next run.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    overwritten = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tqdm(
            tar,
            desc=f"Extracting {tar_path.name}",
            unit=" img",
            disable=not progress,
        ):
            if not member.isfile():
                continue
            target = images_dir / Path(member.name).name
            if target.exists():
                overwritten += 1
            source = tar.extractfile(member)
            if source is None:  # defensive; regular-file members always open
                continue
            with open(target, "wb") as out:
                shutil.copyfileobj(source, out, length=1 << 20)
            written += 1
    return written, overwritten


def parse_args() -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract downloaded Devanagari OCR image batches into the combined "
            "devanagari_ocr/ directory (idempotent: already-extracted batches "
            "are skipped unless --force is given)."
        )
    )
    parser.add_argument(
        "--batches",
        metavar="SPEC",
        help=(
            "extract only these batches; SPEC is a comma-separated list of "
            "numbers and ranges, e.g. 5 or 1-3 or 1,4,9-12 (default: every "
            "tarball currently present in <data-dir>/data/)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="dataset directory holding data/ (default: input/devanagari_ocr_dataset)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be extracted and exit (nothing is unpacked)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-extract selected batches even if marker files say they are done",
    )
    args = parser.parse_args()

    if args.batches is not None:
        try:
            args.batch_numbers = parse_batch_spec(args.batches)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        args.batch_numbers = None
    return args


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    images_dir = data_dir / "devanagari_ocr"
    tar_dir = data_dir / "data"

    # Which tarballs does this run cover? Either the named ones, or every
    # tarball currently on disk when --batches is not given.
    if args.batch_numbers is not None:
        selected = [Path(batch_filename(b)).name for b in args.batch_numbers]
    else:
        selected = sorted(p.name for p in tar_dir.glob("images_batch_*.tar.gz"))
        if not selected:
            raise SystemExit(
                f"No image tarballs found under {tar_dir}.\n"
                "Run download_dataset.py first to fetch some."
            )

    # Classify the selection. The marker check comes first, so a batch whose
    # tarball was deleted (to save space) still counts as extracted.
    pending: list[str] = []    # downloaded, not extracted yet (or --force)
    extracted: list[str] = []  # marker present and --force not given
    missing: list[str] = []    # named but never downloaded
    for name in selected:
        if marker_path(data_dir, name).exists() and not args.force:
            extracted.append(name)
        elif not (tar_dir / name).exists():
            missing.append(name)
        else:
            pending.append(name)

    print(f"Data dir:   {data_dir}")
    print(f"Images dir: {images_dir}")
    print(f"Selected:   {len(selected)} tarball(s)"
          + (" (--force: markers ignored)" if args.force else ""))

    if args.dry_run:
        if pending:
            names = ", ".join(pending)
            print(f"Would extract {len(pending)}: {names}")
        if extracted:
            names = ", ".join(extracted)
            print(f"Already extracted, would skip {len(extracted)}: {names}")
        if missing:
            names = ", ".join(missing)
            print(f"Not downloaded (would warn): {names}")
        print("Dry run: nothing was extracted.")
        return

    for name in pending:
        written, overwritten = extract_batch_tarball(tar_dir / name, images_dir)
        marker_path(data_dir, name).write_text(
            f"tarball: {name}\n"
            f"files: {written}\n"
            f"completed: {datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
        note = f" ({overwritten:,} existing files overwritten)" if overwritten else ""
        print(f"{name}: {written:,} images written{note}")

    if extracted:
        print(f"Skipped {len(extracted)} already-extracted tarball(s).")
    if missing:
        names = ", ".join(missing)
        print(f"WARNING: requested but not downloaded: {names}")
        print("Run download_dataset.py --batches ... to fetch them first.")
        sys.exit(1)
    if not pending:
        print("Nothing to extract: every selected tarball is already unpacked.")


if __name__ == "__main__":
    main()
