"""Download the Devanagari OCR dataset from Hugging Face -- all of it, or a slice.

The dataset (himalaya-ai/devanagari_ocr_dataset, ~60 GB) is 42 image
tarballs plus one ~2.1 GB annotations file. Since the annotations are a
single monolithic file, the smallest useful slice is either the
annotations alone or the annotations plus whole tarballs -- individual
languages, domains, or images cannot be fetched separately.

The script only downloads; extract.py unpacks, so you can fetch many
batches but unpack only the few you want. Downloads are idempotent:
re-running a command re-downloads nothing. Add --dry-run to preview.

Usage:
    python3 download_dataset.py                    # everything (~60 GB)
    python3 download_dataset.py --only-annotations # just the ~2.1 GB annotations
    python3 download_dataset.py --batches 1-3,7   # annotations + chosen batches
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

# The Hugging Face repository to download from.
REPO_ID = "himalaya-ai/devanagari_ocr_dataset"

# Files land in <project>/input/devanagari_ocr_dataset/, mirroring the
# repository layout (data/ + devanagari_ocr.json + README.md).
DEST_DIR = Path(__file__).resolve().parent / "input" / "devanagari_ocr_dataset"

# How many image tarballs the repository contains (numbered 1 through 42).
# extract.py imports this, along with the helpers below, so the batch
# selection rules stay defined in exactly one place.
NUM_BATCHES = 42

# Rough file sizes in GB, used only for the summary printed before
# downloading (real sizes vary slightly per batch).
ANNOTATIONS_SIZE_GB = 2.1
BATCH_SIZE_GB = 1.4
LAST_BATCH_SIZE_GB = 0.34


def parse_batch_spec(spec: str) -> list[int]:
    """Expand a batch selection like "1-3,7" into the sorted list [1, 2, 3, 7].

    Accepts comma-separated numbers and ranges; zero-padded forms such as
    "007" are fine too. Raises ValueError with a plain-English message when
    the spec is malformed, empty, or refers to batches outside 1..42.
    """
    batches: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                lo_n, hi_n = int(lo), int(hi)
            except ValueError:
                raise ValueError(f"'{part}' is not a valid batch range (expected e.g. 4-9)")
            if lo_n > hi_n:
                raise ValueError(f"range '{part}' is backwards (start must not exceed end)")
            batches.update(range(lo_n, hi_n + 1))
        else:
            try:
                batches.add(int(part))
            except ValueError:
                raise ValueError(f"'{part}' is not a valid batch number")

    if not batches:
        raise ValueError("no batch numbers found in --batches")
    out_of_range = sorted(b for b in batches if not 1 <= b <= NUM_BATCHES)
    if out_of_range:
        raise ValueError(
            f"batch numbers must be between 1 and {NUM_BATCHES}, got {out_of_range}"
        )
    return sorted(batches)


def batch_filename(batch: int) -> str:
    """Repository path of one tarball, e.g. batch 7 -> data/images_batch_007.tar.gz."""
    return f"data/images_batch_{batch:03d}.tar.gz"


def build_file_list(batch_numbers: list[int]) -> list[str]:
    """Repository files to download: the annotations plus the chosen tarballs."""
    files = ["devanagari_ocr.json", "README.md"]
    files.extend(batch_filename(b) for b in batch_numbers)
    return files


def estimate_size_gb(batch_numbers: list[int]) -> float:
    """Rough total download size in GB (annotations + the chosen tarballs)."""
    total = ANNOTATIONS_SIZE_GB
    for b in batch_numbers:
        total += LAST_BATCH_SIZE_GB if b == NUM_BATCHES else BATCH_SIZE_GB
    return total


def parse_args() -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the himalaya-ai/devanagari_ocr_dataset dataset from "
            "Hugging Face: everything (~60 GB), annotations only (~2.1 GB), "
            "or annotations plus selected image batches."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only-annotations",
        action="store_true",
        help="download only devanagari_ocr.json and README.md (~2.1 GB)",
    )
    group.add_argument(
        "--batches",
        metavar="SPEC",
        help=(
            "download the annotations plus these image batches; SPEC is a "
            "comma-separated list of numbers and ranges, e.g. 5 or 1-3 or 1,4,9-12"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the files that would be downloaded, then exit",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="number of files to download in parallel (default: 4)",
    )
    args = parser.parse_args()

    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    # Turn the mode flags into a concrete list of batch numbers:
    #   --only-annotations  -> no tarballs
    #   --batches SPEC      -> the chosen tarballs
    #   no flags at all     -> every tarball (full download)
    if args.only_annotations:
        args.batch_numbers = []
    elif args.batches is not None:
        try:
            args.batch_numbers = parse_batch_spec(args.batches)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        args.batch_numbers = list(range(1, NUM_BATCHES + 1))
    return args


def main() -> None:
    args = parse_args()

    files = build_file_list(args.batch_numbers)
    if args.batch_numbers:
        plan = f"annotations + batches {', '.join(map(str, args.batch_numbers))}"
    else:
        plan = "annotations only"

    print(f"Dataset: {REPO_ID}")
    print(f"Target:  {DEST_DIR}")
    print(f"Plan:    {plan}")
    print(f"Files:   {len(files)} (~{estimate_size_gb(args.batch_numbers):.1f} GB)")

    if args.dry_run:
        print("Files that would be downloaded:")
        for name in files:
            print(f"  - {name}")
        print("Dry run: nothing was downloaded.")
        return

    print("Starting download (re-run this script to resume if it is interrupted)...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DEST_DIR,
        allow_patterns=files,
        max_workers=args.max_workers,
    )
    print("Download complete. Run extract.py to unpack the image batches.")


if __name__ == "__main__":
    main()
