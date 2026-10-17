"""Download the Devanagari OCR dataset from Hugging Face -- all of it, or just a slice.

The dataset (himalaya-ai/devanagari_ocr_dataset, ~60 GB in total) is stored
as plain files rather than a streaming-ready dataset:

  - data/images_batch_001.tar.gz ... data/images_batch_042.tar.gz
        42 compressed tarballs of images. Batches 001-041 hold ~1.4 GB
        each and batch 042 holds ~0.34 GB (~58 GB combined).
  - devanagari_ocr.json
        A single annotation file (~2.1 GB) containing every sample: one
        chat-style "messages" conversation (the user asks for the Devanagari
        transcription, the assistant reply holds the ground-truth text) and
        an "images" list with paths like "devanagari_ocr/nepali_0.png".
        Note that this is the format the file ACTUALLY uses; the dataset
        card describes an older ShareGPT-style schema that no longer
        matches the data.
  - README.md
        The dataset card.

Because the annotations live in one monolithic file and the images are
packed into opaque tarballs, the smallest useful slice you can download is
either the annotations alone, or the annotations plus one or more whole
tarballs. It is not possible to download only one language, one domain
(say, handwriting), or individual images.

This script only DOWNLOADS; it does not extract anything. Extraction is
deliberately a separate step (extract.py) so you can fetch many batches at
once but unpack only the few you want to split, train, or benchmark on --
tarballs stay compressed on disk until you decide otherwise. Downloads are
idempotent: snapshot_download() verifies files it has already fetched, so
re-running the same command re-downloads nothing.

This script supports three modes:

  1. Download everything (the default):

         python3 download_dataset.py

     Fetches all 42 tarballs plus the annotations (~60 GB).

  2. Download only the annotations:

         python3 download_dataset.py --only-annotations

     Fetches just devanagari_ocr.json and README.md (~2.1 GB).

  3. Download the annotations plus selected batches:

         python3 download_dataset.py --batches 1-3,7

     --batches accepts comma-separated numbers and ranges (for example
     "5", "1-3", or "1,4,9-12"); each batch adds roughly 1.4 GB. The
     annotations are always included, since image tarballs are of little
     use without them.

Typical workflow across the three scripts:

    python3 download_dataset.py --batches 1-5   # fetch (network-bound)
    python3 extract.py --batches 1-2            # unpack a few (disk-bound)
    python3 make_splits.py                      # split what is unpacked

Add --dry-run to any command to print the files that would be fetched,
without downloading anything. The repository is public, so no Hugging
Face login or token is required.
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
