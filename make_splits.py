"""Create reproducible train/val/test splits over the Devanagari OCR samples that are on disk.

Why split at all
----------------
The dataset ships with no predefined train/validation/test split, and the
annotations file (devanagari_ocr.json, ~2.1 GB) is the fixed, authoritative
universe of samples. We always split -- even when only one image batch has
been downloaded -- so that every benchmark run, pilot or full, reports
against a dedicated test set rather than "whatever happens to be lying
around". Benchmark notebooks read this script's output and evaluate only
on its "test" rows.

How samples are chosen
---------------------
The script streams the annotations file (never loading all ~2.1 GB into
memory at once) and collects every sample whose image file exists under
the data directory. A byte-based progress bar shows how far the scan has
made it through the file -- streaming ~2.1 GB takes a couple of minutes,
and it should never happen in the dark. Image paths such as "devanagari_ocr/nepali_0.png" are
relative to the data directory, and extract.py unpacks the tarballs flat
into exactly that layout. Only downloaded-and-extracted samples enter the
split -- the split covers "as many files as there are in the directory".

Records that reference several images, or none, are skipped and counted
separately: this benchmark is one image -> one transcription, and records
with multiple images do not fit that shape.

Reproducibility
---------------
The candidate list is sorted deterministically, shuffled with
random.Random(seed) (default 42, matching the dataset card's advice), then
sliced: the first slice becomes test, the second validation, the rest
train. Same seed + same candidate set + same Python = the identical split.

Regeneration warning
--------------------
Once more batches are downloaded, re-running this script splits the
LARGER set, and the shuffle runs again over it -- so samples get
reassigned between splits. Test numbers measured under an older split
then stop being comparable to numbers from the new one. For that reason
the script refuses to overwrite an existing splits.csv unless you pass
--force, which records that you are deliberately rebasing the split.

Output
------
Writes <data-dir>/splits.csv: a header row, then one row per sample:

    image_path,split
    devanagari_ocr/nepali_0.png,test
    ...

image_path is exactly the string found in the annotation record's
"images" list, so notebooks can join it back to the ground truth.

Usage
-----
    python3 make_splits.py                    # seed 42, 80/10/10, default data dir
    python3 make_splits.py --force            # regenerate an existing split deliberately
    python3 make_splits.py --seed 7 --val-fraction 0.05 --test-fraction 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "input" / "devanagari_ocr_dataset"


def _progress_lines(f, pbar: tqdm, every_lines: int = 1000):
    """Iterate text lines while advancing a byte-based progress bar.

    The bar tracks the position of the underlying binary buffer, which
    stays byte-accurate for multi-byte UTF-8 text and for files larger
    than 2 GB (where a text-mode tell() returns an opaque cookie rather
    than a plain byte offset). Only every Nth line costs a position
    lookup, so the impact on scan speed is negligible.
    """
    pos = f.buffer.tell()
    for count, line in enumerate(f, start=1):
        if count % every_lines == 0:
            new_pos = f.buffer.tell()
            pbar.update(new_pos - pos)
            pos = new_pos
        yield line
    pbar.update(f.buffer.tell() - pos)


def iter_annotations(annotations_path: Path, progress: bool = True):
    """Yield annotation records one at a time, streaming the file.

    The file on disk is a pretty-printed JSON array: each record opens with
    a bare "{" line and closes with a bare "}" or "}," line, and nested
    objects put their own braces on bare lines too. So we track brace
    nesting depth and emit one record each time the depth returns to zero.
    This never loads the whole ~2.1 GB file into memory. A JSON-lines file
    (one object per line) is supported as well, in case the publisher ever
    changes format.

    When progress is True (the default), a tqdm bar shows how many bytes
    of the file have been scanned so far.
    """
    file_size = annotations_path.stat().st_size
    with open(annotations_path, "r", encoding="utf-8") as f:
        pbar = tqdm(
            total=file_size,
            desc="Scanning annotations",
            unit="B",
            unit_scale=True,
            disable=not progress,
        )
        try:
            first_char = ""
            while True:
                c = f.read(1)
                if c == "":
                    return  # empty file, no records
                if not c.isspace():
                    first_char = c
                    break
            if first_char == "{":
                # JSON-lines format: each non-empty line is one complete object.
                f.seek(0)
                for line in _progress_lines(f, pbar):
                    line = line.strip()
                    if line:
                        yield json.loads(line)
                pbar.update(file_size - pbar.n)
                return
            if first_char != "[":
                raise ValueError(
                    f"{annotations_path} does not look like a JSON array or a JSON-lines file"
                )
            # Pretty-printed array: assemble records between depth-0 braces.
            buffer: list[str] = []
            depth = 0
            for line in _progress_lines(f, pbar):
                stripped = line.strip()
                if not buffer:
                    if stripped == "{":
                        buffer.append("{")
                        depth = 1
                    # Lines outside records ("[", "]", ",") are ignored.
                    continue
                buffer.append(line.rstrip("\n"))
                if stripped == "{":
                    depth += 1
                elif stripped in ("}", "},"):
                    depth -= 1
                    if depth == 0:
                        text = "\n".join(buffer)
                        buffer = []
                        try:
                            yield json.loads(text)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Malformed record in {annotations_path}: {exc}"
                            ) from exc
            # The scan ran to the end of the file; snap the bar to 100%.
            pbar.update(file_size - pbar.n)
        finally:
            pbar.close()


def collect_candidates(
    annotations_path: Path, data_dir: Path, progress: bool = True
) -> tuple[list[str], dict[str, int]]:
    """Return sorted unique image paths that exist on disk, plus scan statistics.

    A record qualifies when it references exactly one image and that image
    file is present under the data directory (the annotations' paths are
    relative to it). Everything else is skipped and counted in the stats.
    The scan shows a byte-based progress bar unless progress is False.
    """
    stats = {
        "records": 0,          # every annotation record scanned
        "multi_image": 0,      # records referencing several images
        "no_image": 0,         # records referencing no image
        "duplicate_paths": 0,  # records repeating an already-seen image path
        "referenced": 0,       # unique single-image records
        "available": 0,        # of those, images actually on disk
    }
    seen: set[str] = set()
    candidates: list[str] = []
    for record in iter_annotations(annotations_path, progress=progress):
        stats["records"] += 1
        images = record.get("images") or []
        if len(images) == 0:
            stats["no_image"] += 1
            continue
        if len(images) > 1:
            stats["multi_image"] += 1
            continue
        rel_path = images[0]
        stats["referenced"] += 1
        if rel_path in seen:
            stats["duplicate_paths"] += 1
            continue
        seen.add(rel_path)
        if (data_dir / rel_path).exists():
            candidates.append(rel_path)
    stats["available"] = len(candidates)
    candidates.sort()
    return candidates, stats


def split_candidates(
    candidates: list[str], seed: int, val_fraction: float, test_fraction: float
) -> dict[str, str]:
    """Shuffle deterministically and assign each path a split name.

    The first slice of the shuffled list becomes test, the second slice
    validation, and the remainder train. The shuffle uses random.Random(seed)
    over a sorted list, so the same seed and the same set of candidates
    always produce the same assignment.
    """
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_fraction)
    n_val = round(len(shuffled) * val_fraction)
    assignment: dict[str, str] = {}
    for path in shuffled[:n_test]:
        assignment[path] = "test"
    for path in shuffled[n_test : n_test + n_val]:
        assignment[path] = "val"
    for path in shuffled[n_test + n_val :]:
        assignment[path] = "train"
    return assignment


def write_splits_csv(output_path: Path, candidates: list[str], assignment: dict[str, str]) -> None:
    """Write splits.csv in sorted path order (stable and diff-friendly)."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "split"])
        for path in candidates:
            writer.writerow([path, assignment[path]])


def parse_args() -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible train/val/test split over the downloaded "
            "Devanagari OCR samples (writes <data-dir>/splits.csv)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="dataset directory holding devanagari_ocr.json (default: input/devanagari_ocr_dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for the shuffle (default: 42, per the dataset card)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="fraction of samples for the validation split (default: 0.1)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="fraction of samples for the test split (default: 0.1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "overwrite an existing splits.csv; this rebases the split, so test "
            "numbers from the old split stop being comparable"
        ),
    )
    args = parser.parse_args()

    if args.val_fraction <= 0 or args.test_fraction <= 0:
        parser.error("--val-fraction and --test-fraction must be positive")
    if args.val_fraction + args.test_fraction >= 1.0:
        parser.error("--val-fraction and --test-fraction must leave room for a train split")
    return args


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    annotations_path = data_dir / "devanagari_ocr.json"
    output_path = data_dir / "splits.csv"

    if not annotations_path.exists():
        raise SystemExit(
            f"Annotations file not found: {annotations_path}\n"
            "Run download_dataset.py first to fetch it."
        )
    if not (data_dir / "devanagari_ocr").is_dir():
        raise SystemExit(
            f"No devanagari_ocr/ image directory under {data_dir}.\n"
            "Run download_dataset.py to download image batches, then extract.py "
            "to unpack them into <data-dir>/devanagari_ocr/."
        )
    if output_path.exists() and not args.force:
        raise SystemExit(
            f"{output_path} already exists.\n"
            "Regenerating a split reshuffles sample assignments once the set of\n"
            "downloaded images changes, so test numbers measured under the old\n"
            "split would no longer be comparable. Re-run with --force if this\n"
            "regeneration is deliberate."
        )

    print(f"Scanning {annotations_path} ...")
    candidates, stats = collect_candidates(annotations_path, data_dir)

    print(f"  annotation records scanned:  {stats['records']:,}")
    print(f"  skipped (multiple images):   {stats['multi_image']:,}")
    print(f"  skipped (no image):          {stats['no_image']:,}")
    print(f"  skipped (duplicate path):    {stats['duplicate_paths']:,}")
    print(f"  unique single-image samples: {stats['referenced']:,}")
    print(f"  images present on disk:      {stats['available']:,}")

    if not candidates:
        raise SystemExit(
            "No extracted images found under the data directory.\n"
            "Run extract.py so the image batches are unpacked into "
            "<data-dir>/devanagari_ocr/, then try again."
        )

    assignment = split_candidates(candidates, args.seed, args.val_fraction, args.test_fraction)
    write_splits_csv(output_path, candidates, assignment)

    counts = {"train": 0, "val": 0, "test": 0}
    for split in assignment.values():
        counts[split] += 1
    train_fraction = 1.0 - args.val_fraction - args.test_fraction
    print(f"\nWrote {output_path}")
    print(
        f"  seed={args.seed}   fractions: train {train_fraction:.0%} / "
        f"val {args.val_fraction:.0%} / test {args.test_fraction:.0%}"
    )
    print(
        f"  train: {counts['train']:,}   val: {counts['val']:,}   "
        f"test: {counts['test']:,}   total: {len(candidates):,}"
    )


if __name__ == "__main__":
    main()
