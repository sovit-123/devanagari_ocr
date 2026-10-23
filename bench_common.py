"""Shared machinery for the benchmark scripts in benchmarks/.

Every benchmark needs the same three things -- the test split, the
ground-truth transcriptions for its samples, and a resumable predictions
file -- and that logic lives here once, so all models run through
identical code. Ground truth is streamed from the ~2.1 GB annotations
file; predictions are appended one JSON object per line, and samples
already recorded are skipped on restart, so interrupted runs resume
instead of starting over.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from make_splits import DEFAULT_DATA_DIR, iter_annotations


def load_test_split(data_dir: Path) -> list[str]:
    """Return the image paths of the test split, in the order splits.csv lists them.

    Exits with a plain-English message when splits.csv is missing -- the
    pipeline is split-first by design, and "score whatever is on disk"
    would make numbers incomparable, so there is no fallback.
    """
    splits_path = data_dir / "splits.csv"
    if not splits_path.exists():
        raise SystemExit(
            f"No splits.csv found under {data_dir}.\n"
            "Run make_splits.py first to create the train/val/test split."
        )
    paths: list[str] = []
    with open(splits_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                paths.append(row["image_path"])
    if not paths:
        raise SystemExit(f"The test split in {splits_path} is empty; re-run make_splits.py.")
    return paths


def transcription_of(record: dict) -> str:
    """Extract the ground-truth transcription from one annotation record.

    The records use a chat format: the assistant message's content is the
    transcription. Content is normally a plain string; a list of content
    blocks is handled too, defensively, in case the publisher mixes shapes.
    """
    for message in record.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text") or "")
            return "\n".join(p for p in parts if p)
        return ""
    return ""


def collect_ground_truth(data_dir: Path, image_paths: list[str]) -> dict[str, str]:
    """Stream the annotations once; return {image_path: transcription} for the wanted paths.

    The annotations file is ~2.1 GB, so this takes a couple of minutes and
    shows a progress bar. Test samples that no annotation record mentions
    are reported (and absent from the result) -- the caller skips them.
    """
    wanted = set(image_paths)
    annotations_path = data_dir / "devanagari_ocr.json"
    truth: dict[str, str] = {}
    for record in iter_annotations(annotations_path):
        images = record.get("images") or []
        if len(images) != 1 or images[0] not in wanted:
            continue
        truth[images[0]] = transcription_of(record)
    missing = sorted(wanted - truth.keys())
    if missing:
        print(
            f"WARNING: {len(missing)} test sample(s) have no annotation record; "
            "they will be skipped."
        )
    return truth


def open_predictions(run_dir: Path) -> tuple[Path, set[str]]:
    """Prepare a model's results directory; return (predictions.jsonl, already-done paths).

    The predictions file is JSON Lines: one object per sample, appended
    over time. On startup, existing records are read back so SUCCESSFUL
    samples can be skipped -- records carrying an "error" field are NOT
    counted as done, so failed samples are retried on the next run.
    A truncated final line (a crash mid-write) is tolerated and simply
    redone on the next run.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"
    done: set[str] = set()
    if pred_path.exists():
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated tail line from an interrupted run
                if record.get("error") is None:
                    done.add(record["image_path"])
    return pred_path, done


def append_prediction(handle, record: dict) -> None:
    """Append one prediction record as a single JSON line (the caller flushes)."""
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_run_info(run_dir: Path, info: dict) -> None:
    """Write run_info.json describing the configuration of the current run.

    Overwritten at each startup, so it always describes the configuration
    that the newest predictions in predictions.jsonl were produced with.
    """
    info = {"started_utc": datetime.now(timezone.utc).isoformat(), **info}
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
