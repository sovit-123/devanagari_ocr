"""Score benchmark predictions -- the one evaluation step shared by every model.

Benchmarks only record what a model said; the numbers happen here, so every
model is scored by identical code. Ground truth and prediction are both
normalized the same way (Unicode NFC, then whitespace collapse), and rates
are reported as percentages: grapheme-cluster CER as the headline
(Devanagari conjuncts count as one unit), codepoint CER, WER, and exact
match -- each micro (corpus-weighted) and macro (per-sample mean).
Results break down by filename prefix (nepali_, hindi_local_, ...), list
the worst samples for quick error analysis, and a metrics.json is archived
next to the predictions file. Failed samples are counted as errors and kept
out of the averages, so model failures stay visible.

Usage:
    python3 evaluate.py benchmarks/gemma4_e2b/bench_gemma4_e2b_hf/results/gemma4_e2b_hf   # one run, detailed view
    python3 evaluate.py results_dir_a results_dir_b                                     # several runs side by side
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import regex
from rapidfuzz.distance import Levenshtein

_WHITESPACE = re.compile(r"\s+")
_GRAPHEME = regex.compile(r"\X")
_TRAILING_ID = re.compile(r"[\d_]+$")

# How many of the worst samples to keep for error analysis.
WORST_TO_KEEP = 10


def normalize(text: str) -> str:
    """The shared protocol: NFC-normalize, then collapse whitespace."""
    text = unicodedata.normalize("NFC", text or "")
    return _WHITESPACE.sub(" ", text).strip()


def source_prefix(image_path: str) -> str:
    """Pseudo-domain label from the filename: strip the trailing numeric id.

    e.g. devanagari_ocr/hindi_local_1000028.png -> "hindi_local".
    """
    prefix = _TRAILING_ID.sub("", Path(image_path).stem)
    return prefix or "unknown"


def _new_accumulator() -> dict:
    return {
        "n": 0,            # scored samples
        "errors": 0,       # records carrying an "error" field
        "empty_gt": 0,      # skipped: no ground truth after normalization
        "d_graph": 0, "len_graph": 0,
        "d_char": 0, "len_char": 0,
        "d_word": 0, "len_word": 0,
        "exact": 0,
    }


def _rates(acc: dict, macro: dict | None = None) -> dict:
    """Derive micro rates (and, for the overall accumulator, macro means)."""
    def micro(dist: int, length: int):
        return round(dist / length, 6) if length else None

    n = acc["n"]
    out = {
        "samples": n,
        "errors": acc["errors"],
        "empty_ground_truth": acc["empty_gt"],
        "grapheme_cer_micro": micro(acc["d_graph"], acc["len_graph"]),
        "codepoint_cer_micro": micro(acc["d_char"], acc["len_char"]),
        "wer_micro": micro(acc["d_word"], acc["len_word"]),
        "exact_match": round(acc["exact"] / n, 6) if n else None,
    }
    if macro is not None and n:
        out["grapheme_cer_macro"] = round(sum(macro["grapheme"]) / n, 6)
        out["codepoint_cer_macro"] = round(sum(macro["char"]) / n, 6)
        out["wer_macro"] = round(sum(macro["word"]) / n, 6)
    return out


def evaluate_file(pred_path: Path) -> dict:
    """Compute every metric for one predictions file; returns the metrics dict."""
    acc = _new_accumulator()
    acc["_macro_grapheme"] = []
    acc["_macro_char"] = []
    acc["_macro_word"] = []
    per_prefix: dict[str, dict] = defaultdict(_new_accumulator)
    worst: list[tuple[float, str, str, str]] = []

    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail line from an interrupted run
            if record.get("error") is not None:
                acc["errors"] += 1
                continue

            gt = normalize(record.get("ground_truth") or "")
            pred = normalize(record.get("prediction") or "")
            if not gt:
                acc["empty_gt"] += 1
                continue

            gt_graph = _GRAPHEME.findall(gt)
            pred_graph = _GRAPHEME.findall(pred)
            gt_words = gt.split()
            pred_words = pred.split()

            d_graph = Levenshtein.distance(gt_graph, pred_graph)
            d_char = Levenshtein.distance(gt, pred)
            d_word = Levenshtein.distance(gt_words, pred_words)
            exact = int(gt == pred)

            for a in (acc, per_prefix[source_prefix(record.get("image_path", ""))]):
                a["n"] += 1
                a["d_graph"] += d_graph
                a["len_graph"] += len(gt_graph)
                a["d_char"] += d_char
                a["len_char"] += len(gt)
                a["d_word"] += d_word
                a["len_word"] += len(gt_words)
                a["exact"] += exact

            acc["_macro_grapheme"].append(d_graph / len(gt_graph))
            acc["_macro_char"].append(d_char / len(gt))
            acc["_macro_word"].append(d_word / len(gt_words))
            worst.append(
                (d_graph / len(gt_graph), record.get("image_path", "?"), gt, pred)
            )

    # Prefix-level rates are micro-only (kept simple); the macro means are
    # maintained for the overall accumulator alone.
    prefix_stats = {
        prefix: _rates(dict(p_acc)) for prefix, p_acc in per_prefix.items()
    }

    macro = {
        "grapheme": acc.pop("_macro_grapheme"),
        "char": acc.pop("_macro_char"),
        "word": acc.pop("_macro_word"),
    }
    overall = _rates(acc, macro=macro)

    worst.sort(reverse=True)
    return {
        "predictions_file": str(pred_path),
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "normalization": "NFC + whitespace collapse",
        "overall": overall,
        "per_prefix": prefix_stats,
        "worst_samples": [
            {"image_path": p, "grapheme_cer": round(c, 6), "ground_truth": g, "prediction": pr}
            for c, p, g, pr in worst[:WORST_TO_KEEP]
        ],
    }


def _fmt(rate) -> str:
    """Format a rate (fraction) as a percentage, or n/a when undefined."""
    return "n/a" if rate is None else f"{rate * 100:.2f}%"


def print_detail(label: str, metrics: dict) -> None:
    overall = metrics["overall"]
    print(f"=== {label} ===")
    print(
        f"samples scored: {overall['samples']:,}   inference errors: {overall['errors']:,}"
        f"   empty ground truth: {overall['empty_ground_truth']:,}"
    )
    if not overall["samples"]:
        print("No scored samples -- nothing more to report.")
        return
    print()
    print("Overall (grapheme CER is the headline number):")
    print(f"  grapheme CER : micro {_fmt(overall['grapheme_cer_micro'])}   macro {_fmt(overall['grapheme_cer_macro'])}")
    print(f"  codepoint CER: micro {_fmt(overall['codepoint_cer_micro'])}   macro {_fmt(overall['codepoint_cer_macro'])}")
    print(f"  WER          : micro {_fmt(overall['wer_micro'])}   macro {_fmt(overall['wer_macro'])}")
    print(f"  exact match  : {_fmt(overall['exact_match'])}")

    print()
    print("Per source prefix:")
    print(f"  {'prefix':<16} {'n':>7} {'gCER micro':>11} {'WER micro':>11} {'exact':>8}")
    for prefix, stats in sorted(metrics["per_prefix"].items(), key=lambda kv: -kv[1]["samples"]):
        print(
            f"  {prefix:<16} {stats['samples']:>7,} "
            f"{_fmt(stats['grapheme_cer_micro']):>11} {_fmt(stats['wer_micro']):>11} "
            f"{_fmt(stats['exact_match']):>8}"
        )

    if metrics["worst_samples"]:
        print()
        print("Worst samples (grapheme CER):")
        for w in metrics["worst_samples"][:5]:
            print(f"  {_fmt(w['grapheme_cer'])}  {w['image_path']}")
            print(f"        gt : {w['ground_truth']}")
            print(f"        pred: {w['prediction']}")


def print_comparison(rows: list[tuple[str, dict]]) -> None:
    print(f"{'model':<22} {'scored':>8} {'errors':>7} {'gCER micro':>11} {'gCER macro':>11} {'WER micro':>11} {'exact':>8}")
    for label, m in rows:
        o = m["overall"]
        print(
            f"{label:<22} {o['samples']:>8,} {o['errors']:>7,} "
            f"{_fmt(o.get('grapheme_cer_micro')):>11} {_fmt(o.get('grapheme_cer_macro')):>11} "
            f"{_fmt(o.get('wer_micro')):>11} {_fmt(o.get('exact_match')):>8}"
        )


def resolve_targets(paths: list[Path]) -> list[tuple[str, Path]]:
    """Turn explicit CLI targets into (label, predictions-file) pairs.

    A target is a results directory holding a predictions.jsonl, or a
    predictions file directly. There are no defaults and no discovery --
    what you pass is what gets scored.
    """
    resolved = []
    for path in paths:
        if path.is_dir():
            path = path / "predictions.jsonl"
        if not path.exists():
            raise SystemExit(f"Predictions file not found: {path}")
        resolved.append((path.parent.name, path))
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score benchmark predictions with the shared protocol "
            "(NFC + whitespace collapse): grapheme/codepoint CER, WER, exact match, "
            "per-source-prefix breakdown."
        )
    )
    parser.add_argument(
        "targets",
        nargs="+",
        type=Path,
        help="results directory(ies) or predictions file(s) to score; required -- no discovery, no defaults",
    )
    return parser.parse_args()


def main() -> None:
    resolved = resolve_targets(parse_args().targets)
    rows = []
    for label, path in resolved:
        metrics = evaluate_file(path)
        out_path = path.parent / "metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        rows.append((label, metrics))
        print(f"scored {label}: {metrics['overall']['samples']:,} samples -> {out_path}")

    print()
    if len(rows) == 1:
        print_detail(rows[0][0], rows[0][1])
    else:
        print_comparison(rows)
        print()
        print("Run with a single model directory for the detailed per-prefix view.")


if __name__ == "__main__":
    main()
