"""Benchmark Google Gemma 4 E2B (google/gemma-4-E2B-it) on the Devanagari OCR test split.

What this script does
---------------------
Runs the multimodal Gemma model over the samples of the test split in
<data-dir>/splits.csv and appends one prediction record per sample to
benchmarks/results/gemma4_e2b/predictions.jsonl. Each record carries the
sample's image path, its ground-truth transcription, the model's parsed
prediction, the raw generated response (for debugging), the inference wall
time, or -- if the sample failed -- an "error" field.

What it deliberately does NOT do
--------------------------------
Compute CER/WER. Scoring is a separate step, so that every model is judged
by one shared evaluation with a single normalization protocol; numbers
computed ad-hoc inside each benchmark would not be comparable across
models. After a run, the evaluation script consumes the predictions file.

Model specifics
---------------
google/gemma-4-E2B-it is a gated repository: you must accept its license
on the Hugging Face website and log in (hf auth login) before the first
run can download the multi-GB weights. (The local hub cache currently
holds only config stubs -- the download happens on first run.)

The inference pattern follows the official usage: the image goes through
processor.apply_chat_template() before the text prompt, generation is
greedy, and the response is decoded with special tokens KEPT so that
processor.parse_response() can strip them properly. parse_response returns
a message dict rather than a plain string, so its text is extracted; the
raw response is always stored alongside in case parsing ever misbehaves.

This machine has an 8 GB GPU. device_map="auto" fits the E2B model, and
samples are processed strictly one image at a time; on smaller GPUs,
accelerate offloads layers to CPU automatically.

The default prompt is "OCR this image without additional text" -- change it
with --prompt if you are exploring prompt sensitivity. --max-new-tokens
defaults to 512 per the official example; page-level samples with long
text may need more.

Crash safety and resuming
-------------------------
Predictions are flushed to disk after every sample, and records already
present in the file are skipped on restart, so interrupted runs resume
where they stopped. A sample that raises is recorded with an "error" field
and the run continues; but if several samples fail in a row at the start,
the script stops instead of writing error records for the whole split.

Usage
-----
    python3 benchmarks/benchmark_gemma4_e2b.py
    python3 benchmarks/benchmark_gemma4_e2b.py --limit 200       # smoke test
    python3 benchmarks/benchmark_gemma4_e2b.py --max-new-tokens 1024
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the project root importable (bench_common and make_splits live there).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm  # noqa: E402

from bench_common import (  # noqa: E402
    DEFAULT_DATA_DIR,
    RESULTS_DIR,
    append_prediction,
    collect_ground_truth,
    load_test_split,
    open_predictions,
    write_run_info,
)

MODEL_ID = "google/gemma-4-E2B-it"
RUN_NAME = "gemma4_e2b"
DEFAULT_PROMPT = "OCR this image without additional text"
DEFAULT_MAX_NEW_TOKENS = 512

# If this many samples fail in a row, something is systematically wrong
# (auth, prompt format, out-of-memory) -- stop rather than writing an
# error record for the entire test split.
MAX_CONSECUTIVE_FAILURES = 5


def extract_text(parsed) -> str:
    """Pull the transcription text out of whatever parse_response returned.

    parse_response produces a message dict (or a list of them, and for some
    templates a plain string). This walks the common shapes; if nothing
    sensible is found it returns an empty string -- the raw response is
    always stored next to it in the prediction record, so nothing is lost.
    """
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, list):
        return "\n".join(p for p in (extract_text(item) for item in parsed) if p)
    if isinstance(parsed, dict):
        content = parsed.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text") or "")
            return "\n".join(p for p in parts if p)
        if parsed.get("type") == "text":
            return parsed.get("text") or ""
    return ""


def parse_args() -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run google/gemma-4-E2B-it over the Devanagari OCR test split and "
            "record per-sample predictions (scoring happens in a separate step)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="dataset directory with splits.csv and devanagari_ocr.json "
        "(default: input/devanagari_ocr_dataset)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only run the first N test samples, in splits.csv order (smoke tests)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f'transcription prompt (default: "{DEFAULT_PROMPT}")',
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="generation cap per sample (default: 512; page-level text may need more)",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()

    # 1. Test samples: only these may be scored, per the split file.
    test_paths = load_test_split(data_dir)
    if args.limit is not None:
        test_paths = test_paths[: args.limit]
    print(f"Test samples selected: {len(test_paths):,}")

    # 2. Ground truth for those samples (streams the 2.1 GB annotations once).
    print("Streaming annotations for ground truth ...")
    truth = collect_ground_truth(data_dir, test_paths)

    # 3. Resumable predictions file.
    run_dir = RESULTS_DIR / RUN_NAME
    pred_path, done = open_predictions(run_dir)
    todo = [p for p in test_paths if p in truth and p not in done]
    print(f"Predictions file: {pred_path}")
    print(f"  already recorded: {len(done):,}    to run now: {len(todo):,}")
    if not todo:
        print("Nothing to do. Score the predictions file with the evaluation script next.")
        return

    # Import the heavy libraries only after the cheap checks have passed.
    import torch  # noqa: E402
    import transformers  # noqa: E402
    from transformers import AutoModelForMultimodalLM, AutoProcessor  # noqa: E402

    print(f"Loading {MODEL_ID} (first run downloads the weights)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
    )

    write_run_info(
        run_dir,
        {
            "model_id": MODEL_ID,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "split": "test",
            "data_dir": str(data_dir),
            "samples_selected": len(test_paths),
            "samples_already_recorded": len(done),
            "samples_to_run": len(todo),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
    )

    consecutive_failures = 0
    written = 0
    with open(pred_path, "a", encoding="utf-8") as out_f:
        for image_path in tqdm(todo, desc="Predicting", unit=" img"):
            record: dict = {
                "image_path": image_path,
                "ground_truth": truth[image_path],
                "prediction": None,
                "raw_response": None,
                "error": None,
            }
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": str(data_dir / image_path)},
                            {"type": "text", "text": args.prompt},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(model.device)
                prompt_len = inputs["input_ids"].shape[-1]

                start = time.perf_counter()
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                elapsed = time.perf_counter() - start

                response = processor.decode(
                    outputs[0][prompt_len:], skip_special_tokens=False
                )
                record["prediction"] = extract_text(
                    processor.parse_response(response, prefix=inputs["input_ids"])
                )
                record["raw_response"] = response
                record["elapsed_s"] = round(elapsed, 3)
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 -- record the failure, keep going
                record["error"] = f"{type(exc).__name__}: {exc}"
                consecutive_failures += 1

            append_prediction(out_f, record)
            out_f.flush()  # crash-safe: every record is on disk immediately
            written += 1

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise SystemExit(
                    f"{MAX_CONSECUTIVE_FAILURES} samples failed in a row -- something "
                    "is systematically wrong (auth, prompt format, or out-of-memory). "
                    "Stopping."
                )

    print(f"Wrote {written:,} prediction record(s) to {pred_path}")
    print("Next step: score the predictions file with the evaluation script (CER/WER).")


if __name__ == "__main__":
    main()
