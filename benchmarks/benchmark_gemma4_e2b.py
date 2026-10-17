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

This machine has an 8 GB GPU. device_map="auto" fits the E2B model, and on
smaller GPUs accelerate offloads layers to CPU automatically. --batch-size
(default 1) sets how many images share each forward pass: a batch is built
as a list of conversations, one per sample, padded together -- this
amortizes the weight reads that dominate on this class of GPU and can
give a 2-3x throughput boost on the many small word-crop images. A batch
that fails (typically out-of-memory on a large page image) is retried
one sample at a time, so it costs speed, not samples. Per-sample
elapsed_s is the batch wall time divided by the batch size -- an
approximation; the batch size is recorded in run_info.json. Batched
generation uses left padding (forced on the tokenizer after loading);
with right padding, pad tokens would land between each row's prompt and
its response and the response slice would return pad spam.

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
    python3 benchmarks/benchmark_gemma4_e2b.py --batch-size 8    # batched inference
    python3 benchmarks/benchmark_gemma4_e2b.py --max-new-tokens 1024
"""

from __future__ import annotations

import os

# The model's vision tower allocates multi-GB transients for every image
# (a fixed 2520-patch representation, regardless of image size). On an
# 8 GB GPU those repeated large allocations fragment the CUDA caching
# allocator; expandable_segments avoids exactly that (it is also what
# PyTorch's own out-of-memory message recommends). Must be set before
# torch initializes CUDA -- torch is imported later, inside main().
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "number of images per forward pass (default: 1). Batches are padded "
            "together; a failed batch is retried one sample at a time"
        ),
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
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
    # Batched generation requires LEFT padding. With the right padding some
    # tokenizers default to, pad tokens land BETWEEN each row's prompt and its
    # generated text, and slicing the response from the padded length returns
    # prompt pads instead of the model's answer -- observed as 96% <pad>-only
    # predictions in a batch_size=64 run.
    # processor.tokenizer.padding_side = "left"
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
            "batch_size": args.batch_size,
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

    # Inner worker: run the model on a chunk of samples -- one conversation
    # each, padded together -- and return one record per sample. Raises
    # whole-chunk exceptions (for example out-of-memory) for the caller to
    # retry sample by sample.
    def predict_batch(chunk: list[str]) -> list[dict]:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": str(data_dir / path)},
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ]
            for path in chunk
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # padding=True,
            add_generation_prompt=True,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        start = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        elapsed = time.perf_counter() - start

        responses = [
            processor.decode(outputs[i][prompt_len:], skip_special_tokens=False)
            for i in range(len(chunk))
        ]
        parsed = processor.parse_response(responses, prefix=inputs["input_ids"])

        return [
            {
                "image_path": path,
                "ground_truth": truth[path],
                "prediction": extract_text(message),
                "raw_response": response,
                "elapsed_s": round(elapsed / len(chunk), 3),
                "error": None,
            }
            for path, response, message in zip(chunk, responses, parsed)
        ]

    def error_record(path: str, exc: Exception) -> dict:
        return {
            "image_path": path,
            "ground_truth": truth[path],
            "prediction": None,
            "raw_response": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    consecutive_failures = 0
    written = 0
    chunks = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    with open(pred_path, "a", encoding="utf-8") as out_f:
        for chunk in tqdm(
            chunks,
            desc="Predicting",
            unit=" img" if args.batch_size == 1 else " batch",
        ):
            try:
                records = predict_batch(chunk)
            except Exception as exc:  # noqa: BLE001 -- a batch-level failure
                # (typically out-of-memory, or one poison image) is retried
                # one sample at a time to isolate the culprit.
                print(
                    f"Batch of {len(chunk)} failed ({type(exc).__name__}); "
                    "retrying its samples individually ..."
                )
                torch.cuda.empty_cache()  # release the failed attempt's reserved memory
                records = []
                for path in chunk:
                    try:
                        records.extend(predict_batch([path]))
                    except Exception as exc2:  # noqa: BLE001
                        records.append(error_record(path, exc2))
            for record in records:
                append_prediction(out_f, record)
                out_f.flush()  # crash-safe: every record is on disk immediately
                written += 1
                if record["error"] is None:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise SystemExit(
                            f"{MAX_CONSECUTIVE_FAILURES} samples failed in a row -- something "
                            "is systematically wrong (auth, prompt format, or out-of-memory). "
                            "Stopping."
                        )
            # The vision tower's multi-GB transients are per-image; freeing
            # the cached blocks keeps the next batch from fragmenting the
            # small GPU. Cheap insurance (a few milliseconds per batch).
            torch.cuda.empty_cache()

    print(f"Wrote {written:,} prediction record(s) to {pred_path}")
    print("Next step: score the predictions file with the evaluation script (CER/WER).")


if __name__ == "__main__":
    main()
