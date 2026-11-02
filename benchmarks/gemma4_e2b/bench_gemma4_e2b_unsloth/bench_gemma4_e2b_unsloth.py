"""Benchmark Google Gemma 4 E2B on the Devanagari OCR test split, using Unsloth.

Runs the model over the test rows of <data-dir>/splits.csv and appends one
record per sample to results/<run-name>/predictions.jsonl inside this
script's directory -- the same JSONL contract as the pure-transformers
benchmark, so evaluate.py scores both identically.

--model-id accepts either the base repository or a fine-tuned adapter
directory: Unsloth loads trained adapters directly through the same
from_pretrained call, so no separate adapter flag is needed. Generation is
greedy (no sampling) to match the benchmark protocol. Benchmarks run the
unquantized model by default; --use-4bit exists for GPUs that cannot host
the bf16 load, at the cost of strict comparability.

Usage:
    python3 benchmarks/gemma4_e2b/bench_gemma4_e2b_unsloth/bench_gemma4_e2b_unsloth.py
    python3 benchmarks/gemma4_e2b/bench_gemma4_e2b_unsloth/bench_gemma4_e2b_unsloth.py --limit 200   # smoke test
    python3 benchmarks/gemma4_e2b/bench_gemma4_e2b_unsloth/bench_gemma4_e2b_unsloth.py --model-id <adapter_dir> --run-name <name>   # fine-tuned adapter
"""

from __future__ import annotations

import os

# The model's vision tower allocates multi-GB transients for every image
# (a fixed 2520-patch representation, regardless of image size). On small
# GPUs those repeated large allocations fragment the CUDA caching allocator;
# expandable_segments avoids exactly that. Must be set before torch
# initializes CUDA -- torch is imported later, inside main().
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
import time
from pathlib import Path

# Make the project root importable (bench_common lives there).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tqdm.auto import tqdm

from bench_common import (
    DEFAULT_DATA_DIR,
    append_prediction,
    collect_ground_truth,
    load_test_split,
    open_predictions,
    write_run_info,
)

MODEL_ID = "unsloth/gemma-4-E2B-it"
DEFAULT_RUN_NAME = "gemma4_e2b_unsloth"
DEFAULT_PROMPT = "Do OCR on this image without additional text. Just the text in the image and nothing else."
DEFAULT_MAX_NEW_TOKENS = 512

# If this many samples fail in a row, something is systematically wrong
# (auth, prompt format, out-of-memory) -- stop rather than writing an
# error record for the entire test split.
MAX_CONSECUTIVE_FAILURES = 5


def parse_args() -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Gemma 4 E2B (Unsloth) over the Devanagari OCR test split and "
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
        "--model-id",
        default=MODEL_ID,
        help="base model repository or fine-tuned adapter directory "
        f"(default: {MODEL_ID})",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help=f"name of the results directory inside this script's folder (default: {DEFAULT_RUN_NAME}; "
        "use a distinct name when benchmarking a fine-tuned model)",
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
        "--use-4bit",
        action="store_true",
        help="load the model in 4-bit precision for GPUs that cannot host the bf16 "
        "load; makes runs NOT strictly comparable to unquantized ones",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if not args.run_name:
        parser.error("--run-name must not be empty")
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

    # 3. Resumable predictions file, inside this script's own directory.
    results_root = Path(__file__).resolve().parent / "results"
    run_dir = results_root / args.run_name
    pred_path, done = open_predictions(run_dir)
    todo = [p for p in test_paths if p in truth and p not in done]
    print(f"Run name       : {args.run_name}")
    print(f"Predictions file: {pred_path}")
    print(f"  already recorded: {len(done):,}    to run now: {len(todo):,}")
    if not todo:
        print("Nothing to do. Score the predictions file with the evaluation script next.")
        return

    # Import the heavy libraries only after the cheap checks have passed.
    import torch
    import transformers
    from unsloth import FastVisionModel

    print(f"Loading {args.model_id} (first run downloads the weights)...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(args.model_id),
        dtype=None,
        max_seq_length=512,
        load_in_4bit=args.use_4bit,
        full_finetuning=False,
    )
    FastVisionModel.for_inference(model)

    write_run_info(
        run_dir,
        {
            "model_id": str(args.model_id),
            "run_name": args.run_name,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "use_4bit": args.use_4bit,
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
                image = str(data_dir / image_path)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": args.prompt},
                        ],
                    }
                ]
                input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                inputs = tokenizer(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to(model.device)
                prompt_len = inputs["input_ids"].shape[-1]

                start = time.perf_counter()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                elapsed = time.perf_counter() - start

                record["raw_response"] = tokenizer.decode(
                    outputs[0][prompt_len:], skip_special_tokens=False
                )
                record["prediction"] = tokenizer.decode(
                    outputs[0][prompt_len:], skip_special_tokens=True
                ).strip()
                record["elapsed_s"] = round(elapsed, 3)
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 -- record the failure, keep going
                record["error"] = f"{type(exc).__name__}: {exc}"
                consecutive_failures += 1
                torch.cuda.empty_cache()  # release the failed attempt's reserved memory

            append_prediction(out_f, record)
            out_f.flush()  # crash-safe: every record is on disk immediately
            written += 1

            # The vision tower's multi-GB transients are per-image; freeing
            # the cached blocks keeps the next sample from fragmenting a
            # small GPU. Cheap insurance (a few milliseconds per sample).
            torch.cuda.empty_cache()

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
