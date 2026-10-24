"""Benchmark Google Gemma 4 E2B (google/gemma-4-E2B-it) on the Devanagari OCR test split.

Runs the model over the test rows of <data-dir>/splits.csv and appends one
record per sample to results/<run-name>/predictions.jsonl inside this
script's directory: image path, ground truth, prediction, raw response,
and timing, or an "error" field if the sample failed.

The script never computes metrics -- scoring is evaluate.py's job, so every
model is judged by the same protocol. Records are flushed after every
sample and completed samples are skipped on restart. The model repository
is gated: accept its license on the Hugging Face website and log in
(hf auth login) before the first run downloads the weights.

--finetuned PATH loads either a LoRA/QLoRA adapter (mounted on the
unquantized base model) or a fully fine-tuned model directory; --run-name
names the results directory and is required whenever --finetuned is given.
Benchmarks always run the unquantized base to stay comparable.

Usage:
    python3 benchmarks/gemma4_e2b/bench_gemma4_e2b_hf/bench_gemma4_e2b_hf.py
    python3 benchmarks/gemma4_e2b/bench_gemma4_e2b_hf/bench_gemma4_e2b_hf.py --limit 200     # smoke test
    python3 benchmarks/gemma4_e2b/bench_gemma4_e2b_hf/bench_gemma4_e2b_hf.py --batch-size 8  # batched inference
    python3 benchmarks/gemma4_e2b/bench_gemma4_e2b_hf/bench_gemma4_e2b_hf.py --finetuned <dir> --run-name <name>
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
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tqdm import tqdm

from bench_common import (
    DEFAULT_DATA_DIR,
    append_prediction,
    collect_ground_truth,
    load_test_split,
    open_predictions,
    write_run_info,
)

MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_RUN_NAME = "gemma4_e2b_hf"
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
    parser.add_argument(
        "--finetuned",
        type=Path,
        default=None,
        help=(
            "path to a LoRA/QLoRA adapter directory or a fully fine-tuned model "
            "directory; by default the plain base model is benchmarked"
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "name of the results directory inside this script's folder "
            f'(default for plain runs: "{DEFAULT_RUN_NAME}"; required with --finetuned)'
        ),
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.finetuned is not None and args.run_name is None:
        parser.error("--run-name is required when --finetuned is given")
    if args.finetuned is not None and not args.finetuned.is_dir():
        parser.error(f"--finetuned path does not exist: {args.finetuned}")
    if args.run_name is None:
        args.run_name = DEFAULT_RUN_NAME
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
    if args.finetuned is None:
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            dtype="auto",
            device_map="auto",
        )

    if args.finetuned is not None:
        if (args.finetuned / "adapter_config.json").exists():
            print(f"Loading fine-tuned model from {args.finetuned}")
            processor = AutoProcessor.from_pretrained(args.finetuned)
            model = AutoModelForMultimodalLM.from_pretrained(
                args.finetuned,
                dtype="auto",
                device_map="cuda",
            )
        # elif (args.finetuned / "config.json").exists():
        #     from peft import PeftModel  # noqa: E402

        #     print(f"Mounting LoRA adapter from {args.finetuned}")
        #     model = PeftModel.from_pretrained(model, args.finetuned)
            # print(f"Loading fine-tuned model from {args.finetuned}")
            # processor = AutoProcessor.from_pretrained(args.finetuned)
            # model = AutoModelForMultimodalLM.from_pretrained(
                # args.finetuned,
                # dtype="auto",
                # device_map="auto",
            # )
        else:
            raise SystemExit(
                f"{args.finetuned} is neither an adapter (no adapter_config.json) "
                "nor a model directory (no config.json)."
            )

    write_run_info(
        run_dir,
        {
            "model_id": MODEL_ID,
            "run_name": args.run_name,
            "finetuned": str(args.finetuned) if args.finetuned is not None else None,
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
                # print(
                #     f"{record['image_path']}: "
                #     f"{'OK' if record['error'] is None else record['error']}"
                # )
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
