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

--finetuned PATH loads either a LoRA adapter or a fully fine-tuned model
directory; --run-name names the results directory and is required whenever
--finetuned is given. Adapters load their recorded base and trained weights
on CPU first, then dispatch the complete model so offload hooks include LoRA.
Use --debug to stop on the first inference failure with its full traceback.
Use --device-map cuda:0 only if the whole model and inference working memory
fit on that GPU. For adapters, auto uses GPU 0 plus CPU offloading, with a
GPU placement budget of at most 6 GiB and enough host RAM for CPU assembly.
Base and full-model checkpoints retain ordinary device_map="auto" loading.

Usage:
    python3 benchmarks/gemma4_e2b/bench_gemma4_e2b_hf/bench_gemma4_e2b_hf.py --batch-size 1 --prompt "Do OCR on this image without additional text. Just the text in the image and nothing else."
    python3 benchmarks/gemma4_e2b/bench_gemma4_e2b_hf/bench_gemma4_e2b_hf.py --batch-size 1 --limit 200 --prompt "Do OCR on this image without additional text. Just the text in the image and nothing else."    # smoke test
    python3 benchmarks/gemma4_e2b/bench_gemma4_e2b_hf/bench_gemma4_e2b_hf.py --batch-size 8 --prompt "Do OCR on this image without additional text. Just the text in the image and nothing else."  # batched inference
    python3 benchmarks/gemma4_e2b/bench_gemma4_e2b_hf/bench_gemma4_e2b_hf.py --prompt "Do OCR on this image without additional text. Just the text in the image and nothing else." --finetuned <dir> --run-name <name>
"""

from __future__ import annotations

import os

# Can reduce allocator fragmentation across repeated image batches.
# Must be set before torch initializes CUDA; torch is imported in main().
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from importlib.metadata import PackageNotFoundError, version  # noqa: E402
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
# (model loading, device placement, prompt format, out-of-memory) -- stop
# rather than writing an error record for the entire test split.
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
    parser.add_argument(
        "--device-map",
        default="auto",
        help=(
            "auto (default), cpu, or a CUDA device such as cuda:0. A fixed CUDA "
            "device disables offload and requires enough VRAM. Adapter auto "
            "uses GPU 0 plus CPU, with a GPU placement budget of at most 6 GiB"
        ),
    )
    parser.add_argument(
        "--processor",
        default=None,
        help=(
            "processor/tokenizer directory or Hub ID; by default use saved "
            "processor files in --finetuned when present, otherwise the base"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="stop on the first inference error with a full traceback, without recording that failure",
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


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def load_adapter_for_inference(adapter_dir: Path, requested_device_map):
    """Assemble base plus adapter on CPU, then dispatch the complete model."""
    import torch
    from accelerate import dispatch_model, infer_auto_device_map
    from peft.tuners.tuners_utils import BaseTunerLayer
    from transformers import AutoModelForMultimodalLM

    max_memory = None
    if requested_device_map == "auto" and torch.cuda.is_available():
        import psutil

        input_device = torch.device("cuda:0")
        # Capture host availability before the full CPU model consumes RAM.
        free_gpu, _ = torch.cuda.mem_get_info(0)
        max_memory = {
            0: min(6 * 1024**3, int(free_gpu * 0.8)),
            "cpu": int(psutil.virtual_memory().available * 0.8),
        }
    elif requested_device_map == "auto":
        input_device = torch.device("cpu")
    else:
        input_device = torch.device(requested_device_map[""])

    print(f"Assembling base model and adapter on CPU: {adapter_dir}")
    # Transformers reads adapter_config.json and loads its recorded base.
    # No GPU offload hooks are installed before the LoRA modules exist.
    model = AutoModelForMultimodalLM.from_pretrained(
        str(adapter_dir),
        dtype="auto",
        device_map={"": "cpu"},
    )

    non_cpu = [
        (name, str(tensor.device))
        for name, tensor in (
            list(model.named_parameters()) + list(model.named_buffers())
        )
        if tensor.device.type != "cpu"
    ]
    if non_cpu:
        raise RuntimeError(
            f"CPU loading left unexpected devices: {non_cpu[:20]}"
        )

    adapter_layers = [
        module for module in model.modules() if isinstance(module, BaseTunerLayer)
    ]
    if not adapter_layers:
        raise RuntimeError("No adapter layers were loaded.")

    if max_memory is not None:
        no_split = set(getattr(model, "_no_split_modules", None) or [])
        no_split.update(type(module).__name__ for module in adapter_layers)
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=sorted(no_split),
            offload_buffers=False,
        )
        if "disk" in device_map.values():
            raise RuntimeError(
                "The selected GPU/CPU memory budgets require disk offload. "
                "This adapter loading route needs enough system RAM for the "
                "full model and does not configure disk offload."
            )
    else:
        # Preserve explicit CPU / whole-GPU placement and CPU-only auto runs.
        device_map = {"": str(input_device)}

    if input_device.type != "cpu":
        model = dispatch_model(
            model,
            device_map=device_map,
            main_device=input_device,
            skip_keys=getattr(model, "_skip_keys_device_placement", None),
            offload_buffers=False,
            force_hooks=True,
        )
    model.eval()
    print(f"Adapter layers: {len(adapter_layers)}")
    if max_memory is not None:
        print(f"Placement memory budgets (bytes): {max_memory}")
    # Keep this execution device even when every weight is stored on CPU.
    return model, input_device, max_memory


def load_model_and_processor(args: argparse.Namespace):
    """Select the checkpoint type before loading exactly one base model."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if args.device_map == "auto":
        device_map = "auto"
    else:
        target = torch.device(args.device_map)
        if target.type not in {"cpu", "cuda"}:
            raise SystemExit("--device-map must be auto, cpu, or a CUDA device such as cuda:0")
        if target.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("A CUDA device was requested, but CUDA is unavailable.")
        device_map = {"": str(target)}

    finetuned = args.finetuned.resolve() if args.finetuned is not None else None
    is_adapter = finetuned is not None and (finetuned / "adapter_config.json").is_file()
    if finetuned is None:
        checkpoint_kind = "base"
        model_source = MODEL_ID
    elif is_adapter:
        checkpoint_kind = "adapter"
        model_source = str(finetuned)
    elif (finetuned / "config.json").is_file():
        checkpoint_kind = "full_model"
        model_source = str(finetuned)
    else:
        raise SystemExit(
            f"{finetuned} is neither an adapter (no adapter_config.json) "
            "nor a model directory (no config.json)."
        )

    processor_files = (
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "chat_template.json",
    )
    if args.processor is not None:
        processor_source = args.processor
    elif finetuned is not None and any((finetuned / name).is_file() for name in processor_files):
        processor_source = str(finetuned)
    else:
        processor_source = MODEL_ID
        if finetuned is not None:
            print(
                "No saved processor/tokenizer files found in the checkpoint; "
                f"using {MODEL_ID}. Use --processor if training changed them."
            )

    print(f"Loading processor from {processor_source}")
    # Do not silently replace a saved but incomplete processor with the base:
    # changed tokens, templates, or image preprocessing must be preserved.
    processor = AutoProcessor.from_pretrained(processor_source)
    processor.tokenizer.padding_side = "left"

    adapter_base_reference = None
    if is_adapter:
        from peft import PeftConfig

        peft_config = PeftConfig.from_pretrained(str(finetuned))
        adapter_base_reference = peft_config.base_model_name_or_path
        print(f"Adapter's recorded training base: {adapter_base_reference}")

    print(f"Loading {checkpoint_kind}: {model_source}, device_map={device_map}")
    input_device = None
    max_memory = None
    if is_adapter:
        model, input_device, max_memory = load_adapter_for_inference(
            finetuned, device_map
        )
    else:
        model = AutoModelForMultimodalLM.from_pretrained(
            model_source,
            dtype="auto",
            device_map=device_map,
        )
    model.eval()

    hf_device_map = getattr(model, "hf_device_map", None) or {}
    meta_parameters = [name for name, tensor in model.named_parameters() if tensor.is_meta]
    meta_buffers = [name for name, tensor in model.named_buffers() if tensor.is_meta]
    print(f"hf_device_map: {hf_device_map}")
    for label, names in (("parameters", meta_parameters), ("buffers", meta_buffers)):
        print(f"Meta {label}: {len(names)}")
        for name in names[:30]:
            print(f"  {name}")
        if len(names) > 30:
            print(f"  ... and {len(names) - 30} more")
    if meta_parameters or meta_buffers:
        print(
            "Meta tensors can be normal with Accelerate CPU/disk offloading; "
            "they must be materialized before the operations that use them. "
            "Use --debug to locate any failing operation."
        )

    # Adapter dispatch supplies its execution device explicitly. Infer it
    # only for the ordinary base/full-model loading branches.
    if input_device is None:
        for mapped_device in hf_device_map.values():
            if isinstance(mapped_device, int):
                input_device = torch.device("cuda", mapped_device)
                break
            if str(mapped_device) not in {"cpu", "disk", "meta"}:
                input_device = torch.device(mapped_device)
                break
    if input_device is None:
        reported_device = getattr(model, "device", torch.device("cpu"))
        if reported_device.type != "meta":
            input_device = reported_device
        elif hf_device_map and all(str(d) in {"cpu", "disk"} for d in hf_device_map.values()):
            input_device = torch.device("cpu")
        else:
            raise RuntimeError(
                "The model reports a meta input device without a usable offload map. "
                "Check the checkpoint and the model-loading warnings."
            )
    print(f"Input device: {input_device}")

    load_info = {
        "checkpoint_kind": checkpoint_kind,
        "model_source": model_source,
        "adapter_base_reference": adapter_base_reference,
        "processor_source": processor_source,
        "device_map_requested": args.device_map,
        "loading_strategy": "cpu_then_dispatch" if is_adapter else "from_pretrained",
        "placement_memory_budget_bytes": (
            {str(key): value for key, value in max_memory.items()}
            if max_memory is not None else None
        ),
        "hf_device_map": {key: str(value) for key, value in hf_device_map.items()},
        "input_device": str(input_device),
        "meta_parameter_count": len(meta_parameters),
        "meta_buffer_count": len(meta_buffers),
    }
    return model, processor, input_device, load_info


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()

    # 1. Test samples: only these may be scored, per the split file.
    test_paths = load_test_split(data_dir)
    if args.limit is not None:
        test_paths = test_paths[: args.limit]
    print(f"Test samples selected: {len(test_paths):,}")

    # 2. Ground truth for those samples (streams the annotations once).
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
    print(
        f"Versions: torch={torch.__version__}, transformers={transformers.__version__}, "
        f"peft={package_version('peft')}, accelerate={package_version('accelerate')}"
    )
    model, processor, input_device, load_info = load_model_and_processor(args)

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
            "peft_version": package_version("peft"),
            "accelerate_version": package_version("accelerate"),
            "device": (
                torch.cuda.get_device_name(input_device)
                if input_device.type == "cuda"
                else str(input_device)
            ),
            **load_info,
        },
    )

    # Run one conversation per image, padded together for generation.
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
            padding=True,
            add_generation_prompt=True,
        ).to(input_device)
        prompt_len = inputs["input_ids"].shape[-1]

        start = time.perf_counter()
        with torch.inference_mode():
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
                if args.debug:
                    # Re-raise preserves the full traceback and avoids writing
                    # a failed prediction for a systematic loading problem.
                    raise
                # Retry a failed batch sample by sample to isolate the culprit.
                print(
                    f"Batch of {len(chunk)} failed ({type(exc).__name__}: {exc}); "
                    "retrying its samples individually ..."
                )
                torch.cuda.empty_cache()  # release unused cached GPU allocations
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
                            "is systematically wrong (model loading/device placement, "
                            "prompt format, or out-of-memory). "
                            "Stopping. Re-run with --debug and a fresh --run-name for a traceback."
                        )
            # Preserve the original benchmark's between-batch cache cleanup.
            torch.cuda.empty_cache()

    print(f"Wrote {written:,} prediction record(s) to {pred_path}")
    print("Next step: score the predictions file with the evaluation script (CER/WER).")


if __name__ == "__main__":
    main()
