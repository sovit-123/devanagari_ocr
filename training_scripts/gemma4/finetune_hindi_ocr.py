"""Fine-tune Gemma 4 E2B for Hindi Devanagari OCR with LoRA, in pure transformers.

Trains on Hindi word-crop images from the train split of splits.csv, validates
on the val split, saves the LoRA adapter, and ends with a short generation
demo on validation samples. Training and benchmarking share one runtime, so
the saved adapter plugs straight into the benchmark harness.

--use-4bit (the default) trains on a 4-bit quantized base (QLoRA), which fits
small GPUs; --no-use-4bit trains in full bf16 on a bigger one. The saved
adapter is quantization-agnostic: it loads onto either base at any time, and
official benchmarks always use the unquantized base to stay comparable with
the existing pilot run.

Usage:
    python3 training_scripts/gemma4/finetune_hindi_ocr.py
    python3 training_scripts/gemma4/finetune_hindi_ocr.py --train-samples 64 --val-samples 8 --epochs 1   # smoke test
    python3 training_scripts/gemma4/finetune_hindi_ocr.py --no-use-4bit --batch-size 16 --grad-accum 2    # big GPU
"""

from __future__ import annotations

import os

# Same allocator fix as the benchmark script: the repeated multi-GB vision
# allocations fragment the CUDA cache on an 8 GB GPU, and expandable_segments
# avoids it. Must be set before torch initializes CUDA (torch is imported
# further below).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import random
import sys
from pathlib import Path

# Make the project root importable (bench_common lives there).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from bench_common import DEFAULT_DATA_DIR, collect_ground_truth  # noqa: E402

def parse_args() -> argparse.Namespace:
    """Parse and validate the command-line arguments.

    Every default lives in its add_argument call and is rendered in --help
    via argparse's %(default)s, so the flags themselves are the single
    source of truth for the configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Gemma 4 E2B for Hindi Devanagari OCR with LoRA "
            "(TRL SFTTrainer, completion-only loss), pure transformers."
        )
    )
    parser.add_argument(
        "--model-id", default="google/gemma-4-E2B-it", help="base model to fine-tune (default: %(default)s)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="dataset directory holding splits.csv (default: input/devanagari_ocr_dataset)",
    )
    parser.add_argument(
        "--hindi-prefixes",
        nargs="+",
        default=["hindi_local", "iiit_hindi", "hindidataocr"],
        help="image filename prefixes that count as Hindi (default: %(default)s)",
    )
    parser.add_argument(
        "--train-samples", type=int, default=2000,
        help="training images drawn from the train split (default: %(default)s)",
    )
    parser.add_argument(
        "--val-samples", type=int, default=200,
        help="validation images drawn from the val split (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        default="OCR this image without additional text",
        help='transcription prompt (default: "%(default)s")',
    )
    parser.add_argument("--seed", type=int, default=3407, help="random seed (default: %(default)s)")
    parser.add_argument(
        "--use-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train on a 4-bit NF4 base (QLoRA, fits 8 GB GPUs); "
        "--no-use-4bit trains in bf16 (default: %(default)s)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="training epochs (default: %(default)s)")
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="images per forward pass; the vision tower is huge per image (default: %(default)s)",
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=1,
        help="images per validation pass (default: %(default)s)",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=32,
        help="gradient accumulation steps; 32 x 1 = effective batch 32 (default: %(default)s)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4,
        help="optimizer learning rate (default: %(default)s)",
    )
    parser.add_argument("--warmup-steps", type=int, default=10, help="LR warmup steps (default: %(default)s)")
    parser.add_argument("--logging-steps", type=int, default=20, help="log training loss every N steps (default: %(default)s)")
    parser.add_argument("--eval-steps", type=int, default=50, help="validate every N steps (default: %(default)s)")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (default: %(default)s)")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha (default: %(default)s)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: %(default)s)")
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="max tokens per sample; image tokens alone are ~256 (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "hindi_ocr_lora",
        help="where the LoRA adapter is saved (default: %(default)s)",
    )
    args = parser.parse_args()

    for name in (
        "train_samples",
        "val_samples",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "grad_accum",
        "warmup_steps",
        "logging_steps",
        "eval_steps",
        "lora_r",
        "lora_alpha",
        "max_length",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 <= args.lora_dropout < 1:
        parser.error("--lora-dropout must be between 0 and 1")
    return args


def select_samples(args, split: str, limit: int) -> list[str]:
    """Return `limit` Hindi image paths from the given split, seed-shuffled."""
    paths = []
    with open(args.data_dir / "splits.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split and Path(row["image_path"]).name.startswith(tuple(args.hindi_prefixes)):
                paths.append(row["image_path"])
    random.Random(args.seed).shuffle(paths)
    return paths[:limit]


def to_prompt_completion(args, image_path: str, ground_truth: str) -> dict:
    """Build one sample in TRL's prompt/completion format: image + prompt in,
    transcription out. completion_only_loss then trains the model only on the
    transcription."""
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(args.data_dir / image_path)},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ground_truth}],
            }
        ],
    }


def build_dataset(args, image_paths: list[str], truth: dict[str, str]) -> Dataset:
    """Turn image paths into an HF dataset of prompt/completion samples."""
    return Dataset.from_list([to_prompt_completion(args, p, truth[p]) for p in image_paths])


def demo(args, model, processor, image_paths: list[str], truth: dict[str, str]) -> None:
    """Generate on a few validation samples: ground truth vs. fine-tuned model."""
    model.eval()
    for path in image_paths:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(args.data_dir / path)},
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
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        prediction = processor.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
        print(f"{path}\n  gt  : {truth[path]}\n  pred: {prediction}\n")


def print_config(args) -> None:
    """Echo the whole run configuration before anything heavy starts, so the
    user can see every decision the flags resolved to."""
    quantization = "4-bit NF4 (QLoRA)" if args.use_4bit else "bf16 (LoRA)"
    line = "-" * 64
    print(line)
    print("Fine-tuning Gemma 4 E2B for Hindi Devanagari OCR")
    print(line)
    print(f"Model          : {args.model_id}")
    print(f"Precision      : {quantization}")
    print(f"Data dir       : {args.data_dir}")
    print(f"Hindi prefixes : {', '.join(args.hindi_prefixes)}")
    print(f"Samples        : {args.train_samples} train / {args.val_samples} val (seed {args.seed})")
    print(f"Prompt         : \"{args.prompt}\"")
    print(f"Schedule       : {args.epochs} epochs, lr {args.learning_rate}, linear decay, "
          f"{args.warmup_steps} warmup steps")
    print(f"Batching       : {args.batch_size} per step x {args.grad_accum} accumulation "
          f"= effective {args.batch_size * args.grad_accum}")
    print(f"LoRA           : r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, all-linear")
    print(f"Max length     : {args.max_length} tokens")
    print(f"Optimizer      : paged AdamW 8-bit, weight decay 0.01")
    print(f"Logging        : every {args.logging_steps} steps; validation every {args.eval_steps} steps")
    print(f"Adapter output : {args.output_dir}")
    print(line)
    print()


def main() -> None:
    args = parse_args()
    print_config(args)

    # --- Data preparation ---
    train_paths = select_samples(args, "train", args.train_samples)
    val_paths = select_samples(args, "val", args.val_samples)
    print(f"Selected {len(train_paths)} training and {len(val_paths)} validation samples.")

    print("Streaming annotations for ground truth ...")
    truth = collect_ground_truth(args.data_dir, train_paths + val_paths)
    train_dataset = build_dataset(args, train_paths, truth)
    val_dataset = build_dataset(args, val_paths, truth)

    # --- Model preparation ---
    processor = AutoProcessor.from_pretrained(args.model_id)
    model_kwargs = {"device_map": "auto"}
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForMultimodalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        args=training_args,
        processing_class=processor,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Trainable LoRA parameters: {trainable:,}")

    # --- Training ---
    trainer.train()
    trainer.model.config.use_cache = True

    # --- Save the adapter ---
    trainer.model.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print(f"Saved LoRA adapter to {args.output_dir}")

    # --- A short look at the fine-tuned model ---
    demo(args, trainer.model, processor, val_paths[:3], truth)


if __name__ == "__main__":
    main()
