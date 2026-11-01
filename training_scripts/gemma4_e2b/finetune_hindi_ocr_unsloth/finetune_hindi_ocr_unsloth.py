"""Fine-tune Gemma 4 E2B for Hindi Devanagari OCR with LoRA, using Unsloth.

Same task as finetune_hindi_ocr_hf.py -- Hindi word-crop images from the
train split of splits.csv, transcription-only loss -- but the training stack
is Unsloth's (FastVisionModel, UnslothVisionDataCollator), whose memory
optimizations are what make fine-tuning feasible on laptop-class GPUs.
Model loading and inference follow Unsloth's own standards, not the
transformers ones used by the pure-transformers variant.

The default prompt differs from the pure-transformers variant: "Do OCR on
this image without additional text. Just the text in the image and nothing
else." The saved adapter loads through the benchmark's --finetuned flag
like any other.

Usage:
    python3 training_scripts/gemma4_e2b/finetune_hindi_ocr_unsloth/finetune_hindi_ocr_unsloth.py
    python3 training_scripts/gemma4_e2b/finetune_hindi_ocr_unsloth/finetune_hindi_ocr_unsloth.py --train-samples 64 --val-samples 8 --epochs 1   # smoke test
    python3 training_scripts/gemma4_e2b/finetune_hindi_ocr_unsloth/finetune_hindi_ocr_unsloth.py --use-4bit --batch-size 2 --grad-accum 16  # small GPU
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import os
from pathlib import Path

# Make the project root importable (bench_common lives there).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Force Unsloth to return actual logits.
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from bench_common import DEFAULT_DATA_DIR, collect_ground_truth


def parse_args() -> argparse.Namespace:
    """Parse and validate the command-line arguments.

    Every default lives in its add_argument call and is rendered in --help
    via argparse's %(default)s, so the flags themselves are the single
    source of truth for the configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Gemma 4 E2B for Hindi Devanagari OCR with LoRA "
            "(Unsloth FastVisionModel + SFTTrainer)."
        )
    )
    parser.add_argument(
        "--model-id",
        default="unsloth/gemma-4-E2B-it",
        help="base model to fine-tune (default: %(default)s)",
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
        default="Do OCR on this image without additional text. Just the text in the image and nothing else.",
        help='transcription prompt (default: "%(default)s")',
    )
    parser.add_argument("--seed", type=int, default=3407, help="random seed (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=4, help="training epochs (default: %(default)s)")
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="images per forward pass (default: %(default)s)",
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=4,
        help="images per validation pass (default: %(default)s)",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=2,
        help="gradient accumulation steps (default: %(default)s)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4,
        help="optimizer learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--use-4bit", action="store_true", help="load the model in 4-bit precision (default: %(default)s)"
    )
    parser.add_argument("--warmup-steps", type=int, default=5, help="LR warmup steps (default: %(default)s)")
    parser.add_argument("--logging-steps", type=int, default=50, help="log training loss every N steps (default: %(default)s)")
    parser.add_argument("--eval-steps", type=int, default=50, help="validate every N steps (default: %(default)s)")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (default: %(default)s)")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha (default: %(default)s)")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout (default: %(default)s)")
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="SFTConfig max tokens per sample (default: %(default)s)",
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


def to_conversation(args, image_path: str, ground_truth: str) -> dict:
    """Build one sample in Unsloth's conversation format: instruction text and
    image in the user turn, transcription in the assistant turn."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image", "image": str(args.data_dir / image_path)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ground_truth}],
            },
        ]
    }


def print_config(args) -> None:
    """Echo the whole run configuration before anything heavy starts, so the
    user can see every decision the flags resolved to."""
    line = "-" * 64
    print(line)
    print("Fine-tuning Gemma 4 E2B for Hindi Devanagari OCR (Unsloth)")
    print(line)
    print(f"Model          : {args.model_id}")
    print(f"Model load     : dtype auto (Unsloth), load_in_4bit={args.use_4bit}, max_seq_length=512")
    print(f"Data dir       : {args.data_dir}")
    print(f"Hindi prefixes : {', '.join(args.hindi_prefixes)}")
    print(f"Samples        : {args.train_samples} train / {args.val_samples} val (seed {args.seed})")
    print(f"Prompt         : \"{args.prompt}\"")
    print(f"Schedule       : {args.epochs} epochs, lr {args.learning_rate}, linear decay, "
          f"{args.warmup_steps} warmup steps")
    print(f"Batching       : {args.batch_size} per step x {args.grad_accum} accumulation "
          f"= effective {args.batch_size * args.grad_accum}")
    print(f"LoRA           : r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, "
          f"vision + language layers")
    print(f"Max length     : {args.max_length} tokens")
    print(f"Optimizer      : AdamW 8-bit, weight decay 0.001")
    print(f"Logging        : every {args.logging_steps} steps; validation every {args.eval_steps} steps")
    print(f"Adapter output : {args.output_dir}")
    print(line)
    print()


def demo(args, model, tokenizer, image_paths: list[str], truth: dict[str, str]) -> None:
    """Stream a few validation generations, Unsloth-style: ground truth vs.
    the fine-tuned model, sampled exactly as the reference article does."""
    FastVisionModel.for_inference(model)
    text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    for path in image_paths:
        image = str(args.data_dir / path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        print(f"{path}\n  gt  : {truth[path]}\n  pred: ", end="", flush=True)
        model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=64,  # word crops transcribe in a few tokens
            use_cache=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
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
    train_dataset = [to_conversation(args, p, truth[p]) for p in train_paths]
    val_dataset = [to_conversation(args, p, truth[p]) for p in val_paths]

    # --- Model preparation (Unsloth standards) ---
    # 4-bit off is Unsloth's current recommendation, but the bf16 load needs
    # ~10 GB; pass --use-4bit if the GPU cannot host that.
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_id,
        dtype=None,
        max_seq_length=512,
        load_in_4bit=args.use_4bit,
        full_finetuning=False,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # --- Training ---
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            eval_strategy="steps",
            do_eval=True,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=str(args.output_dir),
            report_to="none",

            # You MUST put the below items for vision finetuning:
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_length,
        ),
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Trainable LoRA parameters: {trainable:,}")

    trainer.train()

    # --- Save the adapter ---
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved LoRA adapter to {args.output_dir}")

    # --- A short look at the fine-tuned model ---
    demo(args, model, tokenizer, val_paths[:3], truth)


if __name__ == "__main__":
    main()
