"""Fine-tune Gemma 4 E2B for Hindi Devanagari OCR with LoRA, in pure transformers.

Trains on Hindi word-crop images from the train split of splits.csv, validates
on the val split, saves the LoRA adapter, and ends with a short generation
demo on validation samples. Training and benchmarking share one runtime, so
the saved adapter plugs straight into the benchmark harness.

USE_4BIT switches between a 4-bit quantized base (QLoRA, for small GPUs) and
full-precision bf16. The saved adapter is quantization-agnostic: it loads
onto either base at any time, and official benchmarks always use the
unquantized base to stay comparable with the existing pilot run.

Usage:
    python3 training_scripts/gemma4/finetune_hindi_ocr.py
"""

from __future__ import annotations

import os

# Same allocator fix as the benchmark script: the repeated multi-GB vision
# allocations fragment the CUDA cache on an 8 GB GPU, and expandable_segments
# avoids it. Must be set before torch initializes CUDA (torch is imported
# further below).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

MODEL_ID = "google/gemma-4-E2B-it"
DATA_DIR = DEFAULT_DATA_DIR

# Hindi-only sample selection: image filename prefixes that belong to the
# Hindi subset of the dataset.
HINDI_PREFIXES = ("hindi_local", "iiit_hindi", "hindidataocr")
TRAIN_SAMPLES = 2000         # training images drawn from the train split
VAL_SAMPLES = 200            # validation images drawn from the val split
TRAIN_PROMPT = "OCR this image without additional text"
SEED = 3407

USE_4BIT = True              # QLoRA: 4-bit NF4 base, required to fit an 8 GB GPU

EPOCHS = 3
BATCH_SIZE = 1               # one image per step; the vision tower is huge per image
EVAL_BATCH_SIZE = 1
GRAD_ACCUM = 32             # 32 x 1 = effective batch 32, same as the 16 x 2 L40S setup
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10
LOGGING_STEPS = 20
EVAL_STEPS = 50
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
MAX_LENGTH = 512

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "hindi_ocr_lora"


def select_samples(split: str, limit: int) -> list[str]:
    """Return `limit` Hindi image paths from the given split, seed-shuffled."""
    paths = []
    with open(DATA_DIR / "splits.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split and Path(row["image_path"]).name.startswith(HINDI_PREFIXES):
                paths.append(row["image_path"])
    random.Random(SEED).shuffle(paths)
    return paths[:limit]


def to_prompt_completion(image_path: str, ground_truth: str) -> dict:
    """Build one sample in TRL's prompt/completion format: image + prompt in,
    transcription out. completion_only_loss then trains the model only on the
    transcription."""
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(DATA_DIR / image_path)},
                    {"type": "text", "text": TRAIN_PROMPT},
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


def build_dataset(image_paths: list[str], truth: dict[str, str]) -> Dataset:
    """Turn image paths into an HF dataset of prompt/completion samples."""
    return Dataset.from_list([to_prompt_completion(p, truth[p]) for p in image_paths])


def demo(model, processor, image_paths: list[str], truth: dict[str, str]) -> None:
    """Generate on a few validation samples: ground truth vs. fine-tuned model."""
    model.eval()
    for path in image_paths:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(DATA_DIR / path)},
                    {"type": "text", "text": TRAIN_PROMPT},
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


def main() -> None:
    # --- Data preparation ---
    train_paths = select_samples("train", TRAIN_SAMPLES)
    val_paths = select_samples("val", VAL_SAMPLES)
    print(f"Selected {len(train_paths)} training and {len(val_paths)} validation samples.")

    print("Streaming annotations for ground truth ...")
    truth = collect_ground_truth(DATA_DIR, train_paths + val_paths)
    train_dataset = build_dataset(train_paths, truth)
    val_dataset = build_dataset(val_paths, truth)

    # --- Model preparation ---
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model_kwargs = {"device_map": "auto"}
    if USE_4BIT:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=EPOCHS,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_length=MAX_LENGTH,
        packing=False,
        completion_only_loss=True,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        seed=SEED,
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
    trainer.model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")

    # --- A short look at the fine-tuned model ---
    demo(trainer.model, processor, val_paths[:3], truth)


if __name__ == "__main__":
    main()
