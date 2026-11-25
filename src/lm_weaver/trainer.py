# The network itself lives in model.py, which loader.py also imports, so a
# model trained here always loads there. Everything tunable is a flag:
# previously the model was created withing trainer.py
#   python3 trainer.py                     # defaults, saves my_lm.pth
#   python3 trainer.py --preset big        # a named size preset
#   python3 trainer.py --n-layer 8 --steps 20000
#   python3 trainer.py --data my.txt --out a.pth --temp 0.7 --top-k 40
#   python3 trainer.py --list-presets
import os
import random
import time
from argparse import ArgumentParser

import torch

from .model import (
    AppConfig,
    CharTokenizer,
    ModelConfig,
    ScratchLanguageModel,
    save_checkpoint,
)

if torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon Metal
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"  # Fallback to cpu if accelerator missing
print(f"LM Trainer Active on Engine: [{device.upper()}]")


# ----------------------------------------------------------------------------
# Model size presets. Each one is just a ModelConfig with different numbers.
# Measured with this dataset's 44-character vocabulary (from --list-presets):
#   tiny    ~107k    fast smoke test
#   small   ~349k
#   base    ~811k    default; good balance for this dataset
#   big     ~4.8M    often overfits a small corpus -- watch VAL LOSS
# Run `python3 trainer.py --list-presets` to print the live numbers.
# ----------------------------------------------------------------------------
PRESETS = {
    "tiny": ModelConfig(n_embd=64, n_head=4, n_layer=2, block_size=32),
    "small": ModelConfig(n_embd=96, n_head=4, n_layer=3, block_size=64),
    "base": ModelConfig(n_embd=128, n_head=4, n_layer=4, block_size=64),
    "big": ModelConfig(n_embd=256, n_head=8, n_layer=6, block_size=128),
}


def parse_args():
    parser = ArgumentParser(description="Train a small character-level LM.")
    parser.add_argument("--data", default=None,
                        help="training text file (default: train_data.txt)")
    hf = parser.add_argument_group("Hugging Face datasets")
    hf.add_argument("--hf-dataset", default=None,
                    help="dataset id, e.g. HuggingFaceTB/smoltalk "
                         "(or a local .jsonl file)")
    hf.add_argument("--hf-config", default=None,
                    help="dataset config/subset name, e.g. everyday-conversations")
    hf.add_argument("--hf-split", default="train", help="split to use")
    hf.add_argument("--hf-max-examples", type=int, default=None,
                    help="stop after this many conversations")
    parser.add_argument("--out", default="my_lm.pth", help="checkpoint to write")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="base")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--list-presets", action="store_true")

    arch = parser.add_argument_group("model size (override the preset)")
    arch.add_argument("--block-size", type=int)
    arch.add_argument("--n-embd", type=int)
    arch.add_argument("--n-head", type=int)
    arch.add_argument("--n-layer", type=int)
    arch.add_argument("--dropout", type=float)

    train = parser.add_argument_group("training")
    train.add_argument("--lr", type=float)
    train.add_argument("--min-lr", type=float)
    train.add_argument("--warmup", type=int, help="warmup steps")
    train.add_argument("--batch-size", type=int)
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--grad-clip", type=float)
    train.add_argument("--no-bias", action="store_true", help="disable bias terms")
    train.add_argument("--eval-every", type=int, default=250)
    train.add_argument("--eval-iters", type=int, default=20)
    train.add_argument("--shuffle-repeats", type=int, help="corpus duplication")
    train.add_argument("--val-fraction", type=float)

    gen = parser.add_argument_group("generation (saved in the checkpoint)")
    gen.add_argument("--temp", type=float, dest="temperature",
                     help="0 = greedy, 1 = raw sampling")
    gen.add_argument("--top-k", type=int, help="0 = disabled")
    gen.add_argument("--max-new-tokens", type=int)
    return parser.parse_args()


def build_configs(args):
    """Merge preset + explicit flags into (ModelConfig, AppConfig)."""
    arch_changes = {
        "block_size": args.block_size,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "dropout": args.dropout,
    }
    app_changes = {
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "shuffle_repeats": args.shuffle_repeats,
        "val_fraction": args.val_fraction,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    model_config = PRESETS[args.preset].replace(
        **{k: v for k, v in arch_changes.items() if v is not None}
    )
    if args.no_bias:
        model_config = model_config.replace(bias=False)
    app_config = AppConfig().replace(
        **{k: v for k, v in app_changes.items() if v is not None}
    )
    return model_config, app_config

def read_training_text(args):
    """Get the raw training text from a file, a HF dataset, or a local .jsonl."""
    if args.hf_dataset:
        if os.path.exists(args.hf_dataset):
            from .hf_data import load_jsonl_corpus

            return load_jsonl_corpus(
                args.hf_dataset, max_examples=args.hf_max_examples
            )
        from .hf_data import load_hf_corpus

        return load_hf_corpus(
            args.hf_dataset,
            config=args.hf_config,
            split=args.hf_split,
            max_examples=args.hf_max_examples,
        )

    path = args.data or "train_data.txt"
    if not os.path.exists(path):
        raise SystemExit(
            f"\nCRITICAL ERROR: '{path}' was not found in this folder!\n"
            "Create a text file with your data (see readme.md), or train "
            "straight from a Hugging Face dataset with --hf-dataset flag such as:\n"
            "  python3 trainer.py --hf-dataset HuggingFaceTB/smoltalk "
            "--hf-config everyday-conversations"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def describe_source(args):
    if args.hf_dataset:
        where = args.hf_dataset
        if args.hf_config:
            where += f" [{args.hf_config}]"
        return f"{where} split={args.hf_split!r}"
    return args.data or "train_data.txt"


def load_corpus(raw_text, block_size, repeats, seed, source="the dataset"):
    """Duplicate the corpus with reshuffled dialogue turns.

    The shuffles do not add knowledge; they expose the model to the same
    exchanges in different orders so it sees more context variations.
    """
    raw_text = raw_text.strip()
    if len(raw_text) <= block_size:
        raise SystemExit(
            f"\nCRITICAL ERROR: {source} is empty or too short.\n"
            f"It needs more than block_size={block_size} characters."
        )

    rng = random.Random(seed)
    base_turns = raw_text.split("\n\n")
    dialogues = []
    for _ in range(max(repeats, 1)):
        rng.shuffle(base_turns)
        dialogues.append("\n\n".join(base_turns))
    return "\n\n".join(dialogues)


def get_lr(step, app_config, total_steps):
    # Linear warmup then cosine decay -- the standard small-LM schedule.
    warmup = min(app_config.warmup_steps, total_steps)
    if warmup and step < warmup:
        return app_config.lr * (step + 1) / warmup
    if total_steps <= warmup:
        return app_config.lr
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))
    return app_config.min_lr + cosine.item() * (app_config.lr - app_config.min_lr)


def main():
    args = parse_args()

    if args.list_presets:
        from .model import count_parameters

        for name, preset in PRESETS.items():
            params = count_parameters(ScratchLanguageModel(preset))
            print(
                f"{name:6s} {params:>10,} params | n_embd={preset.n_embd} "
                f"n_head={preset.n_head} n_layer={preset.n_layer} "
                f"block_size={preset.block_size}"
            )
        return

    model_config, app_config = build_configs(args)
    torch.manual_seed(app_config.seed)

    print(f"STAGE 1: IMPORTING TRAINING DATA FROM {describe_source(args)}...")
    text = load_corpus(
        read_training_text(args),
        model_config.block_size,
        app_config.shuffle_repeats,
        app_config.seed,
        source=describe_source(args),
    )
    print(f"DATA CONFIGURED! TOTAL ANALYZED CHARS: {len(text):,}")

    tokenizer = CharTokenizer.from_text(text)
    model_config = model_config.replace(vocab_size=tokenizer.vocab_size)

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    # Hold out the tail of the corpus: a loss the model never trained on.
    n_split = int((1.0 - app_config.val_fraction) * len(data))
    train_data, val_data = data[:n_split], data[n_split:]
    if len(val_data) <= model_config.block_size:
        raise SystemExit(
            f"\nCRITICAL ERROR: not enough data to hold out "
            f"{app_config.val_fraction:.0%} for validation.\nAdd more training "
            "text or lower --val-fraction."
        )

    model = ScratchLanguageModel(model_config).to(device)
    print(
        f"MODEL: {model.num_parameters():,} parameters | "
        f"vocab={model_config.vocab_size} block_size={model_config.block_size} "
        f"n_layer={model_config.n_layer} n_embd={model_config.n_embd} "
        f"n_head={model_config.n_head}"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=app_config.lr, weight_decay=app_config.weight_decay
    )

    def get_batch(split):
        source = train_data if split == "train" else val_data
        max_start = max(len(source) - model_config.block_size, 1)
        ix = torch.randint(max_start, (app_config.batch_size,))
        x = torch.stack([source[i : i + model_config.block_size] for i in ix])
        y = torch.stack([source[i + 1 : i + model_config.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(eval_iters):
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # ---------------------------------------------------------------- train
    best_val, best_step = float("inf"), 0
    print("\nSTAGE 2: Trainig started...")
    start = time.time()
    for step in range(args.steps):
        # Learning-rate schedule: warmup, then decay toward min_lr.
        lr = get_lr(step, app_config, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if app_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), app_config.grad_clip)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            stats = estimate_loss(args.eval_iters)
            if stats["val"] <= best_val:
                best_val, best_step = stats["val"], step
            elapsed = time.time() - start
            print(
                f"STEP {step:5d} | LR {lr:.2e} | TRAIN {stats['train']:.4f} "
                f"| VAL {stats['val']:.4f} | {elapsed:5.1f}s"
            )

    # ---------------------------------------------------------------- save
    if best_step < args.steps - 1:
        print(
            f"\nWARNING: validation loss was lowest at step {best_step} "
            f"({best_val:.4f}) and rose afterwards -- the model is overfitting.\n"
            "Try more data, a smaller --preset, or fewer --steps."
        )

    chosen_name = args.out
    # Keep the interactive rename only when the default name was not overridden.
    if args.out == "my_lm.pth" and os.isatty(0):
        chosen_name = ask_for_name(default_name=args.out)
    if not os.path.splitext(chosen_name)[1]:
        chosen_name += ".pth"

    save_checkpoint(
        chosen_name,
        model,
        tokenizer,
        app_config,
        extra={
            "steps": args.steps,
            "final_train_loss": stats["train"],
            "final_val_loss": stats["val"],
            "best_val_loss": best_val,
            "best_step": best_step,
            "data_file": describe_source(args),
        },
    )
    print(
        f"MODEL TRAINING SUCCESS: MODEL BUNDLE SAVED AS '{chosen_name}' "
        f"({os.path.getsize(chosen_name) / 1e6:.1f} MB)"
    )
    print(f"Chat with it:  python3 loader.py {chosen_name}")


def ask_for_name(default_name="my_lm.pth", timeout=45.0):
    print("\nSAVING STAGE:")
    print(
        f"You have {timeout:.0f} seconds to rename your model, otherwise it will "
        f"be saved as '{default_name}'."
    )
    try:
        # inputimeout is optional; it only adds the countdown.
        from inputimeout import TimeoutOccurred, inputimeout

        try:
            answer = inputimeout(
                prompt="Enter model name: ", timeout=timeout
            ).strip()
        except TimeoutOccurred:
            print("\nTIME IS UP!!! Default value applied...")
            return default_name
    except ImportError:
        answer = input("Enter model name (blank = default): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNO INPUT RECEIVED! Default name applied...")
        return default_name
    return answer or default_name


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Nothing was saved.")
        raise SystemExit(130)
    except ValueError as error:
        # Raised by ModelConfig validation, e.g. a bad --n-head value.
        raise SystemExit(f"CONFIG ERROR: {error}")
