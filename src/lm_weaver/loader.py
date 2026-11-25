# Imports the network from model.py, so any checkpoint written by trainer.py
# loads here regardless of the architecture it was trained with.
#
#   python3 loader.py                     # uses ./model.pth
#   python3 loader.py run_1.pth           # a specific checkpoint
#   python3 loader.py --temp 0.6 --top-k 20
#   python3 loader.py --prompt "User: Hi\nBot:"   # one-shot, no chat
import os
from argparse import ArgumentParser

import torch

from .model import (
    AppConfig,
    CharTokenizer,
    ModelConfig,
    ScratchLanguageModel,
    load_checkpoint,
)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# A reply ends at the blank line before the next turn. Stopping here (rather
# than waiting for "User:") is what prevents the model narrating the next turn:
# the blank line is reliably produced, the following word is not.
TURN_BREAK = "\n\n"
# Kept for prompts that still use the old turn format.
STOP_ON = "\n\nUser:"


def parse_args():
    parser = ArgumentParser(description="Chat with a trained checkpoint.")
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="path to a .pth file (default: model.pth)")
    parser.add_argument("--prompt", default=None,
                        help="generate one reply for this prompt, then exit")
    parser.add_argument("--temp", type=float, dest="temperature",
                        help="0 = greedy (default: value saved in checkpoint)")
    parser.add_argument("--top-k", type=int, help="0 = disabled")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                        default="auto")
    return parser.parse_args()


def find_checkpoint(explicit):
    """Resolve which file to load, with a helpful error if none exists."""
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"ERROR: '{explicit}' not found.")
        return explicit
    for candidate in ("model.pth", "my_lm.pth"):
        if os.path.exists(candidate):
            return candidate
    found = sorted(f for f in os.listdir(".") if f.endswith(".pth"))
    if len(found) == 1:
        return found[0]
    hint = f" Available: {', '.join(found)}" if found else ""
    raise SystemExit(
        "ERROR: no checkpoint found.\n"
        "Train one first (python3 trainer.py) or pass a path explicitly." + hint
    )


def load_model(path, requested_device):
    # Rebuild the exact network from the checkpoint's saved config
    run_device = device if requested_device == "auto" else requested_device

    checkpoint = load_checkpoint(path, map_location="cpu")

    config = ModelConfig.from_dict(checkpoint["config"])
    tokenizer = CharTokenizer(checkpoint["chars_mapping"])
    app_config = AppConfig.from_dict(checkpoint.get("app_config", {}))

    model = ScratchLanguageModel(config)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        raise SystemExit(f"ERROR: weights in '{path}' do not match its config.\n{error}")
    model = model.to(run_device).eval()
    return model, tokenizer, app_config, run_device


def reply(model, tokenizer, run_device, history, temperature, top_k, max_new):
    # Generate one bot turn for the given chat history
    prompt_ids = tokenizer.encode(history)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=run_device)
    out = model.generate(
        idx,
        max_new_tokens=max_new,
        temperature=temperature,
        top_k=top_k,
        stop_on=STOP_ON,
        tokenizer=tokenizer,
    )
    generated = tokenizer.decode(out[0, len(prompt_ids) :].tolist())
    # Cut at the turn boundary. STOP_ON is the blank line that separates turns;
    # a stray "\n\nUser:" is also cut so either shape ends the reply.
    for marker in (TURN_BREAK, STOP_ON, "User:"):
        stop = generated.find(marker)
        if stop != -1:
            generated = generated[:stop]
    return generated.strip("\n")


def chat(model, tokenizer, run_device, temperature, top_k, max_new):
    size = sum(p.numel() for p in model.parameters())
    print(f"Chat Engine Active on Hardware: [{run_device.upper()}]")
    print(
        f"Model ready: {size:,} parameters, vocab={tokenizer.vocab_size}, "
        f"block_size={model.config.block_size}, "
        f"temp={temperature}, top_k={top_k or 'off'}"
    )
    print("Type 'exit' or press Ctrl+D to end the session.\n")

    history = ""
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended")
            break
        if user_input.lower() in ("exit", "quit"):
            print("Session ended")
            break
        if not user_input.strip():
            continue

        unknown = tokenizer.unknown_chars(user_input)
        if unknown:
            print(f"(ignoring characters that the model was never trained on: {''.join(unknown)!r})")

        history += f"User: {user_input}\nBot:"
        print("Bot: ", end="", flush=True)
        response = reply(
            model, tokenizer, run_device, history, temperature, top_k, max_new
        )
        print(response or "(no reply)")
        # Only the tail fits in the model's context window anyway.
        history += response + "\n\n"
        history = history[-(2 * model.config.block_size) :]


def main():
    args = parse_args()
    path = find_checkpoint(args.checkpoint)
    model, tokenizer, app_config, run_device = load_model(path, args.device)

    # CLI flags win; otherwise use what the checkpoint was trained/saved with.
    temperature = (
        args.temperature if args.temperature is not None else app_config.temperature
    )
    top_k = args.top_k if args.top_k is not None else app_config.top_k
    max_new = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else app_config.max_new_tokens
    )

    if args.prompt is not None:
        out = reply(
            model, tokenizer, run_device, args.prompt,
            temperature, top_k, max_new,
        )
        print(out)
        return

    chat(model, tokenizer, run_device, temperature, top_k, max_new)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSession ended")
        raise SystemExit(130)
    except ValueError as error:
        # Raised by ModelConfig validation when a checkpoint is malformed.
        raise SystemExit(f"CONFIG ERROR: {error}")
