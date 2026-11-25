"""Turn a Hugging Face dataset into training text for trainer.py.

The training format is plain text: one exchange per turn, turns separated by a
blank line, e.g.

    User: Hi
    Bot: Hello! How can I help you today?

Hugging Face conversation datasets store the same information with a json format:
messages ({"role": "user"|"assistant", "content": "..."}). This module converts
that structure into the flat text above, the braces never reach the model,
they are only a container for the turns.

Usage:
    python3 trainer.py --hf-dataset HuggingFaceTB/smoltalk --hf-config everyday-conversations
    python3 trainer.py --hf-dataset HuggingFaceH4/ultrachat_200k --hf-split train_sft
    python3 trainer.py --hf-dataset ./my_local.jsonl      # local file also works
"""
import json

# Role names used by different dataset families, mapped onto the two speakers.
USER_ROLES = {"user", "human", "prompter", "question"}
BOT_ROLES = {"assistant", "gpt", "bot", "ai", "answer"}
# Roles whose content is dropped (metadata, not dialogue).
SKIP_ROLES = {"system", "tool", "function", "observation"}

# Column names that hold a message list, most common first.
MESSAGE_COLUMNS = ("messages", "conversations", "conversation")
# Fallback: single-turn datasets with one prompt and one response column.
PROMPT_COLUMNS = ("prompt", "question", "instruction", "input")
RESPONSE_COLUMNS = ("response", "answer", "output", "completion")


def _import_datasets():
    try:
        import datasets
    except ImportError:
        raise SystemExit(
            "ERROR: --hf-dataset needs the 'datasets' package.\n"
            "Install it with:  pip install datasets\n"
            "(it is listed in requirements.txt as optional)"
        )
    return datasets


def _find_column(example, candidates):
    for name in candidates:
        if name in example and example[name] is not None:
            return name
    return None


def _normalise_messages(raw):
    """Extract [(speaker, text), ...] from a dataset's message list.

    Handles both shapes seen in the wild:
      * {"role": "user", "content": "..."}   (smoltalk, ultrachat, OpenAI)
      * {"from": "human", "value": "..."}    (ShareGPT style)
    """
    turns = []
    for message in raw:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("from") or message.get("sender")
        content = message.get("content")
        if content is None:
            content = message.get("value") or message.get("text")
        if role is None or content is None:
            continue
        role = str(role).lower()
        text = str(content).strip()
        if not text:
            continue
        if role in USER_ROLES:
            turns.append(("User", text))
        elif role in BOT_ROLES:
            turns.append(("Bot", text))
        # SKIP_ROLES and anything unknown are dropped rather than mislabelled.
    return turns


def _format_turns(turns):
    """Render [(speaker, text)] as one blank-line-separated exchange."""
    lines = []
    for speaker, text in turns:
        # Collapse internal newlines: the blank line is the turn separator.
        flat = " ".join(text.split())
        lines.append(f"{speaker}: {flat}")
    return "\n".join(lines)


def example_to_text(example):
    """Convert one dataset row to training text, or None if it has no dialogue."""
    column = None
    for name in MESSAGE_COLUMNS:
        if name in example and isinstance(example[name], list):
            column = name
            break
    if column is not None:
        turns = _normalise_messages(example[column])
    else:
        prompt_col = _find_column(example, PROMPT_COLUMNS)
        response_col = _find_column(example, RESPONSE_COLUMNS)
        if prompt_col is None or response_col is None:
            return None
        turns = [
            ("User", str(example[prompt_col])),
            ("Bot", str(example[response_col])),
        ]
    if len(turns) < 2:
        return None
    return _format_turns(turns)


def _dataset_hint(name, error):
    return (
        f"ERROR: could not load dataset {name!r}:\n{error}\n\n"
        "Check the dataset id, and list its configs with:\n"
        f"  python3 -c \"from datasets import get_dataset_config_names as g;"
        f" print(g('{name}'))\""
    )


def load_hf_corpus(
    name,
    config=None,
    split="train",
    max_examples=None,
    streaming=True,
):
    """Download (or open) a dataset and return it as training text."""
    datasets = _import_datasets()
    kwargs = {"split": split, "streaming": streaming}
    if config:
        kwargs["name"] = config

    try:
        dataset = datasets.load_dataset(name, **kwargs)
    except Exception as error:
        raise SystemExit(_dataset_hint(name, error))

    examples, skipped = [], 0
    for row in dataset:
        text = example_to_text(row)
        if text is None:
            skipped += 1
            continue
        examples.append(text)
        if max_examples is not None and len(examples) >= max_examples:
            break

    if not examples:
        raise SystemExit(
            f"ERROR: no usable conversations in {name!r} (split {split!r}).\n"
            "The dataset may not contain a message list. Try another split, or "
            "export it to text yourself and pass it via --data."
        )

    print(
        f"Loaded {len(examples):,} conversations from {name!r}"
        + (f" [{config}]" if config else "")
        + f" split={split!r}"
        + (f" (skipped {skipped:,} rows without dialogue)" if skipped else "")
    )
    return "\n\n".join(examples)


def load_jsonl_corpus(path, max_examples=None):
    # Read a local .jsonl/.json file of conversations.
    examples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, list):
                row = {"messages": row}
            text = example_to_text(row) if isinstance(row, dict) else None
            if text is None:
                continue
            examples.append(text)
            if max_examples is not None and len(examples) >= max_examples:
                break
    if not examples:
        raise SystemExit(f"ERROR: no usable conversations found in '{path}'.")
    print(f"Loaded {len(examples):,} conversations from '{path}'")
    return "\n\n".join(examples)
