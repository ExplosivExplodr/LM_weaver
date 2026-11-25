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
import difflib
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


def _pick_config_hint(configs, wanted=None):
    """Suggest the closest config name for a typo, or None."""
    if not configs:
        return None
    if wanted:
        pool = configs
        target = wanted.lower()
        best = max(pool, key=lambda c: difflib.SequenceMatcher(
            None, target, c.lower()).ratio())
        return best
    return configs[0]


def _describe_missing_config(name, configs):
    """A short, copy-pasteable error when a dataset needs a config name."""
    shown = configs[:12]
    more = f" (and {len(configs) - len(shown)} more)" if len(configs) > len(shown) else ""
    suggestion = _pick_config_hint(configs)
    return (
        f"ERROR: {name!r} contains {len(configs)} subsets and has no default,\n"
        "so --hf-config is required.\n\n"
        f"Available: {', '.join(shown)}{more}\n\n"
        f"Try:\n  python3 trainer.py --hf-dataset {name} "
        f"--hf-config {suggestion}"
    )


def _describe_bad_config(name, configs, wanted):
    """A short error when --hf-config doesn't exist (usually a typo)."""
    suggestion = _pick_config_hint(configs, wanted)
    hint = f"\nDid you mean {suggestion!r}?" if suggestion else ""
    shown = ", ".join(configs[:12])
    return (
        f"ERROR: {name!r} has no subset called {wanted!r}.{hint}\n"
        f"Available: {shown}"
    )


def _available_configs(datasets, name):
    """Best-effort list of a dataset's config names; [] if unknown."""
    try:
        return list(datasets.get_dataset_config_names(name) or [])
    except Exception:
        return []


def _load_dataset(datasets, name, config, split, streaming):
    """Call load_dataset, turning Hub errors into actionable messages.

    A dataset with a single default config works without --hf-config; one
    with many configs (smoltalk) refuses to load until you name one, so we
    look the list up and say which to pass instead of echoing Hub output.
    """
    kwargs = {"split": split, "streaming": streaming}
    if config:
        kwargs["name"] = config
    try:
        return datasets.load_dataset(name, **kwargs)
    except Exception as error:
        message = str(error)
        # The Hub's own message for "you must choose a config".
        needs_config = "Config name is missing" in message
        if needs_config or ("BuilderConfig" in message and "not found" in message):
            configs = _available_configs(datasets, name)
            if config:
                if configs:
                    raise SystemExit(_describe_bad_config(name, configs, config))
            elif configs:
                raise SystemExit(_describe_missing_config(name, configs))
        raise SystemExit(_dataset_hint(name, error))


def load_hf_corpus(
    name,
    config=None,
    split="train",
    max_examples=None,
    streaming=True,
):
    """Download (or open) a dataset and return it as training text."""
    datasets = _import_datasets()
    dataset = _load_dataset(datasets, name, config, split, streaming)

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
