This project lets you train your own small language model, export it in pytorch format (.pt/.pth) and learn the basics of how language models work.

## 1. Install Dependencies

Open your terminal and install the required Python libraries:

```bash
pip3 install requirements.txt
```
Or optionally do it manually
```bash
pip3 install torch inputimeout
```

`inputimeout` is optional -- it only adds the 45-second countdown to the save
prompt. Without it the trainer simply asks for a name and waits.
(NumPy is not required; it only comes in as a transitive dependency of torch.)

## 2. Add Training Data

Open `train_data.txt` and paste your training text (dialogue examples, essays,
book excerpts...). Separate entries with a blank line. Avoid excessively large
datasets (e.g. 70 MB or more).

You can use your own txt file too with the new --data flag

### Training straight from a Hugging Face dataset

Instead of writing a text file, you can point the trainer at any conversation
dataset on the Hugging Face Hub:

```bash
pip install datasets        # one-time, optional

# a small everyday-dialogue subset
python3 trainer.py --hf-dataset HuggingFaceTB/smoltalk \
    --hf-config everyday-conversations --hf-max-examples 300

# a bigger one, different split
python3 trainer.py --hf-dataset HuggingFaceH4/ultrachat_200k --hf-split train_sft
```

Supported shapes, auto-detected per row:

| Dataset style | Example fields |
| --- | --- |
| OpenAI / SmolTalk / UltraChat | `messages: [{role: user/assistant, content}]` |
| ShareGPT | `conversations: [{from: human/gpt, value}]` |
| single-turn | `prompt` + `response` (or `question`/`answer`, `instruction`/`output`) |

`system`, `tool` and unknown roles are dropped rather than mislabelled, and
multi-line messages are flattened so a newline can't be mistaken for a turn
break. Everything is converted to the same `User:` / `Bot:` text format used by
`train_data.txt` -- so the loader behaves identically either way.

Flags: `--hf-config` (subset name), `--hf-split` (default `train`),
`--hf-max-examples` (stop after N conversations). `--hf-dataset` also accepts a
local `.jsonl` file with one conversation per line.

`--hf-config` is **required for datasets that bundle several subsets** --
`smoltalk` ships 14, so a bare `--hf-dataset HuggingFaceTB/smoltalk` cannot
pick one and the trainer will stop and list the valid names. Single-subset
datasets such as `HuggingFaceH4/ultrachat_200k` work without it.

Note: HF datasets are far larger than `train_data.txt`. `--hf-max-examples`
controls how much you take, and a preset that is fine for 20 exchanges will be
heavily undertrained on 300 -- use `--preset base` or `big` and a few thousand
steps.

## 3. Train the Model

```bash
python3 trainer.py
```

Monitor TRAIN and VAL loss as the model learns. VAL loss is measured on the last
10% of your data, which is never trained on, if it starts rising while TRAIN
loss keeps falling, the model is overfitting. 

## 4. Run the Model

```bash
python3 loader.py                 # tries to load the default name (model.pth)
python3 loader.py tiny.pth        # a specific checkpoint
python3 loader.py --temp 0.6 --top-k 20
python3 loader.py --prompt "User: Hi
Bot:"   # one-shot reply, then exit
```

Type `exit` (or press Ctrl+D) to end a chat. Lower `--temp` is more predictable,
higher is more creative; `--top-k 20` keeps sampling to the 20 most likely
characters.

Each reply stops at the blank line that ends a turn, so the model answers once
and waits for you instead of continuing to write the next `User:` turn. The
reliable stop signal is the blank line, not the following word.

## Choosing the model size

Everything about the network's shape lives in `ModelConfig` in
`src/lm_weaver/model.py`. You can pick a preset or override any single value
from the command line:

```bash
python3 trainer.py --list-presets        # print sizes with your vocabulary
python3 trainer.py --preset tiny         # 107k params, trains faster
python3 trainer.py --preset big          # 4.8M params
python3 trainer.py --n-layer 8 --steps 20000
python3 trainer.py --data my.txt --out run1.pth
```

| preset | parameters | n_embd | n_head | n_layer | block_size |
|--------|-----------:|-------:|-------:|--------:|-----------:|
| tiny   |    107,436 |     64 |      4 |       2 |         32 |
| small  |    349,484 |     96 |      4 |       3 |         64 |
| base   |    811,308 |    128 |      4 |       4 |         64 |
| big    |  4,789,804 |    256 |      8 |       6 |        128 |

(Measured with this dataset's 44-character vocabulary; your numbers will differ
slightly with a different corpus.)

Rules of thumb:

- **Wider/deeper needs more data.** On a small corpus, `big` will overfit:
  VAL loss bottoms out early and climbs. Watch it and prefer a smaller preset.
- **`n_embd` must be divisible by `n_head`.** The trainer says so clearly if not.
- **`block_size`** is how many characters of context the model sees. Longer
  context needs more data to pay off.
- Each extra layer at `base` width costs about 197k parameters.

Run `PYTHONPATH=src python3 -m lm_weaver.test_model` to check the model still
behaves (shapes, causality, checkpoint round-trip, turn-boundary stopping,
Hugging Face conversion). It needs no extra libraries.

## Using an older model

Checkpoints from before format 2 are no longer supported -- they stored no
architecture information, so the loader could not rebuild the network. Retrain
with the current `trainer.py`; the bundled `example.pth` shows the current
format.

## Project Files

```
trainer.py            thin wrapper: python3 trainer.py  -> src/lm_weaver/trainer.py
loader.py             thin wrapper: python3 loader.py   -> src/lm_weaver/loader.py
train_data.txt        the dataset you place text in
example.pth           a small model trained from train_data.txt
src/lm_weaver/
  __init__.py         package exports
  __main__.py         python3 -m lm_weaver train|chat
  model.py            network, ModelConfig, tokenizer, checkpoint format
  trainer.py          trains a model and saves a .pth
  loader.py           loads a .pth so you can test or chat with it
  hf_data.py          converts Hugging Face datasets into training text
  test_model.py       tests for the model and dataset conversion
```

`src/lm_weaver/model.py` is the single source of truth for the network, so a
model trained by `trainer.py` always loads in `loader.py`.

The two root wrappers only forward to the package, which keeps
`python3 trainer.py` working from a fresh clone with no install step. If you
prefer the packaged entry points:

```bash
pip install -e .                 # or: pip install -e '.[hf]'
lm-weaver train --preset small
lm-weaver chat example.pth
python3 -m lm_weaver train       # same thing, no install (needs PYTHONPATH=src)
```

Run the tests from the repo root:

```bash
PYTHONPATH=src python3 -m lm_weaver.test_model
# or, after pip install -e . :
python3 -m pytest
```

Checkpoints store their architecture and sampling settings next to the weights,
so changing defaults in `trainer.py` never breaks an existing model.

## What each checkpoint contains

| key | meaning |
|-----|---------|
| `format` | checkpoint format version (2) |
| `config` | architecture: vocab, block_size, n_embd, n_head, n_layer... |
| `chars_mapping` | characters, so the model can encode/decode its own text |
| `model_state_dict` | the trained weights |
| `app_config` | sampling settings (temperature, top_k) saved by the trainer |
| `final_train_loss`, `final_val_loss`, `best_val_loss` | training results |

These scripts train small models. While the model can learn basic grammar and
sentence structure, it lacks reasoning abilities and cannot perform math.
