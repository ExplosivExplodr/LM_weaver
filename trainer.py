# ========================================== #
#              AI MODEL TRAINER              #
#              MADE BY EXPLODER              #
# ========================================== #
import os
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

if torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon Metal
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"  # Fallback to cpu if accelerator missing
print(f"LM Trainer Active on Engine: [{device.upper()}]")

# HYPERPARAMETERS
block_size = 64
n_embd = 128
n_head = 4
n_layer = 4


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return F.softmax(wei, dim=-1) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.dropout(self.sa(self.ln1(x)))
        return x + self.dropout(self.ffwd(self.ln2(x)))


class ScratchLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(
            torch.arange(T, device=device)
        )
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


data_filename = "train_data.txt"

print(f"STAGE 1: IMPORTING TRAINING DATA FROM '{data_filename}'...")
if not os.path.exists(data_filename):
    print(f"\n CRITICAL ERROR: '{data_filename}' was not found in this folder!")
    print(
        "Please create a text file named 'train_data.txt' and place your data inside."
    )
    exit()

with open(data_filename, "r", encoding="utf-8") as f:
    raw_text = f.read().strip()

if len(raw_text) <= block_size:
    print(
        "\nCRITICAL ERROR: Your 'train_data.txt' file is empty or too short to analyze."
    )
    exit()

# Split text entries cleanly and shuffle to expand dataset context variations
base_turns = raw_text.split("\n\n")
formatted_dialogues = []
for loop in range(120):
    random.shuffle(base_turns)
    formatted_dialogues.append("\n\n".join(base_turns))

rpg_data = "\n\n".join(formatted_dialogues)
print(f"DATA CONFIGURED! TOTAL ANALYZED CHARS: {len(rpg_data)}")

chars = sorted(list(set(rpg_data)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]

data = torch.tensor(encode(rpg_data), dtype=torch.long)

model = ScratchLanguageModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=5e-4)


def get_batch():
    ix = torch.randint(len(data) - block_size, (32,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# OPTIMIZATION LOOP
# Increase the training_steps value according to your hardware.
# More steps will make the model smarter but it takes more time
training_steps = 5000
print("\nSTAGE 2: INITIATING REAL-TIME OPTIMIZATION LOOP...")
for step in range(training_steps):
    xb, yb = get_batch()
    logits, loss = m(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 250 == 0:
        print(f"STEP {step:4d} | LEARNING LOSS = {loss.item():.4f}")

#  EXPORT COMPILED BUNDLE
try:
    from inputimeout import TimeoutOccurred, inputimeout
except ImportError:
    import os

    os.system("pip install -q inputimeout")
    from inputimeout import TimeoutOccurred, inputimeout

default_name = "my_lm.pth"
print("\nSAVING STAGE:")
print(
    "You have 45 seconds to rename your model. Otherwise it will automatically be named 'my_lm.pth'."
)

try:
    chosen_name = inputimeout(
        prompt="Enter model name (add .pth at the end): ", timeout=45
    ).strip()
    if not chosen_name:
        chosen_name = default_name
except TimeoutOccurred:
    print("\nTIME IS UP!!! INITIATING DEFAULT NAME...")
    chosen_name = default_name

checkpoint = {"model_state_dict": m.state_dict(), "chars_mapping": chars}

torch.save(checkpoint, chosen_name)
print(f"MODEL TRAINING SUCCESS: MODEL BUNDLE SAVED AS '{chosen_name}'")
