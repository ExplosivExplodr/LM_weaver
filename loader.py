# ========================================== #
#              LM MODEL LOADER               #
#              MADE BY EXPLODER              #
# ========================================== #
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Chat Engine Active on Hardware: [{device.upper()}]")

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

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(
            torch.arange(T, device=device)
        )
        return self.lm_head(self.ln_f(self.blocks(x))), None


default_load_name = "H.O.R.C.-1m.pth"
print("\nLOADING MODEL STAGE:")
user_model_name = input(
    f"Enter model file path (Press Enter for default '{default_load_name}'): "
).strip()

if not user_model_name:
    user_model_name = default_load_name

if not os.path.exists(user_model_name):
    print(f"ERROR: Checkpoint file '{user_model_name}' not found!")
else:
    print(f"Extracting weights from '{user_model_name}'...")
    checkpoint = torch.load(user_model_name, map_location=device)

    chars = checkpoint["chars_mapping"]
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join([itos[i] for i in l])

    model = ScratchLanguageModel(vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    m = model.to(device).eval()

    print(
        "Model Loaded And Session Initialised Succesfully! (Type 'exit' to end session)\n"
    )

    chat_history = ""
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Session ended")
            break

        current_turn = f"User: {user_input}\nBot:"
        chat_history += current_turn
        print("Bot: ", end="")
        bot_response = ""

        for _ in range(250):
            constrained_text = chat_history[-block_size:]
            context = torch.tensor(
                [encode(constrained_text)], dtype=torch.long, device=device
            )

            with torch.no_grad():
                logits, _ = m(context)

            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            predicted_char = decode([next_token.item()])

            if predicted_char == "\n" and len(bot_response) > 5:
                break

            print(predicted_char, end="", flush=True)
            bot_response += predicted_char
            chat_history += predicted_char

        print()
        chat_history += "\n"
