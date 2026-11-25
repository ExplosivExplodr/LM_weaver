# ========================================== #
#              LM MODEL DEFINITION           #
# ========================================== #
# This is the one and only definition of the network. Both trainer.py and
# loader.py import it, so a model trained by one is always loadable by the
# other -- there is no second copy of these classes to drift out of sync.
#
# Everything that defines the architecture's *shape* lives in ModelConfig,
# which is saved inside every checkpoint. Change the numbers to scale the
# model up or down; nothing else in the code needs to change.
import math
from dataclasses import asdict, dataclass, fields, replace

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    """The architecture of the network. Saved verbatim in each checkpoint."""

    vocab_size: int = 44  # number of distinct characters (set from data)
    block_size: int = 64  # characters of context the model sees at once
    n_embd: int = 128  # width of the residual stream
    n_head: int = 4  # attention heads per block; must divide cleanly n_embd
    n_layer: int = 4  # stacked transformer blocks
    dropout: float = 0.1
    bias: bool = True  # bias terms in the block Linears and LayerNorms
    bias_head: bool = False  # bias in the per-head key/query/value projections
    bias_head_out: bool = True  # bias on the output projection (lm_head)
    # Tie the output projection to the token embedding. Off by default because
    # this architecture is trained per-character on a small corpus: tying forces
    # p(next) to be an inner product with the *input* embedding, which destroys
    # an already-trained model (measured loss 0.14 -> 46.7 on this dataset).
    tie_embeddings: bool = False
    act: str = "gelu"  # feedforward activation: "gelu" or "relu"
    residual_dropout: bool = False  # dropout on each residual add in a Block

    def __post_init__(self):
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.n_layer < 0:
            raise ValueError(f"n_layer must be >= 0, got {self.n_layer}")
        if self.act not in ("gelu", "relu"):
            raise ValueError(f"act must be 'gelu' or 'relu', got {self.act!r}")

    def replace(self, **changes):
        """Return a copy with some fields changed, re-validated."""
        return replace(self, **changes)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Rebuild from a checkpoint dict, ignoring unknown/future keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def count_parameters(model):
    """Number of parameters. Tied weights are counted once."""
    return sum(p.numel() for p in model.parameters())


def transformer_flops_per_token(config):
    """Rough forward FLOPs per token (~2 * params + attention terms)."""
    params = count_parameters(ScratchLanguageModel(config))
    attention = config.n_layer * 12 * config.block_size * config.n_embd
    return 2 * params + attention


def make_layer_norm(config):
    """LayerNorm whose bias can be disabled.

    nn.LayerNorm gained a `bias` argument in torch 2.1, so set the attribute
    directly instead of passing it -- this keeps the module loadable by older
    PyTorch versions.
    """
    layer_norm = nn.LayerNorm(config.n_embd)
    if not config.bias:
        layer_norm.bias = None
    return layer_norm


class Head(nn.Module):
    """One causal self-attention head."""

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, head_size, bias=config.bias_head)
        self.query = nn.Linear(config.n_embd, head_size, bias=config.bias_head)
        self.value = nn.Linear(config.n_embd, head_size, bias=config.bias_head)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        # a context longer than block_size would leave part of the sequence
        # unmasked, letting the model peek at future characters.
        if T > self.tril.size(0):
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.tril.size(0)}"
            )
        k, q, v = self.key(x), self.query(x), self.value(x)
        # Scale by the per-head size (32 here), not the embedding width. This
        # matches the original model; using C**-0.5 would change the softmax
        # temperature and break previously trained checkpoints.
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return self.dropout(wei) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        act = {"gelu": nn.GELU(), "relu": nn.ReLU()}[config.act]
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            act,
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block: attention then feedforward, both residual."""

    def __init__(self, config):
        super().__init__()
        self.residual_dropout = config.residual_dropout
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = make_layer_norm(config)
        self.ln2 = make_layer_norm(config)
        # A separate dropout on the residual adds (as in the original model).
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Each sub-layer reads the running residual stream, and the
        # feedforward must see the attention output -- so add sa before ffwd.
        x = x + self.sa(self.ln1(x))
        ffwd = self.ffwd(self.ln2(x))
        if self.residual_dropout:
            ffwd = self.dropout(ffwd)
        return x + ffwd


class TiedLinear(nn.Module):
    """A Linear layer that reuses another module's weight (weight tying).

    The weight is not registered here, so it appears in state_dict() only once
    -- under the embedding -- and older PyTorch versions load it cleanly.
    """

    def __init__(self, weight):
        super().__init__()
        # Bypass nn.Module.__setattr__ so the shared tensor is NOT registered
        # here; it stays registered once, on the embedding that owns it.
        object.__setattr__(self, "weight", weight)

    def forward(self, x):
        return F.linear(x, self.weight)


class ScratchLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = make_layer_norm(config)
        if config.tie_embeddings:
            # Optional weight tying: saves vocab_size * n_embd parameters.
            self.lm_head = TiedLinear(self.token_embedding_table.weight)
        else:
            self.lm_head = nn.Linear(
                config.n_embd, config.vocab_size, bias=config.bias_head_out
            )

        self.apply(self._init_weights)
        # GPT-2 trick: scale down residual projections so the residual stream
        # does not blow up as depth increases.
        std = 0.02 / math.sqrt(2 * max(config.n_layer, 1))
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("ffwd.net.2.weight"):
                nn.init.normal_(param, mean=0.0, std=std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self):
        return count_parameters(self)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Positions beyond block_size have no embedding, and the causal mask
        # can't cover them -- so fail loudly instead of leaking future tokens.
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.config.block_size}"
            )
        x = self.token_embedding_table(idx) + self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            # ignore_index=-1 lets callers mask out target positions.
            loss = F.cross_entropy(
                logits.view(B * T, -1), targets.view(B * T), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 stop_on=None, tokenizer=None):
        """Autoregressively extend idx (B, T) to (B, T + max_new_tokens).

        temperature: 0 means greedy; <1 is more conservative; >1 is wilder.
        top_k: sample only among the k most likely characters (0 = disabled).
        stop_on: optional string. Generation stops as soon as the *newly
            generated* text contains it. Requires `tokenizer` to decode.
            This is what keeps a chat model from narrating the next turn: the
            blank line that ends a reply is a reliable stop signal, whereas
            waiting for a following "User:" is not.
        """
        generated = []
        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens: all the model can see.
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # predictions for the next character
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                # top_k <= 0 means "no filtering" (topk(0) would return nothing).
                if top_k is not None and top_k >= 1:
                    k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, k)
                    logits[logits < values[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if stop_on and tokenizer is not None:
                generated.append(int(next_token[0, 0]))
                if len(generated) >= len(stop_on) and stop_on in tokenizer.decode(
                    generated
                ):
                    break
        return idx


# ----------------------------------------------------------------------------
# Tokenizer: character level. Kept with the model so the checkpoint carries
# everything needed to encode/decode its own text.
# ----------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, chars):
        if not chars:
            raise ValueError("tokenizer needs at least one character")
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    def encode(self, text):
        """Characters not in the vocabulary are skipped."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        """Map ids back to text. Unknown ids are skipped rather than raising."""
        return "".join(self.itos[int(i)] for i in ids if int(i) in self.itos)

    def unknown_chars(self, text):
        """Characters in text that this tokenizer cannot represent."""
        return sorted({c for c in text if c not in self.stoi})


# ----------------------------------------------------------------------------
# Checkpoint I/O: one format, defined once.
# ----------------------------------------------------------------------------
CHECKPOINT_FORMAT = 2


@dataclass(frozen=True)
class AppConfig:
    """Training and generation settings. Independent of the network shape."""

    # optimisation
    lr: float = 1e-3
    min_lr: float = 1e-4  # cosine decay floor
    warmup_steps: int = 100
    grad_clip: float = 1.0
    batch_size: int = 32
    weight_decay: float = 0.01
    # data
    shuffle_repeats: int = 120  # duplicate corpus this many times, reshuffled
    val_fraction: float = 0.1
    seed: int = 1337
    # generation
    temperature: float = 0.8  # 0 = greedy
    top_k: int = 0  # 0 = disabled
    max_new_tokens: int = 250

    def to_dict(self):
        return asdict(self)

    def replace(self, **changes):
        """Return a copy with some fields changed."""
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def save_checkpoint(path, model, tokenizer, app_config, extra=None):
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "config": model.config.to_dict(),
        "chars_mapping": tokenizer.chars,
        "model_state_dict": model.state_dict(),
        "app_config": app_config.to_dict(),
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(path, map_location="cpu"):
    """Load a checkpoint dict saved by save_checkpoint()."""
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # weights_only was added in torch 2.0; older versions reject it.
        checkpoint = torch.load(path, map_location=map_location)
    return checkpoint

