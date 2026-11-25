"""Tests for the shared model definition. Runs with pytest or plain python3.

    python3 -m pytest test_model.py -q
    python3 test_model.py
"""
import tempfile
import unittest

import torch
from torch.nn import functional as F

from .hf_data import example_to_text
from .model import (
    AppConfig,
    Block,
    CharTokenizer,
    ModelConfig,
    ScratchLanguageModel,
    load_checkpoint,
    save_checkpoint,
    transformer_flops_per_token,
)

TINY = ModelConfig(vocab_size=12, block_size=16, n_embd=32, n_head=2, n_layer=2)

class TestModelConfig(unittest.TestCase):
    def test_rejects_indivisible_heads(self):
        with self.assertRaises(ValueError):
            ModelConfig(vocab_size=10, n_embd=128, n_head=6)

    def test_rejects_empty_vocab(self):
        with self.assertRaises(ValueError):
            ModelConfig(vocab_size=0)

    def test_replace_revalidates(self):
        with self.assertRaises(ValueError):
            TINY.replace(n_head=5)

    def test_dict_round_trip_ignores_unknown_keys(self):
        data = TINY.to_dict()
        data["future_option"] = 123
        self.assertEqual(ModelConfig.from_dict(data), TINY)

class TestForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ScratchLanguageModel(TINY)

    def test_output_shape(self):
        idx = torch.randint(0, TINY.vocab_size, (2, 16))
        logits, loss = self.model(idx, idx)
        self.assertEqual(tuple(logits.shape), (2, 16, TINY.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_loss_at_init_is_about_log_vocab(self):
        """An untrained model should be close to uniform prediction.

        With an untied output head and small init, logits start near zero, so
        the loss starts at ln(vocab_size). This catches broken initialisation
        and normalisation bugs that would otherwise hide until training fails.
        """
        idx = torch.randint(0, TINY.vocab_size, (16, 16))
        _, loss = self.model(idx, idx)
        uniform = torch.log(torch.tensor(float(TINY.vocab_size))).item()
        self.assertAlmostEqual(loss.item(), uniform, delta=0.1)

    def test_causality_no_peeking(self):
        """Changing the last token must not alter earlier positions."""
        self.model.eval()
        a = torch.randint(0, TINY.vocab_size, (1, 16))
        b = a.clone()
        b[0, -1] = (b[0, -1] + 5) % TINY.vocab_size
        with torch.no_grad():
            la, _ = self.model(a)
            lb, _ = self.model(b)
        self.assertTrue(torch.equal(la[:, :-1], lb[:, :-1]))
        self.assertFalse(torch.equal(la[:, -1], lb[:, -1]))

    def test_rejects_context_longer_than_block_size(self):
        idx = torch.zeros(1, TINY.block_size + 1, dtype=torch.long)
        with self.assertRaises(ValueError):
            self.model(idx)

    def test_head_is_untied_by_default(self):
        keys = self.model.state_dict().keys()
        self.assertIn("lm_head.weight", keys)
        self.assertNotEqual(
            self.model.lm_head.weight.data_ptr(),
            self.model.token_embedding_table.weight.data_ptr(),
        )

    def test_optional_weight_tying_shares_one_tensor(self):
        tied = ScratchLanguageModel(TINY.replace(tie_embeddings=True))
        self.assertIs(tied.lm_head.weight, tied.token_embedding_table.weight)
        # Stored once, under the embedding, and counted once.
        self.assertNotIn("lm_head.weight", tied.state_dict())
        self.assertEqual(
            tied.num_parameters(), sum(p.numel() for p in tied.parameters())
        )
        self.assertLess(tied.num_parameters(), self.model.num_parameters())

    def test_scales_with_depth(self):
        deeper = ScratchLanguageModel(TINY.replace(n_layer=4))
        self.assertGreater(deeper.num_parameters(), self.model.num_parameters())
        self.assertGreater(transformer_flops_per_token(TINY.replace(n_layer=4)), 0)

class TestGenerate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ScratchLanguageModel(TINY).eval()
        # A tokenizer covering the model's whole vocab, so generated ids decode.
        chars = list("ab -")
        chars += [chr(c) for c in range(33, 33 + TINY.vocab_size - len(chars))]
        self.tokenizer = CharTokenizer(chars)
        self.start = torch.randint(0, TINY.vocab_size, (1, 4))

    def test_appends_requested_tokens(self):
        out = self.model.generate(self.start, max_new_tokens=7)
        self.assertEqual(tuple(out.shape), (1, 11))
        self.assertTrue(torch.equal(out[:, :4], self.start))

    def test_handles_context_longer_than_block_size(self):
        long_start = torch.randint(0, TINY.vocab_size, (1, TINY.block_size * 3))
        out = self.model.generate(long_start, max_new_tokens=3)
        self.assertEqual(out.shape[1], TINY.block_size * 3 + 3)

    def test_greedy_is_deterministic(self):
        a = self.model.generate(self.start, 5, temperature=0)
        b = self.model.generate(self.start, 5, temperature=0)
        self.assertTrue(torch.equal(a, b))

    def test_top_k_one_matches_greedy(self):
        greedy = self.model.generate(self.start, 5, temperature=0)
        top1 = self.model.generate(self.start, 5, temperature=1.0, top_k=1)
        self.assertTrue(torch.equal(greedy, top1))

    def test_top_k_zero_means_disabled(self):
        """top_k=0 is the saved default for "no filtering"; topk(0) would crash."""
        out = self.model.generate(self.start, 5, temperature=0.8, top_k=0)
        self.assertEqual(out.shape[1], self.start.shape[1] + 5)

    def test_top_k_larger_than_vocab_is_safe(self):
        out = self.model.generate(self.start, 3, temperature=0.8, top_k=1000)
        self.assertEqual(out.shape[1], self.start.shape[1] + 3)

    def test_stop_on_halts_generation_early(self):
        # Find a string the model reliably produces, then stop on it.
        full = self.model.generate(self.start, 200, temperature=0)
        produced = self.tokenizer.decode(full[0, self.start.shape[1] :].tolist())
        two = produced[10:12]
        self.assertTrue(two, "model produced no text to stop on")
        out = self.model.generate(
            self.start, 200, temperature=0, stop_on=two, tokenizer=self.tokenizer
        )
        self.assertLess(out.shape[1], full.shape[1])

    def test_stop_on_ignores_prompt_text(self):
        # The stop string is only searched in NEWLY generated text, so a prompt
        # that already contains it must not stop generation immediately.
        ids = self.tokenizer.encode("  ")
        start = torch.tensor([ids])
        out = self.model.generate(
            start, 20, temperature=0, stop_on="  ", tokenizer=self.tokenizer
        )
        self.assertGreater(out.shape[1], start.shape[1])

    def test_no_stop_on_runs_full_length(self):
        out = self.model.generate(
            self.start, 12, temperature=0, stop_on=None, tokenizer=self.tokenizer
        )
        self.assertEqual(out.shape[1], self.start.shape[1] + 12)

class TestTokenizer(unittest.TestCase):
    def test_round_trip(self):
        tokens = CharTokenizer.from_text("User: Hi\nBot: Hello!")
        text = "User: Hi\nBot: Hello!"
        self.assertEqual(tokens.decode(tokens.encode(text)), text)

    def test_reports_unknown_characters(self):
        tokens = CharTokenizer.from_text("abc")
        self.assertEqual(tokens.unknown_chars("aé€"), ["é", "€"])
        self.assertEqual(tokens.encode("aéb"), [0, 1])  # unknown char skipped

class TestCheckpoint(unittest.TestCase):
    def test_save_load_round_trip(self):
        model = ScratchLanguageModel(TINY)
        tokens = CharTokenizer.from_text("abc")
        with tempfile.NamedTemporaryFile(suffix=".pth") as handle:
            save_checkpoint(handle.name, model, tokens, AppConfig())
            loaded = load_checkpoint(handle.name)
        self.assertEqual(loaded["format"], 2)
        self.assertEqual(ModelConfig.from_dict(loaded["config"]), TINY)
        self.assertEqual(CharTokenizer(loaded["chars_mapping"]).chars, tokens.chars)

class TestBlockWiring(unittest.TestCase):
    """Regression tests for the residual stream wiring inside a Block."""

    def setUp(self):
        torch.manual_seed(0)
        self.config = TINY.replace(n_layer=1)
        self.block = Block(self.config).eval()
        self.x = torch.randn(2, TINY.block_size, TINY.n_embd)

    def test_feedforward_sees_attention_output(self):
        """ffwd must read x + attention(x), not the raw x.

        Getting this wrong still trains (the model adapts), so it silently
        degrades quality instead of failing -- hence this test.
        """
        with torch.no_grad():
            att_out = self.x + self.block.sa(self.block.ln1(self.x))
            expected = att_out + self.block.ffwd(self.block.ln2(att_out))
            actual = self.block(self.x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_is_not_the_naive_parallel_wiring(self):
        with torch.no_grad():
            naive = (
                self.x
                + self.block.sa(self.block.ln1(self.x))
                + self.block.ffwd(self.block.ln2(self.x))
            )
            actual = self.block(self.x)
        self.assertFalse(torch.allclose(actual, naive, atol=1e-4))

class TestLegacyCompatibility(unittest.TestCase):
    """An old-architecture config must reproduce the old forward pass."""

    def test_original_config_matches_reference_forward(self):
        config = TINY.replace(
            bias=True,
            bias_head=False,
            bias_head_out=True,
            act="relu",
            residual_dropout=False,
        )
        torch.manual_seed(0)
        model = ScratchLanguageModel(config).eval()
        state = model.state_dict()
        idx = torch.randint(0, TINY.vocab_size, (1, TINY.block_size))
        with torch.no_grad():
            logits, _ = model(idx)

        # Hand-rolled reference: the original trainer.py architecture.
        n_embd, n_head, T = config.n_embd, config.n_head, TINY.block_size
        head_size = n_embd // n_head
        x = (
            state["token_embedding_table.weight"][idx]
            + state["position_embedding_table.weight"][:T]
        )
        for i in range(config.n_layer):
            p = f"blocks.{i}."
            h = F.layer_norm(x, (n_embd,), state[p + "ln1.weight"], state[p + "ln1.bias"])
            heads = []
            for j in range(n_head):
                hp = p + f"sa.heads.{j}."
                k = h @ state[hp + "key.weight"].T
                q = h @ state[hp + "query.weight"].T
                v = h @ state[hp + "value.weight"].T
                w = q @ k.transpose(-2, -1) * (head_size**-0.5)
                w = w.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
                heads.append(F.softmax(w, dim=-1) @ v)
            x = x + (
                torch.cat(heads, -1) @ state[p + "sa.proj.weight"].T
                + state[p + "sa.proj.bias"]
            )
            h = F.layer_norm(x, (n_embd,), state[p + "ln2.weight"], state[p + "ln2.bias"])
            h = F.relu(h @ state[p + "ffwd.net.0.weight"].T + state[p + "ffwd.net.0.bias"])
            x = x + h @ state[p + "ffwd.net.2.weight"].T + state[p + "ffwd.net.2.bias"]
        x = F.layer_norm(x, (n_embd,), state["ln_f.weight"], state["ln_f.bias"])
        reference = x @ state["lm_head.weight"].T + state["lm_head.bias"]
        self.assertTrue(torch.allclose(logits, reference, atol=1e-5))

class TestHFDatasetConversion(unittest.TestCase):
    """hf_data turns dataset rows into the plain 'User:'/'Bot:' training text."""

    def test_openai_messages(self):

        text = example_to_text(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            }
        )
        self.assertEqual(text, "User: Hi\nBot: Hello!")

    def test_sharegpt_shape(self):

        text = example_to_text(
            {
                "conversations": [
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello!"},
                ]
            }
        )
        self.assertEqual(text, "User: Hi\nBot: Hello!")

    def test_system_messages_are_dropped(self):

        text = example_to_text(
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            }
        )
        self.assertEqual(text, "User: Hi\nBot: Hello!")
        self.assertNotIn("helpful", text)

    def test_prompt_response_columns(self):

        text = example_to_text({"question": "2+2?", "answer": "4"})
        self.assertEqual(text, "User: 2+2?\nBot: 4")

    def test_multiline_content_is_flattened(self):
        """A newline inside a message must not look like a turn separator."""

        text = example_to_text(
            {
                "messages": [
                    {"role": "user", "content": "one\ntwo"},
                    {"role": "assistant", "content": "ok"},
                ]
            }
        )
        self.assertEqual(text, "User: one two\nBot: ok")
        self.assertNotIn("\n\n", text)

    def test_non_conversational_row_returns_none(self):

        self.assertIsNone(example_to_text({"text": "just prose"}))
        self.assertIsNone(example_to_text({"messages": [{"role": "user",
                                                         "content": "only"}]}))

    def test_unknown_roles_are_not_mislabelled(self):

        text = example_to_text(
            {
                "messages": [
                    {"role": "wizard", "content": "abracadabra"},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            }
        )
        self.assertNotIn("abracadabra", text)

if __name__ == "__main__":
    unittest.main(verbosity=2)
