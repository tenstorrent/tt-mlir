"""Trained Llama-3.2-1B weights and a real token sequence for the seq-512
forward graph, plus a CPU reference that computes the same loss in torch.

Synthetic N(0, 0.02) weights make the model a near-uniform predictor: the loss
sits at ln(vocab) and the logits are flat, which is a forgiving numerical
regime. tt-mlir#9189 reports its divergence at loss 5.84, where the logits are
peaked. This module feeds the graph the weights and tokens that put it there.

The weights come from a local HF snapshot; nothing is downloaded. `lm_head` is
tied to the embedding, matching the graph's final `ttir.linear(h, arg0)`.
"""

import json
import math
import mmap
from pathlib import Path

VOCAB = 128256
SEQ = 512
LABEL_LEN = SEQ - 1

# Per-decoder-layer argument order in the graph, verified against its ops.
LAYER_PARAMS = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
]
NUM_LAYERS = 16
ARG_INV_FREQ = 1 + NUM_LAYERS * len(LAYER_PARAMS) + 1  # 146
ARG_INPUT_IDS = ARG_INV_FREQ + 1
ARG_ATTENTION_MASK = ARG_INV_FREQ + 2
ARG_LABELS_ONE_HOT = ARG_INV_FREQ + 3
ARG_LOSS_WEIGHT = ARG_INV_FREQ + 4

SAFETENSORS_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I64": "int64",
    "I32": "int32",
    "I8": "int8",
    "U8": "uint8",
}

# Ordinary English prose, long enough to fill 512 tokens. Kept in the file so a
# run does not depend on any text on disk staying put.
TEXT = """A compiler for an AI accelerator has an unusual job. It is not
translating a language that people write by hand; it is taking a graph that
another program produced, and deciding where every tensor should live and in
what order the work should happen. The decisions are not local. Choosing to
keep one activation in fast memory may force another out of it, and the cost of
that choice is not visible until much later in the pipeline, when a kernel
either fits or does not.

This is why an optimizer for such a compiler is hard to test. A correctness bug
in a conventional compiler usually shows up as a crash or as obviously wrong
output. Here the program still runs, the shapes still line up, and the numbers
that come back are merely wrong in a way that only a trained model can reveal.
A network with random weights predicts nothing in particular, so its loss sits
at the entropy of the vocabulary no matter what the hardware does. A trained
network is different. Its predictions are sharp, and a sharp prediction is
fragile: a small error early in the forward pass moves the logits, the softmax
amplifies the movement, and the loss records it.

The practical lesson is that a reproduction needs to run in the same regime as
the report. If the original failure happened while fine tuning a language model
on a sentiment classification task, then the reproduction should carry real
weights and real tokens, not a convenient approximation of them. Otherwise a
clean result proves very little. It says the graph compiles and executes, which
is worth knowing, but it does not say the arithmetic is right.

There is a second lesson about measurement. When a run is slow the first time
and fast the second, the difference is usually a cache somewhere, and comparing
a cold run against a warm one will invent a regression that does not exist.
Kernels are compiled once and reused, weights are uploaded once and kept, and
the profiler counts everything the first time through. The only honest
comparison is between two runs in the same state.

None of this is specific to accelerators. It is the ordinary discipline of
measurement: hold everything fixed except the one thing being varied, check
that the instrument responds when the input changes, and be suspicious of a
result that agrees too neatly with what you hoped to find. The difference is
that here the instrument is a compiler, the input is a graph with a billion
parameters in it, and the thing being varied is a single integer on a command
line that reorganizes the entire memory plan of the program."""


def weight_name(index):
    """HF tensor name for a parameter argument, or None if it is not one."""
    if index == 0:
        return "model.embed_tokens.weight"
    if 1 <= index <= NUM_LAYERS * len(LAYER_PARAMS):
        offset = index - 1
        layer, slot = divmod(offset, len(LAYER_PARAMS))
        return f"model.layers.{layer}.{LAYER_PARAMS[slot]}"
    if index == NUM_LAYERS * len(LAYER_PARAMS) + 1:
        return "model.norm.weight"
    return None


def _byte_encoder():
    """GPT-2 byte-to-unicode map, the encoding tokenizer.json vocabs use."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAD))
        + list(range(0xAE, 0x100))
    )
    mapped, extra = list(printable), 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, mapped)}


class Snapshot:
    """A local HF model snapshot: mmapped safetensors, config and tokenizer."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.config = json.loads((self.dir / "config.json").read_text())
        self._file = open(self.dir / "model.safetensors", "rb")
        header_len = int.from_bytes(self._file.read(8), "little")
        self.header = json.loads(self._file.read(header_len))
        self._data_start = 8 + header_len
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def tensor(self, name):
        import torch

        meta = self.header[name]
        start, end = meta["data_offsets"]
        raw = bytearray(
            self._map[self._data_start + start : self._data_start + end]
        )
        dtype = getattr(torch, SAFETENSORS_DTYPES[meta["dtype"]])
        return torch.frombuffer(raw, dtype=dtype).reshape(meta["shape"])

    def inv_freq(self):
        """Rotary inv_freq, with llama3 scaling applied as HF applies it."""
        import torch

        cfg = self.config
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
        inv = 1.0 / (
            cfg["rope_theta"]
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        scaling = cfg.get("rope_scaling")
        if not scaling or scaling.get("rope_type") != "llama3":
            return inv

        factor = scaling["factor"]
        low_factor = scaling["low_freq_factor"]
        high_factor = scaling["high_freq_factor"]
        old_len = scaling["original_max_position_embeddings"]
        low_wavelen, high_wavelen = old_len / low_factor, old_len / high_factor

        wavelen = 2 * math.pi / inv
        scaled = torch.where(wavelen > low_wavelen, inv / factor, inv)
        smooth = (old_len / wavelen - low_factor) / (high_factor - low_factor)
        smoothed = (1 - smooth) * scaled / factor + smooth * scaled
        is_medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
        return torch.where(is_medium, smoothed, scaled)

    def tokenize(self, text, length):
        """Greedy longest-match over the vocab; close enough to BPE for prose.

        Returns exactly `length` ids, cycling the text if it runs short.
        """
        vocab = json.loads((self.dir / "tokenizer.json").read_text())["model"]["vocab"]
        encoder = _byte_encoder()
        encoded = "".join(encoder[b] for b in text.encode("utf-8"))
        longest = max(len(token) for token in vocab)

        ids, position = [], 0
        while position < len(encoded):
            for size in range(min(longest, len(encoded) - position), 0, -1):
                token = encoded[position : position + size]
                if token in vocab:
                    ids.append(vocab[token])
                    position += size
                    break
            else:  # unmapped byte, should not happen with a byte-level vocab
                position += 1
        if not ids:
            raise RuntimeError("tokenizer produced no ids")
        while len(ids) < length:
            ids += ids
        return ids[:length]


class RealInputs:
    """Argument-index keyed inputs: trained weights and a real token sequence.

    Labels are the next token at each position, so the loss is the model's mean
    next-token NLL on the text — the regime the issue reports from.
    """

    def __init__(self, model_dir):
        import torch

        self.snapshot = Snapshot(model_dir)
        self.ids = torch.tensor(
            self.snapshot.tokenize(TEXT, SEQ), dtype=torch.int64
        ).unsqueeze(0)
        self.index = -1

    def targets(self):
        return self.ids[:, 1:]

    def __call__(self, shape, dtype):
        import torch

        self.index += 1
        index = self.index
        shape = list(shape)

        name = weight_name(index)
        if name is not None:
            tensor = self.snapshot.tensor(name)
            if list(tensor.shape) != shape:
                raise RuntimeError(
                    f"argument {index} ({name}) is {shape}, snapshot has "
                    f"{list(tensor.shape)}"
                )
            return tensor.to(dtype)

        if index == ARG_INV_FREQ:
            return self.snapshot.inv_freq().to(dtype)
        if index == ARG_INPUT_IDS:
            return self.ids.to(dtype)
        if index == ARG_ATTENTION_MASK:
            return torch.ones(shape, dtype=dtype)
        if index == ARG_LABELS_ONE_HOT:
            out = torch.zeros(shape, dtype=dtype)
            out.scatter_(2, self.targets().unsqueeze(-1), 1.0)
            return out
        if index == ARG_LOSS_WEIGHT:
            return torch.full(shape, -1.0 / LABEL_LEN, dtype=dtype)

        raise RuntimeError(f"no real input for argument {index} shape={shape}")


def cpu_reference_loss(model_dir, log=print):
    """Recompute the graph's loss in torch on CPU, op for op.

    Mirrors the graph's dtypes: bf16 weights and activations, f32 for the norm
    statistics, rope and attention, bf16 for the softmax over the vocabulary.
    """
    import torch

    snapshot = Snapshot(model_dir)
    cfg = snapshot.config
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["hidden_size"] // heads
    eps = cfg["rms_norm_eps"]

    ids = torch.tensor(snapshot.tokenize(TEXT, SEQ), dtype=torch.int64)
    positions = torch.arange(SEQ, dtype=torch.float32)
    freqs = snapshot.inv_freq()[:, None] @ positions[None, :]
    emb = torch.cat([freqs, freqs], dim=0).transpose(0, 1)
    cos = emb.cos().to(torch.bfloat16)[None, None]
    sin = emb.sin().to(torch.bfloat16)[None, None]

    causal = torch.full((SEQ, SEQ), float("-inf"))
    causal = torch.triu(causal, diagonal=1)[None, None]

    def rms_norm(x, weight):
        f = x.float()
        scale = torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
        return ((f * scale).to(torch.bfloat16) * weight).to(torch.bfloat16)

    def linear(x, weight):
        return (x.float() @ weight.float().t()).to(torch.bfloat16)

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    embed = snapshot.tensor("model.embed_tokens.weight")
    h = embed[ids].unsqueeze(0)

    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        w = lambda suffix: snapshot.tensor(prefix + suffix)

        y = rms_norm(h, w("input_layernorm.weight"))
        q = linear(y, w("self_attn.q_proj.weight")).view(1, SEQ, heads, head_dim)
        k = linear(y, w("self_attn.k_proj.weight")).view(1, SEQ, kv_heads, head_dim)
        v = linear(y, w("self_attn.v_proj.weight")).view(1, SEQ, kv_heads, head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        q = (q * cos + rotate_half(q) * sin).to(torch.bfloat16)
        k = (k * cos + rotate_half(k) * sin).to(torch.bfloat16)
        repeat = heads // kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        scores = (q.float() * head_dim**-0.25) @ (k.float() * head_dim**-0.25).transpose(2, 3)
        attn = torch.softmax(scores + causal, dim=-1) @ v.float()
        attn = attn.to(torch.bfloat16).transpose(1, 2).reshape(1, SEQ, -1)
        h = (h + linear(attn, w("self_attn.o_proj.weight"))).to(torch.bfloat16)

        y = rms_norm(h, w("post_attention_layernorm.weight"))
        gate = torch.nn.functional.silu(linear(y, w("mlp.gate_proj.weight")).float())
        up = linear(y, w("mlp.up_proj.weight")).float()
        mlp = linear((gate * up).to(torch.bfloat16), w("mlp.down_proj.weight"))
        h = (h + mlp).to(torch.bfloat16)
        log(f"  layer {layer} done")

    h = rms_norm(h, snapshot.tensor("model.norm.weight"))
    logits = linear(h, embed)[:, :LABEL_LEN]
    probs = torch.softmax(logits.float(), dim=-1).to(torch.bfloat16)
    target = probs.gather(2, ids[1:].view(1, LABEL_LEN, 1)).float()
    return (target.clamp(min=1e-12).log() * (-1.0 / LABEL_LEN)).sum().item()
