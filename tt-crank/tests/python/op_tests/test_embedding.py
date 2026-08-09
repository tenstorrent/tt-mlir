"""Tests for aten::embedding (token lookup table)."""

import pytest
import torch

from tt_kurbla.torch.testing import assert_close_cpu_vs_tt


@pytest.mark.parametrize(
    "vocab_size,hidden,seq_len",
    [
        (128, 64, 32),
        (512, 128, 64),
        (1024, 64, 128),
    ],
)
def test_embedding(vocab_size: int, hidden: int, seq_len: int) -> None:
    weight = torch.randn((vocab_size, hidden), dtype=torch.bfloat16)
    indices = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    fn = lambda w, idx: torch.nn.functional.embedding(idx, w)
    assert_close_cpu_vs_tt(fn, weight, indices)


def test_embedding_module() -> None:
    # nn.Embedding moves weight to tt; indices stay on CPU (scalar-type Long).
    module = torch.nn.Embedding(256, 64).to(torch.bfloat16)
    indices = torch.randint(0, 256, (1, 128), dtype=torch.long)
    assert_close_cpu_vs_tt(module, indices, atol=1e-2, rtol=1e-2)
