# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for aten::embedding (token lookup table) and its weight gradient."""

import pytest
import torch

from tt_kurbla.torch.testing import (
    ExecutionMode,
    assert_close_cpu_vs_tt,
    strict_no_fallback,
)

_MODES = [ExecutionMode.EAGER, ExecutionMode.COMPILE]
_MODE_IDS = [m.value for m in _MODES]


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


def test_embedding_dense_backward_lowered_for_compile() -> None:
    from tt_kurbla.torch import _compile

    op = torch.ops.aten.embedding_dense_backward.default
    assert op in _compile._LOWERINGS
    assert op not in _compile._TT_DECOMPOSITIONS


def _weight_grad(
    dev: str, weight, indices, grad_output, padding_idx=None, compile: bool = False
):
    module = torch.nn.Embedding(
        *weight.shape, padding_idx=padding_idx, dtype=weight.dtype
    )
    with torch.no_grad():
        module.weight.copy_(weight)
    module.to(dev)
    fn = torch.compile(module, backend="tt") if compile else module
    with strict_no_fallback():
        fn(indices.to(dev)).backward(grad_output.to(dev))
    return module.weight.grad.cpu()


# Sequence lengths 40 and 33 are not whole tiles, so ttir -> ttnn pads indices and gradient.
# Vocabularies 100 and 1023 are not whole tiles either: embedding_bw leaves the rows past the
# last full tile unwritten (tt-mlir#9220), so the table is rounded up and sliced back down.
@pytest.mark.parametrize(
    "vocab_size,hidden,idx_shape,padding_idx",
    [
        (128, 64, (1, 32), None),
        (512, 128, (1, 64), None),
        (128, 48, (1, 40), None),
        (160, 64, (1, 33), None),
        (100, 64, (1, 40), None),
        (1023, 64, (1, 64), None),
        (256, 64, (4, 32), None),
        (256, 64, (64,), 5),
        (256, 64, (1, 64), 0),
        (256, 64, (1, 64), 255),
    ],
    ids=[
        "32",
        "64",
        "seq40",
        "seq33",
        "vocab100",
        "vocab1023",
        "batch4",
        "1d_pad5",
        "pad0",
        "pad255",
    ],
)
@pytest.mark.parametrize("mode", _MODES, ids=_MODE_IDS)
def test_embedding_dense_backward(
    vocab_size, hidden, idx_shape, padding_idx, mode: ExecutionMode
) -> None:
    weight = torch.randn((vocab_size, hidden), dtype=torch.bfloat16)
    indices = torch.randint(0, vocab_size, idx_shape, dtype=torch.long)
    # Always hit the last row (the one #9220 drops) and, when there is one, the padding row.
    indices.view(-1)[0] = vocab_size - 1
    if padding_idx is not None:
        indices.view(-1)[1::4] = padding_idx
    grad_output = torch.randn((*idx_shape, hidden), dtype=torch.bfloat16)

    tt = _weight_grad(
        "tt",
        weight,
        indices,
        grad_output,
        padding_idx,
        compile=mode is ExecutionMode.COMPILE,
    )
    cpu = _weight_grad("cpu", weight, indices, grad_output, padding_idx)
    assert tt.shape == cpu.shape
    torch.testing.assert_close(tt, cpu, atol=0.2, rtol=0.2)


def test_embedding_dense_backward_float32() -> None:
    # embedding_bw is a bf16 kernel, so an f32 weight gets an f32 gradient at bf16 accuracy.
    weight = torch.randn((256, 64), dtype=torch.float32)
    indices = torch.randint(0, 256, (1, 64), dtype=torch.long)
    grad_output = torch.randn((1, 64, 64), dtype=torch.float32)
    tt = _weight_grad("tt", weight, indices, grad_output)
    assert tt.dtype == torch.float32
    torch.testing.assert_close(
        tt, _weight_grad("cpu", weight, indices, grad_output), atol=0.2, rtol=0.2
    )


def test_embedding_dense_backward_grad_feeds_other_programs() -> None:
    # ttnn::embedding_bw returns (1, 1, vocab, hidden) whatever the weight's rank; the gradient
    # still has to bind into later programs, eager and compiled.
    w = torch.randn((256, 64), dtype=torch.bfloat16).to("tt").requires_grad_(True)
    indices = torch.randint(0, 256, (1, 64), dtype=torch.long).to("tt")
    torch.nn.functional.embedding(indices, w).backward(
        torch.randn((1, 64, 64), dtype=torch.bfloat16).to("tt")
    )
    g = w.grad
    torch.testing.assert_close((g * 2.0).cpu(), g.cpu() * 2.0)
    step = torch.compile(lambda p, g: p - 0.1 * g, backend="tt")
    torch.testing.assert_close(
        step(w.detach(), g).cpu(),
        w.detach().cpu() - 0.1 * g.cpu(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("mode", _MODES, ids=_MODE_IDS)
def test_embedding_dense_backward_scale_grad_by_freq_rejected(
    mode: ExecutionMode,
) -> None:
    # No histogram kernel to scale rows by index frequency; both paths must refuse rather than
    # return an unscaled gradient.
    module = torch.nn.Embedding(
        64, 32, scale_grad_by_freq=True, dtype=torch.bfloat16
    ).to("tt")
    fn = (
        torch.compile(module, backend="tt") if mode is ExecutionMode.COMPILE else module
    )
    out = fn(torch.randint(0, 16, (1, 32), dtype=torch.long).to("tt"))
    with pytest.raises(NotImplementedError, match="scale_grad_by_freq"):
        out.backward(torch.randn((1, 32, 32), dtype=torch.bfloat16, device="tt"))


@pytest.mark.parametrize(
    "grad_shape,idx_shape,num_weights,padding_idx,match",
    [
        ((1, 16, 64), (1, 32), 64, -1, "match the indices"),
        ((32, 64), (1, 32), 64, -1, "one more than the indices rank"),
        ((2, 3, 32, 64), (2, 3, 32), 128, -1, "indices must be 1D or 2D"),
        ((1, 32, 64), (1, 32), 64, -2, "padding_idx"),
        ((1, 32, 64), (1, 32), 64, 64, "padding_idx"),
        ((1, 32, 64), (1, 32), 0, -1, "num_weights must be positive"),
    ],
    ids=[
        "seq_mismatch",
        "rank_mismatch",
        "3d_indices",
        "padding_below_-1",
        "padding_past_end",
        "no_rows",
    ],
)
def test_embedding_dense_backward_rejects_bad_args(
    grad_shape, idx_shape, num_weights, padding_idx, match
) -> None:
    indices = torch.randint(0, 16, idx_shape, dtype=torch.long).to("tt")
    grad_output = torch.randn(grad_shape, dtype=torch.bfloat16).to("tt")
    with pytest.raises(RuntimeError, match=match):
        torch.ops.aten.embedding_dense_backward(
            grad_output, indices, num_weights, padding_idx, False
        )
