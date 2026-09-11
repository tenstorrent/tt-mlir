import pytest
import torch
import torch.nn.functional as F

from tt_kurbla.torch.testing import assert_close_cpu_vs_tt, strict_no_fallback

_REDUCTIONS = ["none", "mean", "sum"]


# Tile-aligned (multiples of 32) bf16 shapes only for now:
#  - bf16 because ttsim hits UB on TTNN-emitted f32 kernels
#    (see tests/engine_execution_payload_test.cpp).
#  - Tile-aligned because non-aligned shapes segfault inside the TTNN
#    compile pipeline on small inputs — to be investigated separately.
@pytest.mark.parametrize("shape", [(64, 128), (32, 32), (32, 64, 32)])
def test_add(shape: tuple[int, ...]) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.add, a, b)


@pytest.mark.parametrize("alpha", [2.0, 0.5, -1.0])
def test_add_alpha(alpha: float) -> None:
    a = torch.randn((32, 32), dtype=torch.bfloat16)
    b = torch.randn((32, 32), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x, y: torch.add(x, y, alpha=alpha), a, b)


@pytest.mark.parametrize("shape", [(64, 128), (32, 32), (32, 64, 32)])
def test_relu(shape: tuple[int, ...]) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.relu, a)


@pytest.mark.parametrize("shape", [(64, 128), (32, 32), (32, 64, 32)])
def test_sub(shape: tuple[int, ...]) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.sub, a, b)


@pytest.mark.parametrize("alpha", [2.0, 0.5, -1.0])
def test_sub_alpha(alpha: float) -> None:
    a = torch.randn((32, 32), dtype=torch.bfloat16)
    b = torch.randn((32, 32), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x, y: torch.sub(x, y, alpha=alpha), a, b)


@pytest.mark.parametrize("shape", [(64, 128), (32, 32), (32, 64, 32)])
def test_mul(shape: tuple[int, ...]) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.mul, a, b)


@pytest.mark.parametrize("shape", [(64, 128), (32, 32), (32, 64, 32)])
def test_rsqrt(shape: tuple[int, ...]) -> None:
    a = torch.rand(shape, dtype=torch.bfloat16).add(0.1)
    assert_close_cpu_vs_tt(torch.rsqrt, a)


@pytest.mark.parametrize("keepdim", [True, False])
def test_mean_single_dim(keepdim: bool) -> None:
    # Loose tolerance: mean over 128 bf16 elements accumulates O(sqrt(N)) rounding
    # error relative to CPU; near-zero outputs drive up relative error further.
    a = torch.randn((64, 128), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.mean(x, dim=1, keepdim=keepdim), a, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("keepdim", [True, False])
def test_mean_multi_dim(keepdim: bool) -> None:
    a = torch.randn((32, 64, 32), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.mean(x, dim=[1, 2], keepdim=keepdim), a, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [1, -1, [1, 2], [-1, -2]], ids=["dim1", "dim_neg1", "dim12", "dim_neg12"])
def test_sum(dim: int | list[int], keepdim: bool) -> None:
    a = torch.randn((32, 64, 32), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x: torch.sum(x, dim=dim, keepdim=keepdim), a, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [1, -1, [1, 2], None], ids=["dim1", "dim_neg1", "dim12", "all_dims"])
def test_linalg_vector_norm(dim: int | list[int] | None, keepdim: bool) -> None:
    a = torch.randn((32, 64, 32), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(
        lambda x: torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim), a, atol=0.05, rtol=0.05
    )


@pytest.mark.parametrize("reduction", _REDUCTIONS)
def test_mse_loss(reduction: str) -> None:
    a = torch.randn((64, 128), dtype=torch.bfloat16)
    b = torch.randn((64, 128), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda x, y: F.mse_loss(x, y, reduction=reduction), a, b, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("reduction", _REDUCTIONS)
def test_mse_loss_backward(reduction: str) -> None:
    shape = (64, 128)
    self_t = torch.randn(shape, dtype=torch.bfloat16)
    target = torch.randn(shape, dtype=torch.bfloat16)
    # grad_output mirrors the loss shape: elementwise for 'none', scalar otherwise.
    grad_output = (
        torch.randn(shape, dtype=torch.bfloat16) if reduction == "none" else torch.randn((), dtype=torch.bfloat16)
    )

    def mse_loss_grad(s: torch.Tensor, t: torch.Tensor, go: torch.Tensor) -> torch.Tensor:
        s = s.detach().requires_grad_(True)
        F.mse_loss(s, t, reduction=reduction).backward(go)
        return s.grad

    assert_close_cpu_vs_tt(mse_loss_grad, self_t, target, grad_output, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("threshold", [0.0, 0.5, -0.25], ids=["thr0", "thr0.5", "thr_neg0.25"])
def test_threshold_backward(threshold: float) -> None:
    shape = (64, 128)
    grad_output = torch.randn(shape, dtype=torch.bfloat16)
    self_t = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(
        lambda go, s: torch.ops.aten.threshold_backward(go, s, threshold),
        grad_output,
        self_t,
    )


@pytest.mark.parametrize(
    "n,c_in,length,c_out,ksize,stride,padding,dilation,groups,bias",
    [
        (1, 32, 32, 64, 3, 1, 1, 1, 1, False),
        (1, 32, 32, 64, 3, 2, 1, 1, 1, False),
        (1, 32, 32, 64, 1, 1, 0, 1, 1, True),
        (1, 32, 32, 64, 3, 1, 1, 1, 4, False),
        (1, 32, 32, 64, 3, 1, 2, 2, 1, False),
        (1, 32, 64, 64, 5, 2, 2, 1, 1, False),
        (1, 32, 32, 64, 3, 1, 2, 2, 4, True),
        (2, 32, 32, 64, 3, 1, 1, 1, 1, True),
    ],
    ids=[
        "k3_s1_pad1_nobias",
        "k3_s2_pad1_nobias",
        "k1_bias",
        "grouped",
        "dilation2",
        "k5_s2_len64",
        "grouped_dilation_bias",
        "batch2_bias",
    ],
)
def test_conv1d(n: int, c_in: int, length: int, c_out: int, ksize: int, stride: int,
                padding: int, dilation: int, groups: int, bias: bool) -> None:
    x = torch.randn((n, c_in, length), dtype=torch.bfloat16)
    conv = torch.nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=ksize, stride=stride, padding=padding,
                           dilation=dilation, groups=groups, bias=bias).to(torch.bfloat16)
    assert_close_cpu_vs_tt(conv, x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "n,c_in,h,w,c_out,ksize,stride,padding,dilation,groups,bias",
    [
        (1, 32, 32, 32, 64, 3, 1, 1, 1, 1, False),
        (1, 32, 32, 32, 64, 3, 2, 1, 1, 1, False),
        (1, 32, 32, 32, 64, 1, 1, 0, 1, 1, True),
        # General configs: the NCHW Conv2dOp path (channel_dim=1) must handle grouped,
        # depthwise, dilated, asymmetric, and rectangular-kernel convs, not just resnet's.
        (1, 32, 32, 32, 64, 3, 1, 1, 1, 4, False),
        (1, 32, 32, 32, 32, 3, 1, 1, 1, 32, False),
        (1, 32, 32, 32, 64, 3, 1, 2, 2, 1, False),
        (1, 32, 48, 32, 64, 3, 1, 1, 1, 1, False),
        (1, 32, 32, 32, 64, (3, 5), (2, 1), (1, 2), 1, 1, False),
        (1, 32, 32, 32, 64, 3, 1, 2, 2, 4, True),
        (1, 16, 28, 28, 16, 3, 1, 1, 1, 16, True),
    ],
    ids=[
        "3x3_s1_nopad_nobias", "3x3_s2_pad1_nobias", "1x1_bias",
        "grouped", "depthwise", "dilation2", "asymmetric_hw",
        "rect_kernel_asym_stride", "grouped_dilation_bias", "depthwise_bias",
    ],
)
def test_conv2d(n: int, c_in: int, h: int, w: int, c_out: int, ksize: int | tuple[int, int],
                stride: int | tuple[int, int], padding: int | tuple[int, int], dilation: int,
                groups: int, bias: bool) -> None:
    x = torch.randn((n, c_in, h, w), dtype=torch.bfloat16)
    conv = torch.nn.Conv2d(c_in, c_out, ksize, stride=stride, padding=padding,
                           dilation=dilation, groups=groups, bias=bias).to(torch.bfloat16)
    assert_close_cpu_vs_tt(conv, x, atol=0.05, rtol=0.05)

# TODO(bmijanovicTT)
# groups: https://github.com/tenstorrent/tt-mlir/pull/9279
# dilation: https://github.com/tenstorrent/tt-mlir/issues/9280
@pytest.mark.parametrize(
    "n,c_in,d,h,w,c_out,ksize,stride,padding,bias",
    [
        (1, 32, 8, 16, 16, 64, 3, 1, 1, False),
        (1, 32, 8, 16, 16, 64, 3, 2, 1, False),
        (1, 32, 8, 16, 16, 64, 1, 1, 0, True),
        (2, 32, 4, 16, 16, 32, 3, 1, 1, True),
    ],
    ids=["k3_s1_pad1_nobias", "k3_s2_pad1_nobias", "k1_bias", "batch2_bias"],
)
def test_conv3d(n: int, c_in: int, d: int, h: int, w: int, c_out: int, ksize: int, stride: int,
                padding: int, bias: bool) -> None:
    x = torch.randn((n, c_in, d, h, w), dtype=torch.bfloat16)
    conv = torch.nn.Conv3d(c_in, c_out, ksize, stride=stride, padding=padding, bias=bias).to(torch.bfloat16)
    assert_close_cpu_vs_tt(conv, x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "n,c,h,w,k,stride,padding",
    [
        (1, 32, 64, 64, 3, 2, 1),
        (1, 64, 32, 32, 2, 2, 0),
    ],
    ids=["stride2_pad1", "stride2_nopad"],
)
def test_max_pool2d(n: int, c: int, h: int, w: int, k: int, stride: int, padding: int) -> None:
    x = torch.randn((n, c, h, w), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(
        lambda t: torch.nn.functional.max_pool2d(t, kernel_size=k, stride=stride, padding=padding),
        x,
    )


@pytest.mark.parametrize("n,c,h,w", [(32, 32, 32, 32)])
def test_batch_norm_inference(n: int, c: int, h: int, w: int) -> None:
    # nn.BatchNorm2d in eval mode dispatches to _native_batch_norm_legit_no_training.
    # Tile-aligned NCHW only — same constraint as the rest of the elementwise suite.
    model = torch.nn.BatchNorm2d(c).eval().to(torch.bfloat16)
    x = torch.randn((n, c, h, w), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(model, x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("affine", [True, False])
def test_layer_norm_inference(affine: bool) -> None:
    model = torch.nn.LayerNorm(64, elementwise_affine=affine).eval().to(torch.bfloat16)
    model.requires_grad_(False)
    x = torch.randn((2, 32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(model, x, atol=0.05, rtol=0.05)


def test_native_layer_norm_stats_eager() -> None:
    # mean/rstd are what native_layer_norm_backward consumes, so shapes and values
    # must match aten. Eager only: the compile path returns fp32 stats by contract,
    # so a raw dtype comparison against aten would (correctly) disagree.
    def stats(x):
        _, mean, rstd = torch.ops.aten.native_layer_norm(x, (64,), None, None, 1e-5)
        return torch.cat([mean, rstd], dim=-1)

    x = torch.randn((2, 32, 64), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(stats, x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("shape", [(64, 128), (32, 64, 32)])
def test_where(shape: tuple[int, ...]) -> None:
    condition = torch.randn(shape, dtype=torch.bfloat16) > 0
    x = torch.randn(shape, dtype=torch.bfloat16)
    y = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.where, condition, x, y)


@pytest.mark.parametrize("shape", [(64, 128), (32, 64, 32)])
def test_isneginf(shape: tuple[int, ...]) -> None:
    x = torch.randn(shape, dtype=torch.bfloat16)
    x.view(-1)[0] = float('-inf')
    x.view(-1)[1] = float('inf')
    assert_close_cpu_vs_tt(torch.isneginf, x)


@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [0, 1, -1], ids=["dim0", "dim1", "dim_neg1"])
def test_all(dim: int, keepdim: bool) -> None:
    x = torch.randn((64, 128), dtype=torch.bfloat16) > 0
    assert_close_cpu_vs_tt(lambda t: torch.all(t, dim=dim, keepdim=keepdim), x)


def _scattered_bool(shape: tuple[int, ...]) -> torch.Tensor:
    """Bool tensor with True only in the even rows of column 0. Reducing it with
    `any` over either dim gives a mix of True and False — a dense random mask
    would answer True everywhere and hide a broken reduction.
    """
    x = torch.zeros(shape, dtype=torch.bool)
    x[::2, 0] = True
    return x


@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [0, 1, -1], ids=["dim0", "dim1", "dim_neg1"])
def test_any_dim(dim: int, keepdim: bool) -> None:
    x = _scattered_bool((64, 128))
    assert_close_cpu_vs_tt(lambda t: torch.any(t, dim=dim, keepdim=keepdim), x)


@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [[0], [0, 1], [-1]], ids=["one", "both", "neg"])
def test_any_dims(dim: list[int], keepdim: bool) -> None:
    x = _scattered_bool((64, 128))
    assert_close_cpu_vs_tt(lambda t: torch.any(t, dim=dim, keepdim=keepdim), x)


# Reduction over every element, so the result is a rank-0 Bool scalar. Both
# outcomes are covered: an all-False input must not come back True.
@pytest.mark.parametrize("any_true", [True, False], ids=["some_true", "all_false"])
def test_any_all(any_true: bool) -> None:
    x = _scattered_bool((64, 128)) if any_true else torch.zeros((64, 128), dtype=torch.bool)
    assert_close_cpu_vs_tt(torch.any, x)


# Non-Bool input: torch counts any nonzero element as true, which the reduction
# has to express as `!= 0` — a cast to i1 would truncate 0.5 to False.
@pytest.mark.parametrize("dim", [1, -1], ids=["dim1", "dim_neg1"])
def test_any_nonbool(dim: int) -> None:
    x = torch.zeros((64, 128), dtype=torch.bfloat16)
    x[::2, 0] = 0.5
    assert_close_cpu_vs_tt(lambda t: torch.any(t, dim=dim), x)


# in-place relu_: the result aliases self, so assert correctness AND that the
# returned tensor is the same object whose storage was mutated.
@pytest.mark.parametrize("shape", [(64, 128), (32, 32), (32, 64, 32)])
def test_relu_inplace(shape: tuple[int, ...]) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    expected = torch.relu(a)
    tt = a.to("tt")
    with strict_no_fallback():
        ret = tt.relu_()
    assert ret is tt, "relu_ must return self"
    torch.testing.assert_close(tt.cpu(), expected)


@pytest.mark.parametrize("shape", [(64, 128), (32, 32), (32, 64, 32)])
def test_le_tensor(shape: tuple[int, ...]) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.le, a, b)


def test_le_tensor_equal_boundary() -> None:
    # `<=` (not `<`): equal operands must come back all-True, pinning the
    # boundary that distinguishes le from lt.
    a = torch.randn((32, 32), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.le, a, a.clone())


def test_le_tensor_out() -> None:
    # Drive the registered le.Tensor_out kernel directly (write_result_into path).
    a = torch.randn((32, 32), dtype=torch.bfloat16)
    b = torch.randn((32, 32), dtype=torch.bfloat16)
    expected = torch.le(a, b)
    out = torch.empty((32, 32), dtype=torch.bool, device="tt")
    with strict_no_fallback():
        ret = torch.le(a.to("tt"), b.to("tt"), out=out)
    assert ret is out, "le.Tensor_out must return the provided out tensor"
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize("shape", [(64, 128), (32, 32), (32, 64, 32)])
def test_gt_tensor(shape: tuple[int, ...]) -> None:
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.gt, a, b)


@pytest.mark.parametrize("shape", [(64, 128), (32, 32)])
def test_bitwise_and_bool(shape: tuple[int, ...]) -> None:
    # On Bool operands bitwise_and is logical AND — the form attention-mask
    # combination uses, and the only case the FPU kernel supports.
    a = torch.randn(shape, dtype=torch.bfloat16) > 0
    b = torch.randn(shape, dtype=torch.bfloat16) > 0
    assert_close_cpu_vs_tt(torch.bitwise_and, a, b)


# index_copy lowers to ttir.ScatterOp, which ttsim doesn't support (it aborts
# the simulator process) — exercise it on silicon only.
@pytest.mark.parametrize("dim", [0, 1, -1], ids=["dim0", "dim1", "dim_neg1"])
def test_index_copy(dim: int) -> None:
    self_t = torch.randn((32, 64), dtype=torch.bfloat16)
    index = torch.tensor([0, 2, 5])
    src_shape = list(self_t.shape)
    src_shape[dim] = index.numel()
    source = torch.randn(src_shape, dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(lambda s, i, src: torch.index_copy(s, dim, i, src), self_t, index, source)


def test_index_copy_inplace_kv_cache_like() -> None:
    # Mirrors the Llama StaticCache update: scatter new key/value slabs into a
    # [batch, heads, cache_len, head_dim] cache along the sequence dim.
    seq_dim = 2
    cache = torch.zeros((2, 4, 32, 16), dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 2])
    values = torch.randn((2, 4, positions.numel(), 16), dtype=torch.bfloat16)
    expected = cache.clone().index_copy_(seq_dim, positions, values)
    tt = cache.to("tt")
    with strict_no_fallback():
        ret = tt.index_copy_(seq_dim, positions.to("tt"), values.to("tt"))
    assert ret is tt, "index_copy_ must return self"
    torch.testing.assert_close(tt.cpu(), expected)


def test_index_copy_inplace_kv_cache_decode_step() -> None:
    # Single-token decode step at a non-zero position: the [batch, heads, 1,
    # head_dim] write along the seq dim lowers to ttir.update_cache (vs the
    # multi-token prefill above, which lowers to ttir.fill_cache). update_cache
    # honors the runtime position, so a non-zero index must land exactly.
    seq_dim = 2
    cache = torch.randn((2, 4, 32, 16), dtype=torch.bfloat16)
    positions = torch.tensor([7])
    values = torch.randn((2, 4, 1, 16), dtype=torch.bfloat16)
    expected = cache.clone().index_copy_(seq_dim, positions, values)
    tt = cache.to("tt")
    with strict_no_fallback():
        tt.index_copy_(seq_dim, positions.to("tt"), values.to("tt"))
    torch.testing.assert_close(tt.cpu(), expected)
