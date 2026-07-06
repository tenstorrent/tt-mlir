"""Smoke tests for the `torch.compile(model, backend="tt")` path.

Phase-0 surface: `aten.add.Tensor` only - but with full semantics
(broadcasting, alpha scaling, dtype promotion), mirroring the eager kernel.
Each test builds a small `nn.Module` made entirely of additions, compiles
it through the tt dynamo backend, runs it on the tt device, and compares
the result against the eager CPU output.
"""

import pytest
import torch
import torch.nn as nn

from tt_kurbla.torch.testing import DeviceType, ExecutionMode, assert_close_cpu_vs_tt
from tt_kurbla.torch._compile import _compile_options, CompileOption, BfpDtype, MathFidelity
from _models import MNISTLinear


# Tile-aligned bf16 shapes only - same constraints as the eager elementwise
# tests: ttsim trips on TTNN-emitted f32 kernels, and the TTNN pipeline
# segfaults on non-tile-aligned small inputs (see
# tests/python/op_tests/test_elementwise.py).
_TILE_SHAPES: list[tuple[int, ...]] = [(32, 32), (64, 128), (32, 64, 32)]


def _assert_compile_matches_eager(
    model: nn.Module,
    *cpu_inputs: torch.Tensor,
    atol: float | None = None,
    rtol: float | None = None,
    options: dict [CompileOption, str | int | bool] | None = None,
) -> None:
    """Compile `model` with the tt dynamo backend and compare its output to
    eager-CPU.

    Pattern mirrors the remote phase-0 smoke test: take the eager-CPU output
    first while parameters are still on CPU, then `.to("tt")` both the
    model (so parameters become tt-resident graph inputs) and the user
    inputs before invoking the compiled callable.
    """
    with torch.no_grad():
        cpu_out = model(*cpu_inputs)

        model_tt = model.to("tt")
        compiled = torch.compile(model_tt, backend="tt", options=options)
        tt_inputs = tuple(t.to("tt") for t in cpu_inputs)
        tt_out = compiled(*tt_inputs).cpu()

    torch.testing.assert_close(tt_out, cpu_out, atol=atol, rtol=rtol)


class _ChainAdd(nn.Module):
    """`((a + b) + c) + a` - exercises operand reuse and a 3-deep add chain
    from a single FX graph."""

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t = a + b
        t = t + c
        return t + a


class _AddWithParam(nn.Module):
    """`x + bias` where `bias` is an nn.Parameter - the parameter crosses
    the aot boundary as a graph input alongside `x`, so the compile path
    sees a 2-arg function regardless of how the module advertises arity."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.randn(shape, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_single_add(shape: tuple[int, ...]) -> None:
    """The minimal compile graph: one add. Smoke-tests that the FX walker,
    builder, compile, and run plumbing all line up end-to-end."""
    class _Add(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Add(), a, b)


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_chain_add(shape: tuple[int, ...]) -> None:
    """Multiple stacked adds, with one input reused twice. The whole chain
    must lower into a single TTIR module - if anything falls out of the
    walker into an unhandled FX target, this will trip.

    Loosened tolerance: accumulated bf16 quantization across three adds
    drifts up to one ULP per add at the operands' magnitude (~0.05 absolute
    at randn() scale), which is well beyond the default bf16 atol of 1e-5.
    """
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    c = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_ChainAdd(), a, b, c, atol=0.1, rtol=0.05)


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_add_with_parameter(shape: tuple[int, ...]) -> None:
    """nn.Parameter as the second operand. aot lifts module parameters into
    the graph's argument list, so the runner closure receives them alongside
    the user's input."""
    model = _AddWithParam(shape)
    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(model, x)


@pytest.mark.parametrize("alpha", [2.0, 0.5, -1.0])
def test_compile_add_alpha(alpha: float) -> None:
    """`torch.add(a, b, alpha=k)` - the alpha scale must travel from the FX
    kwarg through the lowering into the same `scale_tensor` subgraph the
    eager kernel emits. Mirror of the eager `test_add_alpha`."""
    class _AddAlpha(nn.Module):
        def __init__(self, k: float) -> None:
            super().__init__()
            self.k = k

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.add(a, b, alpha=self.k)

    a = torch.randn((32, 32), dtype=torch.bfloat16)
    b = torch.randn((32, 32), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_AddAlpha(alpha), a, b)


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape",
    [
        # Row vector broadcasts across a matrix.
        ((32, 64), (1, 64)),
        # Column vector broadcasts across a matrix.
        ((32, 64), (32, 1)),
        # Bias-style broadcast: 3D activation + per-channel 1D bias.
        ((32, 64, 32), (32,)),
    ],
    ids=["row_bcast", "col_bcast", "channel_bcast"],
)
def test_compile_add_broadcast(lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> None:
    """Broadcasting matrices: lhs and rhs have different ranks/shapes, the
    compile lowering must compute the broadcasted result shape via
    `at::infer_size` - same path the eager kernel takes - instead of
    assuming shapes match."""
    class _Add(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

    a = torch.randn(lhs_shape, dtype=torch.bfloat16)
    b = torch.randn(rhs_shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Add(), a, b)


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_relu(shape: tuple[int, ...]) -> None:
    """Single aten::relu in a compiled graph."""
    class _ReLU(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_ReLU(), x)


@pytest.mark.parametrize("m,n", [(32, 64), (64, 32)])
def test_compile_t(m: int, n: int) -> None:
    """Single aten::t in a compiled graph."""
    class _T(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.t(x)

    x = torch.randn((m, n), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_T(), x)


@pytest.mark.parametrize(
    "m,k,n",
    [(32, 64, 32), (64, 64, 64)],
    ids=["32x64x32", "64x64x64"],
)
def test_compile_mm(m: int, k: int, n: int) -> None:
    """Single aten::mm in a compiled graph — exercises the FX lowering,
    TTIR MatmulOp emission, and runner round-trip for matrix multiply."""
    class _MM(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.mm(a, b)

    a = torch.randn((m, k), dtype=torch.bfloat16)
    b = torch.randn((k, n), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_MM(), a, b, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("beta,alpha", [(1.0, 1.0), (0.5, 2.0), (0.0, 1.0)], ids=["default", "scaled", "no_bias"])
def test_compile_addmm(beta: float, alpha: float) -> None:
    """aten::addmm with varying beta/alpha — covers the LinearOp fast path
    (beta==alpha==1), the scaled matmul+add path, and the zero-bias path."""
    class _AddMM(nn.Module):
        def __init__(self, b: float, a: float) -> None:
            super().__init__()
            self.b = b
            self.a = a

        def forward(self, bias: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return torch.addmm(bias, mat1, mat2, beta=self.b, alpha=self.a)

    bias = torch.randn((32, 32), dtype=torch.bfloat16)
    mat1 = torch.randn((32, 64), dtype=torch.bfloat16)
    mat2 = torch.randn((64, 32), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_AddMM(beta, alpha), bias, mat1, mat2, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_sub(shape: tuple[int, ...]) -> None:
    """Single aten::sub in a compiled graph."""
    class _Sub(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.sub(a, b)

    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Sub(), a, b)


@pytest.mark.parametrize("alpha", [2.0, 0.5, -1.0])
def test_compile_sub_alpha(alpha: float) -> None:
    """`torch.sub(a, b, alpha=k)` — the alpha scale must pass through the
    FX kwarg into the same scale_tensor subgraph the eager kernel emits."""
    class _SubAlpha(nn.Module):
        def __init__(self, k: float) -> None:
            super().__init__()
            self.k = k

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.sub(a, b, alpha=self.k)

    a = torch.randn((32, 32), dtype=torch.bfloat16)
    b = torch.randn((32, 32), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_SubAlpha(alpha), a, b)


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_mul(shape: tuple[int, ...]) -> None:
    """Single aten::mul in a compiled graph."""
    class _Mul(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.mul(a, b)

    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Mul(), a, b)


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_rsqrt(shape: tuple[int, ...]) -> None:
    """Single aten::rsqrt in a compiled graph. Positive inputs only."""
    class _Rsqrt(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.rsqrt(x)

    x = torch.rand(shape, dtype=torch.bfloat16).add(0.1)
    _assert_compile_matches_eager(_Rsqrt(), x, atol=0.01, rtol=0.01)


@pytest.mark.parametrize(
    "src_shape,dst_shape",
    [
        ((64, 128), (32, 256)),
        ((32, 64, 32), (32, 2048)),
        ((32, 32), (1024,)),
    ],
    ids=["2d_reshape", "3d_to_2d", "2d_to_1d"],
)
def test_compile_view(src_shape: tuple[int, ...], dst_shape: tuple[int, ...]) -> None:
    """aten::view in a compiled graph — exercises ReshapeOp emission and
    shape-attr construction for the FX lowering."""
    class _View(nn.Module):
        def __init__(self, shape: tuple[int, ...]) -> None:
            super().__init__()
            self.shape = shape

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.view(self.shape)

    x = torch.randn(src_shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_View(dst_shape), x)


@pytest.mark.parametrize(
    "shape,dim,keepdim",
    [
        ((64, 128), [1], False),
        ((64, 128), [1], True),
        ((32, 64, 32), [1, 2], False),
        ((32, 64, 32), [1, 2], True),
    ],
    ids=["2d_dim1", "2d_dim1_keepdim", "3d_dims12", "3d_dims12_keepdim"],
)
def test_compile_mean(shape: tuple[int, ...], dim: list[int], keepdim: bool) -> None:
    """aten::mean.dim in a compiled graph — exercises MeanOp with dim_arg and
    keep_dim attrs, covering the AdaptiveAvgPool2d decomposition pattern."""
    class _Mean(nn.Module):
        def __init__(self, d: list[int], k: bool) -> None:
            super().__init__()
            self.d = d
            self.k = k

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=self.d, keepdim=self.k)

    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Mean(dim, keepdim), x, atol=0.01, rtol=0.01)


@pytest.mark.parametrize(
    "batch,feat,hidden,classes",
    [(32, 32 * 32, 128, 32), (32, 28 * 28, 128, 10)],
    ids=["tile_aligned", "real_mnist"],
)
def test_compile_mnist(batch: int, feat: int, hidden: int, classes: int) -> None:
    """MNISTLinear (two fc layers + relu) compiled end-to-end — exercises
    aten.t, aten.addmm, and aten.relu in a single TTIR module."""
    model = MNISTLinear(feat, hidden, classes).to(torch.bfloat16)
    x = torch.randn(batch, feat, dtype=torch.bfloat16)
    _assert_compile_matches_eager(model, x, atol=0.05, rtol=0.1)


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
        "3x3_s1_nobias", "3x3_s2_nobias", "1x1_bias",
        "grouped", "depthwise", "dilation2", "asymmetric_hw",
        "rect_kernel_asym_stride", "grouped_dilation_bias", "depthwise_bias",
    ],
)
def test_compile_conv2d(n: int, c_in: int, h: int, w: int, c_out: int, ksize: int | tuple[int, int],
                        stride: int | tuple[int, int], padding: int | tuple[int, int], dilation: int,
                        groups: int, bias: bool) -> None:
    """aten::convolution in a compiled graph — exercises Conv2dOp with NCHW dim
    attrs and optional bias reshape across grouped/depthwise/dilated/rectangular configs."""
    model = nn.Conv2d(c_in, c_out, ksize, stride=stride, padding=padding,
                      dilation=dilation, groups=groups, bias=bias).to(torch.bfloat16)
    x = torch.randn((n, c_in, h, w), dtype=torch.bfloat16)
    _assert_compile_matches_eager(model, x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "n,c,h,w,k,stride,padding",
    [
        (1, 32, 64, 64, 3, 2, 1),
        (1, 64, 32, 32, 2, 2, 0),
    ],
    ids=["stride2_pad1", "stride2_nopad"],
)
def test_compile_max_pool2d(n: int, c: int, h: int, w: int, k: int, stride: int, padding: int) -> None:
    """aten::max_pool2d_with_indices in a compiled graph — exercises the NCHW→NHWC
    permute, MaxPool2dOp, and NHWC→NCHW permute path through the TTIR emitter."""
    class _MaxPool(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.max_pool2d(x, kernel_size=k, stride=stride, padding=padding)

    x = torch.randn((n, c, h, w), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_MaxPool(), x)


@pytest.mark.usefixtures("skip_if_sim")
def test_compile_batch_norm() -> None:
    """nn.BatchNorm2d in eval mode compiled end-to-end — exercises
    aten._native_batch_norm_legit_no_training + operator.getitem in one FX graph."""
    model = nn.BatchNorm2d(32).eval().to(torch.bfloat16)
    x = torch.randn((32, 32, 32, 32), dtype=torch.bfloat16)
    _assert_compile_matches_eager(model, x, atol=0.05, rtol=0.05)


@pytest.mark.usefixtures("skip_if_sim")
def test_compile_add_dtype_promotion() -> None:
    """bf16 + f32 must promote to f32 - same `at::promote_types` semantics
    the eager kernel applies. Validates that the compile path's MLIR-level
    promotion matches the torch-level promotion."""
    class _Add(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

    a = torch.randn((32, 32), dtype=torch.bfloat16)
    b = torch.randn((32, 32), dtype=torch.float32)
    _assert_compile_matches_eager(_Add(), a, b)


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_cos(shape: tuple[int, ...]) -> None:
    class _Cos(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.cos(x)

    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Cos(), x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_sin(shape: tuple[int, ...]) -> None:
    class _Sin(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sin(x)

    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Sin(), x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_neg(shape: tuple[int, ...]) -> None:
    class _Neg(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.neg(x)

    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Neg(), x)


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_silu(shape: tuple[int, ...]) -> None:
    class _SiLU(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.silu(x)

    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_SiLU(), x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_gelu(shape: tuple[int, ...]) -> None:
    # We emit the accurate gelu; a "tanh" request gets that same op (see lowering).
    class _GELU(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.gelu(x, approximate="tanh")

    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_GELU(), x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_div_tensor(shape: tuple[int, ...]) -> None:
    class _Div(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a / b

    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.rand(shape, dtype=torch.bfloat16).add(0.1)
    _assert_compile_matches_eager(_Div(), a, b, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("scalar", [2.0, 0.5, -1.0])
def test_compile_div_scalar(scalar: float) -> None:
    class _DivScalar(nn.Module):
        def __init__(self, s: float) -> None:
            super().__init__()
            self.s = s

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x / self.s

    x = torch.randn((32, 64), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_DivScalar(scalar), x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("scalar", [2.0, 0.5])
def test_compile_div_scalar_via_aten_op(scalar: float) -> None:
    """Mirrors test_compile_add_scalar_via_aten_op for div.Scalar."""
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.div.Scalar(x, scalar)

    x = torch.randn((32, 64), dtype=torch.bfloat16)
    compiled = torch.compile(f, backend="tt")
    with torch.no_grad():
        tt_out = compiled(x.to("tt")).cpu()
    torch.testing.assert_close(tt_out, f(x), atol=0.05, rtol=0.05)


@pytest.mark.parametrize("exp", [2.0, 0.5])
def test_compile_pow_tensor_scalar(exp: float) -> None:
    class _Pow(nn.Module):
        def __init__(self, e: float) -> None:
            super().__init__()
            self.e = e

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.pow(self.e)

    x = torch.rand((32, 64), dtype=torch.bfloat16).add(0.1)
    _assert_compile_matches_eager(_Pow(exp), x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("scalar", [2.0, -0.5])
def test_compile_add_scalar(scalar: float) -> None:
    class _AddScalar(nn.Module):
        def __init__(self, s: float) -> None:
            super().__init__()
            self.s = s

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.s

    x = torch.randn((32, 64), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_AddScalar(scalar), x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("scalar", [2.0, -0.5])
def test_compile_add_scalar_via_aten_op(scalar: float) -> None:
    """Explicitly calling `torch.ops.aten.add.Scalar` forces the `add.Scalar`
    lowering — Dynamo emits `add.Tensor` for plain `x + 2.0` and never hits
    this path. fullgraph=True prevents silent CPU fallback so any lowering bug
    surfaces as a real failure."""
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.add.Scalar(x, scalar)

    x = torch.randn((32, 64), dtype=torch.bfloat16)
    compiled = torch.compile(f, backend="tt")
    with torch.no_grad():
        tt_out = compiled(x.to("tt")).cpu()
    torch.testing.assert_close(tt_out, f(x), atol=0.05, rtol=0.05)


@pytest.mark.parametrize("scalar", [2.0, -0.5])
def test_compile_mul_scalar(scalar: float) -> None:
    class _MulScalar(nn.Module):
        def __init__(self, s: float) -> None:
            super().__init__()
            self.s = s

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.s

    x = torch.randn((32, 64), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_MulScalar(scalar), x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("scalar", [2.0, -0.5])
def test_compile_mul_scalar_via_aten_op(scalar: float) -> None:
    """Mirrors test_compile_add_scalar_via_aten_op for mul.Scalar."""
    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.mul.Scalar(x, scalar)

    x = torch.randn((32, 64), dtype=torch.bfloat16)
    compiled = torch.compile(f, backend="tt")
    with torch.no_grad():
        tt_out = compiled(x.to("tt")).cpu()
    torch.testing.assert_close(tt_out, f(x), atol=0.05, rtol=0.05)


@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_compile_softmax(dim: int) -> None:
    class _Softmax(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.d = d

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.softmax(x, dim=self.d)

    x = torch.randn((32, 64), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Softmax(dim), x, atol=0.05, rtol=0.05)


@pytest.mark.usefixtures("skip_if_sim")
@pytest.mark.parametrize("dim,keepdim", [(0, False), (1, True), (-1, False)])
def test_compile_argmax(dim: int, keepdim: bool) -> None:
    class _Argmax(nn.Module):
        def __init__(self, d: int, k: bool) -> None:
            super().__init__()
            self.d = d
            self.k = k

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.argmax(x, dim=self.d, keepdim=self.k)

    x = torch.randn((32, 64), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Argmax(dim, keepdim), x)


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_compile_unsqueeze(dim: int) -> None:
    class _Unsqueeze(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.d = d

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(self.d)

    x = torch.randn((32, 64), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Unsqueeze(dim), x)


def test_compile_squeeze() -> None:
    class _Squeeze(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(dim=1)

    x = torch.randn((32, 1, 64), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Squeeze(), x)


@pytest.mark.parametrize("dim0,dim1", [(0, 1), (1, 2), (0, 2)])
def test_compile_transpose(dim0: int, dim1: int) -> None:
    class _Transpose(nn.Module):
        def __init__(self, d0: int, d1: int) -> None:
            super().__init__()
            self.d0 = d0
            self.d1 = d1

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.transpose(self.d0, self.d1)

    x = torch.randn((32, 64, 32), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Transpose(dim0, dim1), x)


@pytest.mark.parametrize(
    "src_shape,target_shape",
    [((1, 64), (32, 64)), ((32, 1), (32, 64)), ((1, 1, 64), (32, 32, 64))],
)
def test_compile_expand(src_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> None:
    class _Expand(nn.Module):
        def __init__(self, shape: tuple[int, ...]) -> None:
            super().__init__()
            self.shape = shape

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.expand(self.shape)

    x = torch.randn(src_shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Expand(target_shape), x)


@pytest.mark.parametrize("shape,perm", [
    ((32, 64), (1, 0)),
    ((32, 64, 32), (2, 0, 1)),
    ((32, 64, 32), (0, 2, 1)),
])
def test_compile_permute(shape: tuple[int, ...], perm: tuple[int, ...]) -> None:
    class _Permute(nn.Module):
        def __init__(self, p: tuple[int, ...]) -> None:
            super().__init__()
            self.p = p

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.permute(self.p)

    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Permute(perm), x)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_compile_cat(dim: int) -> None:
    class _Cat(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.d = d

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.cat([a, b], dim=self.d)

    a = torch.randn((32, 64), dtype=torch.bfloat16)
    b = torch.randn((32, 64), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Cat(dim), a, b)


@pytest.mark.parametrize(
    "shape,dim,start,end",
    [
        ((32, 64), 1, 0, 32),
        ((32, 64), 0, 16, 32),
        ((32, 64, 32), -1, 0, 16),
    ],
)
def test_compile_slice(shape: tuple[int, ...], dim: int, start: int, end: int) -> None:
    class _Slice(nn.Module):
        def __init__(self, d: int, s: int, e: int) -> None:
            super().__init__()
            self.d = d
            self.s = s
            self.e = e

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.narrow(x, self.d, self.s, self.e - self.s)

    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Slice(dim, start, end), x)


@pytest.mark.usefixtures("skip_if_sim")
def test_compile_arange() -> None:
    class _Arange(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.arange(128, dtype=torch.float32)

    _assert_compile_matches_eager(_Arange())


@pytest.mark.usefixtures("skip_if_sim")
def test_compile_embedding() -> None:
    class _Embedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(256, 64).to(torch.bfloat16)

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            return self.emb(idx)

    idx = torch.randint(0, 256, (1, 128), dtype=torch.long)
    _assert_compile_matches_eager(_Embedding(), idx, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "b,m,k,n",
    [(2, 32, 64, 32), (4, 64, 32, 64)],
)
def test_compile_matmul_3d(b: int, m: int, k: int, n: int) -> None:
    class _Matmul(nn.Module):
        def forward(self, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, x)

    a = torch.randn((b, m, k), dtype=torch.bfloat16)
    x = torch.randn((b, k, n), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Matmul(), a, x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "b,h,s,d",
    [(1, 8, 32, 64), (1, 32, 32, 32)],
)
def test_compile_matmul_4d(b: int, h: int, s: int, d: int) -> None:
    """Attention QK^T pattern: [B,H,S,D] @ [B,H,D,S] -> [B,H,S,S]."""
    class _Matmul(nn.Module):
        def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
            return torch.matmul(q, k)

    q = torch.randn((b, h, s, d), dtype=torch.bfloat16)
    k = torch.randn((b, h, d, s), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Matmul(), q, k, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "b,m,k,n",
    [(2, 32, 64, 32), (4, 64, 32, 64)],
)
def test_compile_bmm(b: int, m: int, k: int, n: int) -> None:
    class _BMM(nn.Module):
        def forward(self, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return torch.bmm(a, x)

    a = torch.randn((b, m, k), dtype=torch.bfloat16)
    x = torch.randn((b, k, n), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_BMM(), a, x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "shape,dim,keepdim",
    [
        ((64, 128), [-1], False),
        ((64, 128), [0], True),
        ((32, 64, 32), [1, 2], False),
        ((32, 64, 32), [1, 2], True),
    ],
    ids=["2d_last", "2d_first_keepdim", "3d_last_two", "3d_last_two_keepdim"],
)
def test_compile_sum(shape: tuple[int, ...], dim: list[int], keepdim: bool) -> None:
    """aten::sum.dim_IntList in a compiled graph — exercises SumOp with dim
    and keep_dim attrs. Mirrors test_compile_mean."""
    class _Sum(nn.Module):
        def __init__(self, d: list[int], k: bool) -> None:
            super().__init__()
            self.d = d
            self.k = k

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.sum(dim=self.d, keepdim=self.k)

    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Sum(dim, keepdim), x, atol=0.05, rtol=0.05)


def test_compile_le() -> None:
    """aten::le.Tensor in a compiled graph — produces a Bool result tensor."""
    class _Le(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a <= b

    a = torch.randn((32, 64), dtype=torch.bfloat16)
    b = torch.randn((32, 64), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Le(), a, b)


def test_compile_where() -> None:
    """aten::where.self in a compiled graph — Bool mask selects between two
    float tensors. The mask is produced by aten::le so the graph exercises
    both lowerings end-to-end."""
    class _Where(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return torch.where(a <= b, b, c)

    a = torch.randn((32, 64), dtype=torch.bfloat16)
    b = torch.randn((32, 64), dtype=torch.bfloat16)
    c = torch.randn((32, 64), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Where(), a, b, c)


@pytest.mark.parametrize(
    "shape,diagonal",
    [
        ((32, 32), 0),
        ((32, 64), 0),
        ((32, 64), 2),
        ((32, 64), -2),
    ],
    ids=["square", "rect", "above_main", "below_main"],
)
def test_compile_tril(shape: tuple, diagonal: int) -> None:
    """aten::tril.default in a compiled graph — lower-triangular extraction with
    various shapes and diagonal offsets. Exercises build_tril including the
    arange/reshape/le/where subgraph it decomposes into."""
    class _Tril(nn.Module):
        def __init__(self, diagonal: int) -> None:
            super().__init__()
            self.diagonal = diagonal

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tril(x, diagonal=self.diagonal)

    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_Tril(diagonal), x)


@pytest.mark.usefixtures("skip_if_sim")
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "noncausal"])
def test_compile_sdpa(is_causal: bool) -> None:
    """scaled_dot_product_attention in a compiled graph — Dynamo dispatches to
    _scaled_dot_product_flash_attention_for_cpu on CPU which the tt lowering
    maps to build_sdpa. Tests both the causal (is_causal=True, no explicit mask)
    and non-causal (is_causal=False) variants."""
    class _SDPA(nn.Module):
        def __init__(self, causal: bool) -> None:
            super().__init__()
            self.causal = causal

        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=self.causal
            )

    q = torch.randn((1, 8, 32, 64), dtype=torch.bfloat16)
    k = torch.randn((1, 8, 32, 64), dtype=torch.bfloat16)
    v = torch.randn((1, 8, 32, 64), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_SDPA(is_causal), q, k, v, atol=0.05, rtol=0.05)


@pytest.mark.usefixtures("skip_if_sim")
def test_compile_index_copy() -> None:
    """aten::index_copy.default in a compiled graph — copies rows from `src`
    into `dst` at positions given by `index` along dim 0. Used by StaticCache
    to scatter KV-states into pre-allocated buffers."""
    class _IndexCopy(nn.Module):
        def forward(
            self, dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor
        ) -> torch.Tensor:
            return dst.index_copy(0, index, src)

    dst = torch.zeros((64, 32), dtype=torch.bfloat16)
    index = torch.arange(32, dtype=torch.int64)
    src = torch.randn((32, 32), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_IndexCopy(), dst, index, src)


def test_compile_threshold_backward() -> None:
    """aten::threshold_backward in a compiled graph — ReLU backward gate:
    grad_output * (self > threshold). Threshold 0 matches the relu backward."""
    class _ThresholdBwd(nn.Module):
        def forward(self, grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.threshold_backward.default(grad, x, 0.0)

    grad = torch.randn((32, 64), dtype=torch.bfloat16)
    x = torch.randn((32, 64), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_ThresholdBwd(), grad, x)


@pytest.mark.parametrize("reduction", [0, 1, 2], ids=["none", "mean", "sum"])
def test_compile_mse_loss(reduction: int) -> None:
    """aten::mse_loss in a compiled graph. reduction=0 (None) returns the
    elementwise squared error; 1 (Mean) and 2 (Sum) reduce to a scalar."""
    class _MSELoss(nn.Module):
        def __init__(self, r: int) -> None:
            super().__init__()
            self.r = r

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.mse_loss.default(pred, target, self.r)

    pred = torch.randn((32, 32), dtype=torch.bfloat16)
    target = torch.randn((32, 32), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_MSELoss(reduction), pred, target, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("reduction", [0, 1, 2], ids=["none", "mean", "sum"])
def test_compile_mse_loss_backward(reduction: int) -> None:
    """aten::mse_loss_backward in a compiled graph. For None reduction grad_output
    is the same shape as the inputs; for Mean/Sum it is a scalar [1] that
    broadcasts over the result."""
    class _MSELossBwd(nn.Module):
        def __init__(self, r: int) -> None:
            super().__init__()
            self.r = r

        def forward(
            self, grad: torch.Tensor, pred: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            return torch.ops.aten.mse_loss_backward.default(grad, pred, target, self.r)

    pred = torch.randn((32, 32), dtype=torch.bfloat16)
    target = torch.randn((32, 32), dtype=torch.bfloat16)
    grad = torch.randn((32, 32) if reduction == 0 else (1,), dtype=torch.bfloat16)
    _assert_compile_matches_eager(_MSELossBwd(reduction), grad, pred, target, atol=0.05, rtol=0.05)


def test_compile_options() -> None:
    """All in-process `_compile_options` parsing/validation in one test:
    optimization_level parsing, unset-option defaults, a full round-trip of every
    Category-A option, plain string keys, unknown-key rejection, and nanobind's
    native type enforcement."""
    # optimization_level parsing; an absent or empty dict defaults to 0.
    for options, expected in [(None, 0), ({}, 0), ({CompileOption.OPT_LEVEL: 1}, 1), ({CompileOption.OPT_LEVEL: 2}, 2)]:
        assert _compile_options(options).optimization_level == expected

    # Unset options stay None so the compiler keeps its own defaults; only
    # optimization_level (0) and experimental_enable_permute_matmul_fusion (True)
    # carry an explicit default.
    defaults = _compile_options(None)
    assert defaults.optimization_level == 0
    assert defaults.experimental_weight_dtype is None
    assert defaults.experimental_kv_cache_dtype is None
    assert defaults.math_fidelity is None
    assert defaults.fp32_dest_acc_en is None
    assert defaults.experimental_enable_permute_matmul_fusion is True
    assert defaults.enable_const_eval is None
    assert defaults.all_reduce_workaround_enabled is None

    # Every Category-A option round-trips into the native CompileOptions struct.
    c = _compile_options({
        CompileOption.OPT_LEVEL: 2,
        CompileOption.EXPERIMENTAL_WEIGHT_DTYPE: BfpDtype.BfpBf8,
        CompileOption.EXPERIMENTAL_KV_CACHE_DTYPE: BfpDtype.BfpBf4,
        CompileOption.MATH_FIDELITY: MathFidelity.HiFi3,
        CompileOption.FP32_DEST_ACC_EN: False,
        CompileOption.EXPERIMENTAL_ENABLE_FUSING_CONV2D_WITH_MULTIPLY_PATTERN: True,
        CompileOption.EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION: False,
        CompileOption.ENABLE_TRACE: True,
        CompileOption.ENABLE_CONST_EVAL: False,
        CompileOption.ENABLE_CONST_EVAL_ON_CPU: False,
        CompileOption.ENABLE_CONST_EVAL_INPUTS_TO_SYSTEM_MEMORY: False,
        CompileOption.EXPERIMENTAL_ENABLE_DRAM_SPACE_SAVING_OPTIMIZATION: True,
        CompileOption.ENABLE_CREATE_D2M_SUBGRAPHS: True,
        CompileOption.TTNN_PERF_METRICS_ENABLED: True,
        CompileOption.TTNN_PERF_METRICS_OUTPUT_FILE: "/tmp/perf.json",
        CompileOption.ALL_REDUCE_WORKAROUND_ENABLED: False,
    })
    assert c.optimization_level == 2
    assert c.experimental_weight_dtype == BfpDtype.BfpBf8
    assert c.experimental_kv_cache_dtype == BfpDtype.BfpBf4
    assert c.math_fidelity == MathFidelity.HiFi3
    assert c.fp32_dest_acc_en is False
    assert c.experimental_enable_fusing_conv2d_with_multiply_pattern is True
    assert c.experimental_enable_permute_matmul_fusion is False
    assert c.enable_trace is True
    assert c.enable_const_eval is False
    assert c.enable_const_eval_on_cpu is False
    assert c.enable_const_eval_inputs_to_system_memory is False
    assert c.experimental_enable_dram_space_saving_optimization is True
    assert c.enable_create_d2m_subgraphs is True
    assert c.ttnn_perf_metrics_enabled is True
    assert c.ttnn_perf_metrics_output_file == "/tmp/perf.json"
    assert c.all_reduce_workaround_enabled is False

    # Plain string keys are accepted alongside CompileOption members.
    string_keyed = _compile_options({"enable_trace": True, "math_fidelity": MathFidelity.LoFi})
    assert string_keyed.enable_trace is True
    assert string_keyed.math_fidelity == MathFidelity.LoFi

    # Unknown option keys are rejected.
    with pytest.raises(ValueError):
        _compile_options({"invalid": 0})

    # nanobind enforces the native field types: a wrong-typed value is rejected
    # when assigned into the CompileOptions struct.
    for bad_options in [
        {CompileOption.ENABLE_TRACE: "yes"},  # bool option, str value
        {CompileOption.FP32_DEST_ACC_EN: "true"},  # optional-bool option, str value
        {CompileOption.MATH_FIDELITY: "hifi4"},  # MathFidelity option, str value
        {CompileOption.EXPERIMENTAL_WEIGHT_DTYPE: "bfp_bf8"},  # BfpDtype option, str value
        {CompileOption.OPT_LEVEL: 1.0},  # int option, float value
    ]:
        with pytest.raises(TypeError):
            _compile_options(bad_options)



@pytest.mark.parametrize("shape", _TILE_SHAPES)
def test_compile_with_opt_level_0(shape: tuple[int, ...]) -> None:
    """This is just a sanity compile options API test.
    It tests different functions that we use is tests with compile options."""
    class _ReLU(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

    x = torch.randn(shape, dtype=torch.bfloat16)
    _assert_compile_matches_eager(_ReLU(), x, options={CompileOption.OPT_LEVEL: 0})
    assert_close_cpu_vs_tt(_ReLU(), x, mode=ExecutionMode.EAGER, options={CompileOption.OPT_LEVEL: 0})
    assert_close_cpu_vs_tt(_ReLU(), x, mode=ExecutionMode.COMPILE, options={CompileOption.OPT_LEVEL: 0})
