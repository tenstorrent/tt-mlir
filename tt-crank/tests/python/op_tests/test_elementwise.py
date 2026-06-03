import pytest
import torch

from tt_kurbla.torch.testing import assert_close_cpu_vs_tt


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


@pytest.mark.parametrize("n,c,h,w", [(32, 32, 32, 32)])
def test_batch_norm_inference(n: int, c: int, h: int, w: int) -> None:
    # nn.BatchNorm2d in eval mode dispatches to _native_batch_norm_legit_no_training.
    # Tile-aligned NCHW only — same constraint as the rest of the elementwise suite.
    model = torch.nn.BatchNorm2d(c).eval().to(torch.bfloat16)
    x = torch.randn((n, c, h, w), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(model, x, atol=0.05, rtol=0.05)
