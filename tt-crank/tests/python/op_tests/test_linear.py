import pytest
import torch

from tt_kurbla.torch.testing import assert_close_cpu_vs_tt


@pytest.mark.parametrize("m,n", [(32, 64), (64, 32), (32, 32)])
def test_t(m: int, n: int) -> None:
    a = torch.randn((m, n), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.t, a)


@pytest.mark.parametrize(
    "m,k,n",
    [(32, 64, 32), (64, 128, 64), (32, 32, 32)],
)
def test_mm(m: int, k: int, n: int) -> None:
    a = torch.randn((m, k), dtype=torch.bfloat16)
    b = torch.randn((k, n), dtype=torch.bfloat16)
    assert_close_cpu_vs_tt(torch.mm, a, b, atol=0.05, rtol=0.05)


@pytest.mark.parametrize(
    "m,k,n",
    [(32, 64, 32), (64, 128, 64), (32, 32, 32)],
)
@pytest.mark.parametrize(
    "beta,alpha",
    [(1.0, 1.0), (2.0, 1.0), (1.0, 0.5), (0.0, 1.0)],
    ids=["default", "beta2", "alpha05", "beta0"],
)
def test_addmm(m: int, k: int, n: int, beta: float, alpha: float) -> None:
    bias = torch.randn((n,), dtype=torch.bfloat16)
    mat1 = torch.randn((m, k), dtype=torch.bfloat16)
    mat2 = torch.randn((k, n), dtype=torch.bfloat16)
    fn = lambda b, m1, m2: torch.addmm(b, m1, m2, beta=beta, alpha=alpha)
    # beta=2 scales the bias before the BF16 add, doubling the rounding budget;
    # atol reflects the observed worst-case for beta=2 at K=128.
    assert_close_cpu_vs_tt(fn, bias, mat1, mat2, atol=0.22, rtol=0.1)
