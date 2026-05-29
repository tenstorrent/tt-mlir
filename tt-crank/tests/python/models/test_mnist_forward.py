import pytest
import torch

from tt_kurbla.torch.testing import get_supported_dtypes, assert_close_cpu_vs_tt

from _models import MNISTLinear


@pytest.mark.parametrize("dtype", get_supported_dtypes(), ids=lambda d: str(d).removeprefix("torch."))
@pytest.mark.parametrize(
    "batch,feat,hidden,classes",
    [
        (32, 32 * 32, 128, 32),  # tile-aligned baseline
        (64, 28 * 28, 128, 10),  # real MNIST shapes
        (1, 28 * 28, 128, 10),   # batch=1 single-sample inference
    ],
    ids=["tile_aligned", "real_mnist_batch64", "real_mnist_batch1"],
)
def test_mnist_forward(
    batch: int, feat: int, hidden: int, classes: int, dtype: torch.dtype
) -> None:
    model = MNISTLinear(feat, hidden, classes).to(dtype)
    x = torch.randn(batch, feat, dtype=dtype)
    assert_close_cpu_vs_tt(model, x, atol=0.02, rtol=0.1)
