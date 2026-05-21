import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tt_kurbla.torch.testing import get_supported_dtypes, strict_no_fallback, assert_close_cpu_vs_tt


class MNISTLinear(nn.Module):
    def __init__(self, feat: int, hidden: int, classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feat, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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
def test_mnist_forward_raises_under_strict(
    batch: int, feat: int, hidden: int, classes: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    model_tt = MNISTLinear(feat, hidden, classes).to(dtype).to("tt")
    x_tt = torch.randn(batch, feat, dtype=dtype).to("tt")

    with pytest.raises(RuntimeError, match="strict-fallback"):
        assert_close_cpu_vs_tt(model_tt, x_tt)
