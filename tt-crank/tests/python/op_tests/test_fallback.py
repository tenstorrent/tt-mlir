"""Coverage for the global CPU fallback registered in src/torch/ops/fallback.cpp.

The fallback is the catch-all that routes any aten op without a native
PrivateUse1 kernel through CPU.
"""

import pytest
import torch

from tt_kurbla.torch import _native
from tt_kurbla.torch.testing import strict_no_fallback


# torch.sigmoid is not natively registered on tt — exercises the fallback.
# If a future change adds a native sigmoid, swap this for another unimplemented
# op (e.g. torch.cos, torch.log) — the test is about the fallback mechanism,
# not sigmoid specifically.
_FALLBACK_OP = torch.sigmoid


def test_fallback_matches_cpu() -> None:
    src = torch.randn((32, 32), dtype=torch.bfloat16)
    expected = _FALLBACK_OP(src)
    got = _FALLBACK_OP(src.to("tt")).cpu()
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)


def test_fallback_strict_raises() -> None:
    src = torch.randn((32, 32), dtype=torch.bfloat16).to("tt")
    with strict_no_fallback():
        with pytest.raises(RuntimeError, match="strict-fallback"):
            _FALLBACK_OP(src)


def test_strict_mode_lets_native_ops_through() -> None:
    # add.Tensor is natively registered — strict mode should not affect it.
    a = torch.randn((32, 32), dtype=torch.bfloat16)
    b = torch.randn((32, 32), dtype=torch.bfloat16)
    expected = a + b
    with strict_no_fallback():
        got = (a.to("tt") + b.to("tt")).cpu()
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)
