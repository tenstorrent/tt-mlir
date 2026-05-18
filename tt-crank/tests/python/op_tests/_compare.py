"""CPU-vs-TT numerical comparison helper.

The architecture doc calls out a "simple test infra for testing single pytorch
ops (TT vs. CPU)" — this is that. Each op test computes ``fn(*cpu_args)`` on
CPU, mirrors the args to ``"tt"``, computes ``fn(*tt_args)``, brings the
result back, and asserts numerical equivalence within tolerance.
"""

from collections.abc import Callable
from typing import Any

import torch


def assert_close_cpu_vs_tt(
    fn: Callable[..., torch.Tensor],
    *cpu_args: Any,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    cpu_out = fn(*cpu_args)

    tt_args = tuple(a.to("tt") if isinstance(a, torch.Tensor) else a for a in cpu_args)
    tt_out = fn(*tt_args).cpu()

    torch.testing.assert_close(tt_out, cpu_out, atol=atol, rtol=rtol)
