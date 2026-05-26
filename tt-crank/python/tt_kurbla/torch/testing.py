"""
Test-time helpers for the tt-kurbla torch backend.
"""

import enum
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from . import _native


class DeviceType(enum.Enum):
    """Where tt-backend kernels actually run for this process. `SIM` routes
    through ttsim (TT_KURBLA_USE_SIMULATOR=1), `REAL` is silicon.
    """

    SIM = "sim"
    REAL = "real"


@contextmanager
def strict_no_fallback() -> Iterator[None]:
    """Make the global CPU fallback raise instead of running, for the body.

    Use to assert that wrapped code stays on native tt kernels — any op that
    would otherwise silently route through CPU raises a RuntimeError with the
    op name. Restores the previous strict-mode state on exit even if the body
    raises.
    """
    previous = _native.fallback_strict()
    _native.set_fallback_strict(True)
    try:
        yield
    finally:
        _native.set_fallback_strict(previous)


def assert_close_cpu_vs_tt(
    fn: Callable[..., torch.Tensor],
    *cpu_args: Any,
    atol: float | None = None,
    rtol: float | None = None,
    assert_native: bool = True,
) -> None:
    """Run ``fn`` on CPU and on tt with mirrored args; assert the outputs match.

    Mirrors each tensor arg to the ``"tt"`` device, runs ``fn`` on both sides,
    brings the tt result back to CPU, and compares with
    ``torch.testing.assert_close``. Non-tensor args are passed through unchanged.

    If ``fn`` is an :class:`torch.nn.Module`, its parameters are moved to tt
    for the tt-side call and restored to CPU afterward, so the caller's device
    state is not mutated.

    Defaults to ``assert_native=True``: the tt-side call runs inside
    :func:`strict_no_fallback`, so any op that would silently fall back to CPU
    raises instead. This is how op tests enforce "this op is implemented
    natively on tt" by default — without it, a fallback-only path would still
    match CPU and pass spuriously. Pass ``assert_native=False`` to explicitly
    allow the fallback (e.g. when testing the fallback path itself).
    """
    cpu_out = fn(*cpu_args)

    if isinstance(fn, torch.nn.Module): fn.to("tt")
    tt_args = tuple(a.to("tt") if isinstance(a, torch.Tensor) else a for a in cpu_args)

    if assert_native:
        with strict_no_fallback():
            tt_out = fn(*tt_args).cpu()
    else:
        tt_out = fn(*tt_args).cpu()

    torch.testing.assert_close(tt_out, cpu_out, atol=atol, rtol=rtol)


def get_supported_dtypes() -> list[torch.dtype]:
    """Return the torch dtypes the tt backend can execute in the current
    process. Intended for ``pytest.mark.parametrize("dtype", ...)`` so a test
    suite only collects cases that can actually run.

    Currently:
      - Real silicon: bf16 and f32.
      - ttsim (TT_KURBLA_USE_SIMULATOR=1): bf16 only. TTNN-emitted f32 kernels
        trip ``tensix_execute_unpacr: in_data_format=0`` UB on the simulator
        (unpacker config registers come up zeroed and ttsim flags that as
        undefined). Same caveat documented in
        ``tests/engine_execution_payload_test.cpp``.
    """
    if os.environ.get("TT_KURBLA_USE_SIMULATOR") == "1":
        return [torch.bfloat16]
    return [torch.bfloat16, torch.float32]
