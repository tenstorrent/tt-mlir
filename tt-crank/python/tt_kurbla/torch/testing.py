# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test-time helpers for the tt-kurbla torch backend.
"""

import enum
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.fx
from tt_kurbla.torch._compile import CompileOption

from . import _compile, _native


class DeviceType(enum.Enum):
    """Where tt-backend kernels actually run for this process. `SIM` routes
    through ttsim (TT_KURBLA_USE_SIMULATOR=1), `REAL` is silicon.
    """

    SIM = "sim"
    REAL = "real"


class ExecutionMode(enum.Enum):
    """How a callable runs on the tt backend.

    ``EAGER``: invoked directly through the dispatcher, one ATen kernel at a
    time. ``COMPILE``: wrapped with ``torch.compile(backend="tt")`` so the
    whole graph lowers into a single TTIR module before execution.
    """

    EAGER = "eager"
    COMPILE = "compile"


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
    mode: ExecutionMode = ExecutionMode.EAGER,
    options: dict[CompileOption, str | int | bool] | None = None,
) -> None:
    """Run ``fn`` on CPU and on tt with mirrored args; assert the outputs match.

    Mirrors each tensor arg to the ``"tt"`` device, runs ``fn`` on both sides,
    brings the tt result back to CPU, and compares with
    ``torch.testing.assert_close`` (which enforces ``tt_out.dtype ==
    cpu_out.dtype`` by default). Non-tensor args are passed through unchanged.

    If ``fn`` is an :class:`torch.nn.Module`, its parameters are moved to tt
    for the tt-side call.

    ``mode`` picks the tt execution path:

      - :attr:`ExecutionMode.EAGER` (default): invokes ``fn`` directly on
        tt-resident operands. When ``assert_native`` is true, wraps the call
        in :func:`strict_no_fallback` so any op that would silently route
        through CPU raises instead. This is how op tests enforce "this op is
        implemented natively on tt" by default — without it, a fallback-only
        path would still match CPU and pass spuriously. Pass
        ``assert_native=False`` to explicitly allow the fallback (e.g. when
        testing the fallback path itself).
      - :attr:`ExecutionMode.COMPILE`: wraps ``fn`` in an ``nn.Module``
        (unless it already is one), moves the module to tt, runs it through
        ``torch.compile(backend="tt", options=options)``, and compares. The
        compile path doesn't share the strict-fallback toggle — dynamo
        graph-breaks surface as logs/warnings, not as a captured exception
        from the eager dispatcher — so ``assert_native`` is ignored here.

    ``options`` is forwarded to ``torch.compile`` and applies only to ``COMPILE`` mode;
    it is ignored in ``EAGER`` mode.
    """
    # Detach to free the CPU forward's autograd graph before the move: with
    # swap_module_params_on_conversion enabled (tt_kurbla/torch/__init__.py),
    # `.to("tt")` refuses to swap parameters still referenced by a live
    # graph's SavedVariables. Only the output values are compared here, so the
    # graph is dead weight anyway.
    cpu_out = fn(*cpu_args)
    if isinstance(cpu_out, torch.Tensor):
        cpu_out = cpu_out.detach()

    if isinstance(fn, torch.nn.Module):
        fn.to("tt")
    tt_args = tuple(a.to("tt") if isinstance(a, torch.Tensor) else a for a in cpu_args)

    if mode is ExecutionMode.EAGER:
        if assert_native:
            with strict_no_fallback():
                tt_out = fn(*tt_args).cpu()
        else:
            tt_out = fn(*tt_args).cpu()
    elif mode is ExecutionMode.COMPILE:
        model = fn if isinstance(fn, torch.nn.Module) else _wrap_callable_as_module(fn)
        compiled = torch.compile(model.to("tt"), backend="tt", options=options)
        with torch.no_grad():
            tt_out = compiled(*tt_args).cpu()
    else:
        raise ValueError(f"unknown ExecutionMode: {mode!r}")

    torch.testing.assert_close(tt_out, cpu_out, atol=atol, rtol=rtol)


def _wrap_callable_as_module(fn: Callable[..., torch.Tensor]) -> torch.nn.Module:
    """Wrap a callable in a parameter-less ``nn.Module`` so it can be moved
    to a device with ``.to(...)`` before ``torch.compile``. The wrapper has
    no parameters or buffers, so the FX graph aot traces from it contains
    only the user's tensor inputs, no extra module-state placeholders.
    """

    class _FnModule(torch.nn.Module):
        def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            return fn(*args, **kwargs)

    return _FnModule()


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


@contextmanager
def post_aot_fx_hook(hook: Callable[[torch.fx.GraphModule], None]) -> Iterator[None]:
    """Install `hook` to receive each post-aot FX graph the compile backend lowers
    (forward and backward) for the duration of the context, then restore the previous
    hook. The compile itself proceeds unchanged, so callers can inspect the graph (e.g.
    assert an op stayed atomic) without patching backend internals.
    """
    prev = _compile._post_aot_fx_hook
    _compile._post_aot_fx_hook = hook
    try:
        yield
    finally:
        _compile._post_aot_fx_hook = prev
