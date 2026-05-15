# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager
from typing import Iterator

from _ttmlir_runtime import runtime as tt_runtime
from _ttmlir_runtime.binary import Binary
from _ttmlir_runtime.runtime import CallbackContext, OpContext

from . import context
from .callbacks import CallbackPhase, run_op_callback
from .context import ChiselContext, _UNSET
from .op_configs import ChiselOpConfig
from .safety import chisel_safe
from .report import ChiselReport
from .utils import debug_wrap


@chisel_safe
@debug_wrap
def _pre_program_callback(
    rt_binary: Binary, rt_program_context: CallbackContext
) -> None:
    context.get_instance().preprogram(rt_binary, rt_program_context)


@chisel_safe
@debug_wrap
def _post_program_callback(
    rt_binary: Binary, rt_program_context: CallbackContext
) -> None:
    context.get_instance().postprogram(rt_binary, rt_program_context)


@chisel_safe
@debug_wrap
def _pre_op_callback(
    rt_binary: Binary,
    rt_program_context: CallbackContext,
    rt_op_context: OpContext,
) -> None:
    run_op_callback(
        rt_binary, rt_program_context, rt_op_context, phase=CallbackPhase.PRE
    )


@chisel_safe
@debug_wrap
def _post_op_callback(
    rt_binary: Binary,
    rt_program_context: CallbackContext,
    rt_op_context: OpContext,
) -> None:
    run_op_callback(
        rt_binary, rt_program_context, rt_op_context, phase=CallbackPhase.POST
    )


def bind() -> None:
    # Set the context before registering hooks: a hook firing on an unbound
    # context would log the raise via chisel_safe. Hooks register
    # synchronously, so this ordering is sufficient.
    if context.is_initialized():
        raise RuntimeError("chisel.bind() called twice without unbind()")
    context.set_current(ChiselContext())
    tt_runtime.DebugHooks.get(
        pre_op=_pre_op_callback,
        post_op=_post_op_callback,
        pre_program=_pre_program_callback,
        post_program=_post_program_callback,
    )


def configure(
    *,
    results_path=_UNSET,
    report_capacity=_UNSET,
    debug_chisel_dir=_UNSET,
    checks_config=_UNSET,
) -> None:
    context.get_instance().configure(
        results_path=results_path,
        report_capacity=report_capacity,
        debug_chisel_dir=debug_chisel_dir,
        checks_config=checks_config,
    )


def get_report() -> ChiselReport:
    return context.get_instance().report


def register_op_config(op_type: type, config: ChiselOpConfig) -> None:
    context.get_instance().register_op_config(op_type, config)


def unbind() -> None:
    # TODO(ndrakulic): test in tt-xla for race conditions due to asynchronous runtime execution.
    tt_runtime.unregister_hooks()
    if context.is_initialized():
        context.get_instance().close_results()
    context.set_current(None)


@contextmanager
def session(
    *,
    results_path=_UNSET,
    report_capacity=_UNSET,
    debug_chisel_dir=_UNSET,
    checks_config=_UNSET,
) -> Iterator[ChiselReport]:
    # Bind/configure/unbind in a `with` block; yields the in-memory report.
    bind()
    try:
        configure(
            results_path=results_path,
            report_capacity=report_capacity,
            debug_chisel_dir=debug_chisel_dir,
            checks_config=checks_config,
        )
        yield get_report()
    finally:
        unbind()
