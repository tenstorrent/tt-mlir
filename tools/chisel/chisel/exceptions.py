# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Typed chisel exceptions and the unified record-on-failure context manager.

Utilities raise a specific ChiselError subclass so callback bodies don't need
to care about the original exception type. The `record_check` context manager
translates those into per-slot records against a ChiselChecker.
"""
import contextlib
import logging
import traceback
from typing import Iterable, Optional

logger = logging.getLogger("chisel")


class ChiselError(Exception):
    """Base for all chisel-internal operational failures."""


class SkippableChiselError(ChiselError):
    """Failures that should be recorded as 'skipped' rather than 'error'."""


class NoGoldenImplementation(SkippableChiselError):
    """No golden mapping registered for this op type."""


class GoldenInputMissing(SkippableChiselError):
    """An input required by the golden function is not present in the pool."""


class GoldenExecutionError(ChiselError):
    """The golden function itself raised during execution."""


class TensorRetrievalError(ChiselError):
    """retrieve_torch_tensor failed to pull a tensor from the runtime pool."""


class TensorWriteError(ChiselError):
    """write_torch_tensor_to_pool failed to update a tensor in the runtime pool."""


@contextlib.contextmanager
def record_check(
    slots: Iterable[str],
    check_label: str,
    checker,
    *,
    log_prefix: str,
    success: Optional[str] = None,
    record: bool = True,
):
    """Translate a ChiselError from the guarded block into per-slot records.

    - SkippableChiselError → "skipped" record on every slot.
    - Other ChiselError → "error" record with traceback on every slot.
    - Clean exit: if `success` is set, record that status on every slot.
    - record=False → log but emit no records (skip-mode isolation golden).
    - Non-ChiselError exceptions propagate; the @chisel_safe top-level wrapper
      is expected to catch those as "bug in chisel".
    """
    try:
        yield
    except SkippableChiselError as e:
        logger.debug(f"{log_prefix}: {check_label} skipped ({type(e).__name__}: {e})")
        if record:
            for slot in slots:
                checker.record(slot, check_label, "skipped")
        return
    except ChiselError:
        tb = traceback.format_exc()
        logger.error(f"{log_prefix}: {check_label} error\n{tb}")
        if record:
            for slot in slots:
                checker.record(slot, check_label, "error", traceback=tb)
        return
    if success is not None and record:
        for slot in slots:
            checker.record(slot, check_label, success)
