# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import functools
import logging
import traceback

from .exceptions import ChiselFailure
from .report import ChiselBugPayload, ChiselRecord
from .utils import get_op_asm

logger = logging.getLogger("chisel")


def chisel_safe(fn):
    # Swallows exceptions so a chisel bug never kills the ttmlir runtime.
    # ChiselFailure -> structured check record; anything else -> chisel_bug
    # record with traceback. Returns True on clean run, False if a failure
    # was caught - callers use this to gate dependent work.

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> bool:
        # Local import: avoid context <-> safety import cycle.
        from . import context

        try:
            fn(*args, **kwargs)
            return True
        except ChiselFailure as e:
            logger.warning(str(e))
            try:
                context.get_instance().write_record(e.to_record())
            except Exception:
                logger.exception("chisel_safe failed to record check failure")
        except Exception:
            tb = traceback.format_exc()
            try:
                ctx = context.get_instance()
                if ctx.is_in_op_scope:
                    op = ctx.op
                    ctx.write_record(
                        ChiselRecord(
                            op=op.name,
                            check=fn.__name__,
                            op_asm=get_op_asm(op),
                            payload=ChiselBugPayload(traceback=tb),
                        )
                    )
                else:
                    logger.exception(
                        "chisel callback %s crashed outside op scope", fn.__name__
                    )
            except Exception:
                logger.exception("chisel_safe failed to record chisel_bug")
        return False

    return wrapper
