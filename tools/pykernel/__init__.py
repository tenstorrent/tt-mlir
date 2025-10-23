# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from importlib import import_module

from pykernel._src.kernel_types import CircularBuffer, Kernel, CompileTimeValue

from pykernel.api import (
    ttkernel_compile,
    compute_thread,
    reader_thread,
    writer_thread,
    ttkernel_tensix_compile,
    ttkernel_noc_compile,
)

import pykernel.d2m_api as d2m_api


# Configure logging
def _configure_logging():
    """Configure logging for the pykernel package based on environment variables."""
    # Add custom TRACE level
    TRACE_LEVEL = logging.DEBUG - 5
    logging.addLevelName(TRACE_LEVEL, "TRACE")

    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(TRACE_LEVEL):
            self._log(TRACE_LEVEL, message, args, **kwargs)

    logging.Logger.trace = trace

    # Make TRACE_LEVEL available as logging.TRACE
    logging.TRACE = TRACE_LEVEL

    # Check for PYKERNEL_LOG_LEVEL environment variable
    log_level = os.environ.get("PYKERNEL_LOG_LEVEL", "INFO").upper()

    # Map string levels to logging constants
    level_map = {
        "TRACE": TRACE_LEVEL,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    level = level_map.get(log_level, logging.INFO)

    # Configure the root logger for pykernel
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set specific logger levels
    loggers = [
        "pykernel._src.d2m_ast",
        "pykernel.d2m_api",
        "pykernel._src.utils",
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)


# Initialize logging
_configure_logging()

# Hide ttnn import behind a lazy import for now.
# `import pykernel` will not import ttnn, but `from pykernel import PykernelOp` will
_lazy = {"PyKernelOp": "pykernel._src.kernel_op"}


def __getattr__(name):
    if name in _lazy:
        mod = import_module(_lazy[name])
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(name)


__all__ = [
    "ttkernel_compile",
    "compute_thread",
    "reader_thread",
    "writer_thread",
    "ttkernel_tensix_compile",
    "ttkernel_noc_compile",
    "CircularBuffer",
    "Kernel",
    "CompileTimeValue",
    "PyKernelOp",
]
