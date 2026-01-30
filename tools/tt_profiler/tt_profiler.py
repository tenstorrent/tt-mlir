# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
import os
import sys
import signal
import subprocess
import shutil

from ._ttmlir_profiler import *

@contextmanager
def trace(log_dir: str):
    start_profiler(log_dir)

    try:
        yield
    finally:
        stop_profiler()
