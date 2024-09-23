# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import importlib.machinery
import sys
import signal
import os
import io
import subprocess
import time
import socket
from pkg_resources import get_distribution
import shutil
import atexit
import pytest

TT_MLIR_HOME = os.environ.get("TT_MLIR_HOME", "")

BINARY_FILE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Silicon/TTNN/Output/simple_matmul.mlir.tmp.ttnn"
)
DIRECTORY_PATH = f"{TT_MLIR_HOME}/build/test/ttmlir/Silicon/TTNN"
SYSTEM_DESC_FILE_PATH = f"{TT_MLIR_HOME}/ttrt-artifacts/system_desc.ttsys"
SYSTEM_DESC_DIRECTORY_PATH = f"{TT_MLIR_HOME}/build/test/ttmlir/Silicon/TTNN"


def sub_process_command(test_command):
    result = subprocess.run(
        test_command, shell=True, capture_output=True, text=True, check=True
    )

    assert (
        result.returncode == 0
    ), f"subprocess command failed, test return result={result.returncode}"
