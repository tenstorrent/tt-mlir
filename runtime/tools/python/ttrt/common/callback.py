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

from ttrt.common.util import *

GOLDENS = []
curr_golden_index = -1


def add_global_golden(golden_tensor):
    global GOLDENS
    GOLDENS.append(golden_tensor.get_torch_tensor())


def get_next_golden():
    global GOLDENS, curr_golden_index
    curr_golden_index += 1
    return curr_golden_index


def golden():
    print("right now golden doesn't do anything")

    golden_index = get_next_golden()
    golden_torch_tensor = GOLDENS[golden_index]
    print(golden_torch_tensor)


def pdb():
    print("right now pdb doesn't do anything")
