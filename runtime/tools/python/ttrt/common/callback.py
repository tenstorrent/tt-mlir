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


def golden():
    print("right now golden doesn't do anything")


def pdb():
    print("right now pdb doesn't do anything")
