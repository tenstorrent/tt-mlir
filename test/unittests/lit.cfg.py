# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import subprocess

import lit.formats

# name: The name of this test suite.
config.name = "TTMLIR-Unit"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# is_early; Request to run this suite early.
config.is_early = True

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, "test/unittests")
config.test_source_root = config.test_exec_root

# testFormat: The test format to use to interpret test
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")

# Propagate the temp directory. Windows requires this because it uses \Windows\
# if none of these are present.
if "TMP" in os.environ:
    config.environment["TMP"] = os.environ["TMP"]
if "TEMP" in os.environ:
    config.environment["TEMP"] = os.environ["TEMP"]

# Propagate HOME as it can be used to override incorrect homedir in passwd
# that causes the tests to fail.
if "HOME" in os.environ:
    config.environment["HOME"] = os.environ["HOME"]


if "TT_METAL_HOME" in os.environ:
    print(f"{os.environ['TT_METAL_HOME']}")
    config.environment["TT_METAL_HOME"] = os.environ["TT_METAL_HOME"]
else:
    print("Error: TT_METAL_HOME not set")

if "ARCH_NAME" in os.environ:
    print(f"ARCH_NAME={os.environ['ARCH_NAME']}")
    config.environment["ARCH_NAME"] = os.environ["ARCH_NAME"]
else:
    print("ARCH_NAME not set. Defaulting to wormhole")
    config.environment["ARCH_NAME"] = os.environ["wormhole_b0"]
