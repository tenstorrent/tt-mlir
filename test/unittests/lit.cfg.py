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
config.suffixes = [".cpp"]

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

# Propagate testlib env variables.
for var in ("TTMLIR_TEST_WORKFLOW", "TTMLIR_TEST_SEED"):
    if var in os.environ:
        config.environment[var] = os.environ[var]

if "TT_MLIR_HOME" in os.environ:
    config.environment["TT_MLIR_HOME"] = os.environ["TT_MLIR_HOME"]
else:
    raise OSError("TT_MLIR_HOME environment variable is not set")

if "TT_METAL_RUNTIME_ROOT" in os.environ:
    config.environment["TT_METAL_RUNTIME_ROOT"] = os.environ["TT_METAL_RUNTIME_ROOT"]
else:
    raise OSError("TT_METAL_RUNTIME_ROOT environment variable is not set")


# Some optimizer unittests must be run serially. There is no way to that in llvm-lit
# without running all tests serially which will take a long time. Exclude them and
# run them in CI separately.
config.excludes.add("Optimizer")
config.excludes.add("Validation")
