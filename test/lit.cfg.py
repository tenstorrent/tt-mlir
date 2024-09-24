# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# -*- Python -*-

import os
import sys
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool


# Configuration file for the 'lit' test runner.


def set_system_desc_features(system_desc):
    config.available_features.add(system_desc["chip_descs"][0]["arch"])
    if len(system_desc["chip_desc_indices"]) > 1:
        config.available_features.add("multi-chip")


# name: The name of this test suite.
config.name = "TTMLIR"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# Stablehlo tests can be optionally enabled.
if config.enable_stablehlo:
    config.available_features.add("stablehlo")

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.ttmlir_obj_root, "test")

# system_desc_path: The system desc that is to be used to generate the binary files.
config.system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")

if config.system_desc_path:
    try:
        import ttrt

        system_desc = ttrt.binary.as_dict(
            ttrt.binary.load_system_desc_from_path(config.system_desc_path)
        )["system_desc"]
        set_system_desc_features(system_desc)
    except ImportError:
        print(
            "ttrt not found. Please install ttrt to use have system desc driven test requirements set.",
            file=sys.stderr,
        )

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%system_desc_path%", config.system_desc_path))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.ttmlir_obj_root, "test")
config.ttmlir_tools_dir = os.path.join(config.ttmlir_obj_root, "bin")
config.ttmlir_libs_dir = os.path.join(config.ttmlir_obj_root, "lib")

config.substitutions.append(("%ttmlir_libs", config.ttmlir_libs_dir))

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.ttmlir_tools_dir, config.llvm_tools_dir]
tools = ["mlir-opt", "ttmlir-opt", "ttmlir-translate"]

llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(config.mlir_obj_dir, "python_packages"),
    ],
    append_path=True,
)

# add test related parameters
test_arch = os.getenv("TEST_ARCH", "").split(",")
test_pipeline = os.getenv("TEST_PIPELINE", "").split(",")
test_target_family = os.getenv("TEST_TARGET_FAMILY", "").split(",")
test_target_silicon = os.getenv("TEST_TARGET_SILICON", "").split(",")
test_duration = os.getenv("TEST_DURATION", "").split(",")

if "wormhole_b0" in test_arch:
    config.available_features.add("wormhole_b0")

if "functional" in test_pipeline:
    config.available_features.add("functional")

if "perf" in test_pipeline:
    config.available_features.add("perf")

if "functional" in test_pipeline or "perf" in test_pipeline:
    config.available_features.add("functional,perf")

if "n150" in test_target_family:
    config.available_features.add("n150")

if "n300" in test_target_family:
    config.available_features.add("n300")

if "n150" in test_target_family or "n300" in test_target_family:
    config.available_features.add("n150,n300")

if "push" in test_duration:
    config.available_features.add("push")
