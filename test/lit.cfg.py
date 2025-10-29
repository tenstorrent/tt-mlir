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


class OpModelAwareShTest(lit.formats.ShTest):
    """Custom test format that assigns parallelism groups based on test requirements."""

    def __init__(self, execute_external):
        super().__init__(execute_external)

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        """Override to set parallelism group based on test requirements."""
        for test in super().getTestsInDirectory(
            testSuite, path_in_suite, litConfig, localConfig
        ):
            # Read the test file to check for REQUIRES directives
            try:
                with open(test.getSourcePath(), "r") as f:
                    # Read first few lines to find REQUIRES directives
                    for line_num, line in enumerate(f):
                        # Stop searching after reasonable number of lines
                        if line_num > 10:
                            break

                        # Look for REQUIRES directives that include opmodel
                        if line.strip().startswith("// REQUIRES:"):
                            requires_content = line.strip()[
                                12:
                            ].strip()  # Remove '// REQUIRES:'
                            # Split by comma to handle multiple requirements
                            requirements = [
                                req.strip() for req in requires_content.split(",")
                            ]
                            if "opmodel" in requirements:
                                # Assign opmodel tests to single-threaded group
                                test.config.parallelism_group = "opmodel"
                                break
                    # If not opmodel test, it will use default parallelism (multithreaded)
            except (IOError, OSError):
                # If we can't read the file, default to regular parallelism
                pass
            yield test


# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "TTMLIR"

config.test_format = OpModelAwareShTest(not llvm_config.use_lit_shell)

# Stablehlo tests can be optionally enabled.
if config.enable_stablehlo:
    config.available_features.add("stablehlo")

# Pykernel tests are optionally enabled.
if config.enable_pykernel:
    config.available_features.add("pykernel")

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.ttmlir_obj_root, "test")

# When op models (constraints) are enabled all optimizer enabled tests open the physical device so they are not parallelizable.
# TODO(odjuricic): This can be removed once https://github.com/tenstorrent/tt-metal/issues/14000 is implemented.
# When op models (constraints) are enabled, set up parallelism groups.
# OpModel tests will be assigned to single-threaded group by OpModelAwareShTest,
# while other tests will use default parallelism (multithreaded).
if config.enable_opmodel:
    # Set up single-threaded parallelism group for opmodel tests
    lit_config.parallelism_groups["opmodel"] = 1
    config.available_features.add("opmodel")

# Optimizer models performance tests are optionally enabled for specific CI jobs via lit parameter.
if lit_config.params.get("TTMLIR_ENABLE_OPTIMIZER_MODELS_PERF_TESTS", "") == "1":
    config.available_features.add("perf")

if os.path.exists("/opt/tenstorrent/sfpi/compiler/bin/riscv32-tt-elf-g++"):
    config.available_features.add("sfpi")

# system_desc_path: The system desc that is to be used to generate the binary files.
config.system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")

# set features based on system
system_desc = None
if config.system_desc_path:
    try:
        import ttrt

        system_desc = ttrt.binary.fbb_as_dict(
            ttrt.binary.load_system_desc_from_path(config.system_desc_path)
        )["system_desc"]
        config.available_features.add(system_desc["chip_descs"][0]["arch"])
    except ImportError:
        print(
            "ttrt not found. Please install ttrt to use have system desc driven test requirements set.",
            file=sys.stderr,
        )


# set targets based on the system (default = n150)
"""
available_targets:
- n150
- n300
- llmbox
- tg
"""
config.targets = {"n150"}

if system_desc != None:
    if len(system_desc["chip_desc_indices"]) == 1:
        config.targets = {"n150"}
    elif len(system_desc["chip_desc_indices"]) == 2:
        config.targets = {"n300"}
    elif len(system_desc["chip_desc_indices"]) == 8:
        config.targets = {"llmbox"}
    elif len(system_desc["chip_desc_indices"]) == 32:
        config.targets = {"tg"}

for target in config.targets:
    config.available_features.add(target)

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

config.test_root = os.path.join(config.ttmlir_source_dir, "test")
config.scripts_root = os.path.join(config.ttmlir_source_dir, "tools/scripts")
config.models_root = os.path.join(config.ttmlir_source_dir, "test/ttmlir/models")

config.substitutions.append(("%ttmlir_test_root", config.test_root))
config.substitutions.append(("%models", config.models_root))
config.substitutions.append(("%ttmlir_scripts_root", config.scripts_root))

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

# Add `TT_MLIR_HOME` to lit environment.
if "TT_MLIR_HOME" in os.environ:
    llvm_config.with_environment("TT_MLIR_HOME", os.environ["TT_MLIR_HOME"])
else:
    raise OSError("Error: TT_MLIR_HOME not set")

# Add `TT_METAL_RUNTIME_ROOT` to lit environment.
if "TT_METAL_RUNTIME_ROOT" in os.environ:
    llvm_config.with_environment(
        "TT_METAL_RUNTIME_ROOT", os.environ["TT_METAL_RUNTIME_ROOT"]
    )
else:
    raise OSError("Error: TT_METAL_RUNTIME_ROOT not set")

# Add `TT_METAL_BUILD_HOME` to lit environment.
if "TT_METAL_BUILD_HOME" in os.environ:
    llvm_config.with_environment(
        "TT_METAL_BUILD_HOME", os.environ["TT_METAL_BUILD_HOME"]
    )
else:
    raise OSError("Error: TT_METAL_BUILD_HOME not set")
