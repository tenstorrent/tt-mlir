# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from setuptools import setup
import shutil
import subprocess

TTMLIR_VERSION_MAJOR = os.getenv("TTMLIR_VERSION_MAJOR", "0")
TTMLIR_VERSION_MINOR = os.getenv("TTMLIR_VERSION_MINOR", "0")
TTMLIR_VERSION_PATCH = os.getenv("TTMLIR_VERSION_PATCH", "0")

__version__ = f"{TTMLIR_VERSION_MAJOR}.{TTMLIR_VERSION_MINOR}.{TTMLIR_VERSION_PATCH}"

src_dir = os.environ.get(
    "SOURCE_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
)
# Use 'src_dir/build' as default location if TTMLIR_BINARY_DIR env variable is not available.
ttmlir_build_dir = os.environ.get(
    "TTMLIR_BINARY_DIR",
    os.path.join(src_dir, "build"),
)
toolchain = os.environ.get("TTMLIR_TOOLCHAIN_DIR", "/opt/ttmlir-toolchain")
metaldir = f"{src_dir}/third_party/tt-metal/src/tt-metal/build"
ttmetalhome = os.environ.get("TT_METAL_HOME", "")

os.environ["LDFLAGS"] = "-Wl,-rpath,'$ORIGIN'"
enable_runtime = os.environ.get("TTMLIR_ENABLE_RUNTIME", "OFF") == "ON"
enable_ttnn = os.environ.get("TT_RUNTIME_ENABLE_TTNN", "OFF") == "ON"
enable_ttmetal = os.environ.get("TT_RUNTIME_ENABLE_TTMETAL", "OFF") == "ON"
enable_runtime_tests = os.environ.get("TTMLIR_ENABLE_RUNTIME_TESTS", "OFF") == "ON"
enable_perf = os.environ.get("TT_RUNTIME_ENABLE_PERF_TRACE", "OFF") == "ON"
debug_runtime = os.environ.get("TT_RUNTIME_DEBUG", "OFF") == "ON"
arch = os.environ.get("CMAKE_SYSTEM_PROCESSOR", "x86_64")

runtime_module = f"_ttmlir_runtime.cpython-310-{arch}-linux-gnu.so"
dylibs = []
runlibs = []
perflibs = []
metallibs = []
install_requires = []
install_requires += ["nanobind"]

if enable_ttnn:
    runlibs += ["_ttnncpp.so"]

if enable_ttmetal:
    runlibs += ["libtt_metal.so"]

if enable_ttnn or enable_ttmetal:
    runlibs += ["libdevice.so"]
    runlibs += ["libtt_stl.so"]
    runlibs += ["libtracy.so.0.10.0"]

if enable_perf:
    perflibs += ["capture-release"]
    perflibs += ["csvexport-release"]

if enable_runtime:
    assert enable_ttmetal or enable_ttnn, "At least one runtime must be enabled"

    shutil.copy(
        f"{ttmlir_build_dir}/runtime/lib/libTTMLIRRuntime.so",
        f"{ttmlir_build_dir}/runtime/tools/ttrt/ttrt/runtime",
    )

    shutil.copy(
        f"{ttmlir_build_dir}/runtime/python/{runtime_module}",
        f"{ttmlir_build_dir}/runtime/tools/ttrt/ttrt/runtime",
    )

    for runlib in runlibs:
        shutil.copy(
            f"{metaldir}/lib/{runlib}",
            f"{ttmlir_build_dir}/runtime/tools/ttrt/ttrt/runtime",
        )

    for dylib in perflibs:
        shutil.copy(
            f"{metaldir}/tools/profiler/bin/{dylib}",
            f"{ttmlir_build_dir}/runtime/tools/ttrt/ttrt/runtime",
        )
        shutil.copy(
            f"{metaldir}/tools/profiler/bin/{dylib}",
            f"{ttmetalhome}/{dylib}",
        )

    tt_metal_folders_to_ignore = [
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-addr2line",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-ar",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-as",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-c++",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-c++filt",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-cpp",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-elfedit",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-gcc",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-gcc-10.2.0",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-gcc-12.2.0",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-gcc-ar",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-gcc-nm",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-gcc-ranlib",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-gcov",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-gcov-dump",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-gcov-tool",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-gdb",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-gdb-add-index",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-gprof",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-ld",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-ld.bfd",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-lto-dump",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-nm",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-objdump",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-ranlib",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-readelf",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-run",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-size",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-strings",
        "third_party/sfpi/compiler/bin/riscv32-unknown-elf-strip",
        "third_party/sfpi/compiler/share",
        "third_party/sfpi/compiler/compiler",
        "third_party/fmt",
        "third_party/pybind11",
        "third_party/taskflow",
        "third_party/json",
        "third_party/magic_enum",
        "third_party/tracy",
    ]

    def tt_metal_ignore_folders(folder, contents):
        relative_folder = os.path.relpath(folder, start=f"{ttmetalhome}/tt_metal")

        ignored_items = [
            item
            for item in contents
            if any(
                os.path.join(relative_folder, item).startswith(ignore)
                for ignore in tt_metal_folders_to_ignore
            )
        ]

        return ignored_items

    # copy metal dir folder
    shutil.copytree(
        f"{ttmetalhome}/tt_metal",
        f"{ttmlir_build_dir}/runtime/tools/ttrt/ttrt/runtime/tt_metal",
        dirs_exist_ok=True,
        ignore=tt_metal_ignore_folders,
    )

    # copy runtime dir folder
    shutil.copytree(
        f"{ttmetalhome}/runtime",
        f"{ttmlir_build_dir}/runtime/tools/ttrt/ttrt/runtime/runtime",
        dirs_exist_ok=True,
    )

    # copy kernels
    shutil.copytree(
        f"{ttmetalhome}/ttnn",
        f"{ttmlir_build_dir}/runtime/tools/ttrt/ttrt/runtime/ttnn",
        dirs_exist_ok=True,
    )

    import os

    def package_files(directory):
        paths = []
        for path, directories, filenames in os.walk(directory):
            for filename in filenames:
                paths.append(os.path.join("..", path, filename))
        return paths

    extra_files_tt_metal = package_files(
        f"{ttmlir_build_dir}/runtime/tools/ttrt/ttrt/runtime/tt_metal/"
    )
    extra_files_runtime = package_files(
        f"{ttmlir_build_dir}/runtime/tools/ttrt/ttrt/runtime/runtime/"
    )
    extra_files_ttnn = package_files(
        f"{ttmlir_build_dir}/runtime/tools/ttrt/ttrt/runtime/ttnn/"
    )
    extra_files_tests = package_files(
        f"{ttmlir_build_dir}/runtime/tools/ttrt/ttrt/runtime/tests/"
    )

    metallibs += extra_files_tt_metal
    metallibs += extra_files_runtime
    metallibs += extra_files_ttnn
    metallibs += extra_files_tests

dylibs += ["libTTMLIRRuntime.so", runtime_module]
dylibs += runlibs
dylibs += perflibs
dylibs += metallibs

packages = ["ttrt", "ttrt.common", "ttrt.binary", "ttrt.runtime"]
package_dir = {}
if enable_perf:
    install_requires += ["loguru"]
    install_requires += ["pandas"]
    install_requires += ["seaborn"]
    install_requires += ["graphviz"]
    install_requires += ["pyyaml"]
    install_requires += ["click"]
    packages += ["tracy"]
    packages += ["tt_metal"]
    package_dir["tracy"] = f"{ttmetalhome}/ttnn/tracy"
    package_dir["tt_metal"] = f"{ttmetalhome}/tt_metal"

setup(
    name="ttrt",
    version=__version__,
    author="Nicholas Smith",
    author_email="nsmith@tenstorrent.com",
    url="https://github.com/tenstorrent/tt-mlir",
    description="Python bindings to runtime libraries",
    long_description="",
    packages=packages,
    package_dir=package_dir,
    install_requires=install_requires,
    entry_points={
        "console_scripts": ["ttrt = ttrt:main"],
    },
    package_data={"ttrt.runtime": dylibs},
    # include_package_data=True,
    zip_safe=False,
    python_requires=">=3.7",
)
