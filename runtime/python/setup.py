# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import shutil
from pathlib import Path
from setuptools import setup

TTMLIR_VERSION_MAJOR = os.getenv("TTMLIR_VERSION_MAJOR", "0")
TTMLIR_VERSION_MINOR = os.getenv("TTMLIR_VERSION_MINOR", "0")
TTMLIR_VERSION_PATCH = os.getenv("TTMLIR_VERSION_PATCH", "0")

__version__ = f"{TTMLIR_VERSION_MAJOR}.{TTMLIR_VERSION_MINOR}.{TTMLIR_VERSION_PATCH}"

src_dir = os.environ.get(
    "SOURCE_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
)
ttmlir_build_dir = os.environ.get(
    "TTMLIR_BINARY_DIR",
    os.path.join(src_dir, "build"),
)

metaldir = f"{src_dir}/third_party/tt-metal/src/tt-metal/build"
ttmetalhome = os.environ.get("TT_METAL_RUNTIME_ROOT", "")

os.environ["LDFLAGS"] = "-Wl,-rpath,'$ORIGIN'"
enable_runtime = os.environ.get("TTMLIR_ENABLE_RUNTIME", "OFF") == "ON"
enable_ttnn = os.environ.get("TT_RUNTIME_ENABLE_TTNN", "OFF") == "ON"
enable_ttmetal = os.environ.get("TT_RUNTIME_ENABLE_TTMETAL", "OFF") == "ON"
enable_runtime_tests = os.environ.get("TTMLIR_ENABLE_RUNTIME_TESTS", "OFF") == "ON"
enable_perf = os.environ.get("TT_RUNTIME_ENABLE_PERF_TRACE", "OFF") == "ON"
arch = os.environ.get("CMAKE_SYSTEM_PROCESSOR", "x86_64")
py_maj_ver, py_min_ver, py_patch_ver = platform.python_version_tuple()

runtime_module = f"_ttmlir_runtime.cpython-{py_maj_ver}{py_min_ver}-{arch}-linux-gnu.so"
dylibs = []
runlibs = []
perflibs = []
metallibs = []

install_requires = ["torch"]

if enable_ttnn:
    runlibs += ["_ttnncpp.so"]

if enable_ttmetal:
    runlibs += ["libtt_metal.so"]

if enable_ttnn or enable_ttmetal:
    runlibs += ["libtt-umd.so"]
    runlibs += ["libtt_stl.so"]
    runlibs += ["libtracy.so.0.10.0"]

if enable_perf:
    perflibs += ["capture-release"]
    perflibs += ["csvexport-release"]

pkg_runtime_dir = f"{ttmlir_build_dir}/python_packages/ttmlir_runtime/runtime"

if enable_runtime:
    shutil.copy(
        f"{ttmlir_build_dir}/runtime/lib/libTTMLIRRuntime.so",
        pkg_runtime_dir,
    )
    shutil.copy(
        f"{ttmlir_build_dir}/runtime/python/{runtime_module}",
        pkg_runtime_dir,
    )

    for runlib in runlibs:
        shutil.copy(
            f"{metaldir}/lib/{runlib}",
            pkg_runtime_dir,
        )

    for dylib in perflibs:
        shutil.copy(
            f"{metaldir}/tools/profiler/bin/{dylib}",
            pkg_runtime_dir,
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
        return [
            item
            for item in contents
            if any(
                os.path.join(relative_folder, item).startswith(ignore)
                for ignore in tt_metal_folders_to_ignore
            )
        ]

    shutil.copytree(
        f"{ttmetalhome}/tt_metal",
        f"{pkg_runtime_dir}/tt_metal",
        dirs_exist_ok=True,
        ignore=tt_metal_ignore_folders,
        ignore_dangling_symlinks=True,
    )
    shutil.copytree(
        f"{ttmetalhome}/runtime",
        f"{pkg_runtime_dir}/runtime",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        f"{ttmetalhome}/ttnn",
        f"{pkg_runtime_dir}/ttnn",
        dirs_exist_ok=True,
    )

    def package_files(directory):
        paths = []
        for path, directories, filenames in os.walk(directory):
            for filename in filenames:
                paths.append(os.path.join("..", path, filename))
        return paths

    metallibs += package_files(f"{pkg_runtime_dir}/tt_metal/")
    metallibs += package_files(f"{pkg_runtime_dir}/runtime/")
    metallibs += package_files(f"{pkg_runtime_dir}/ttnn/")

dylibs += ["libTTMLIRRuntime.so", runtime_module]
dylibs += runlibs
dylibs += perflibs
dylibs += metallibs

packages = [
    "ttmlir_runtime",
    "ttmlir_runtime.runtime",
    "ttmlir_runtime.binary",
    "ttmlir_runtime.utils",
]
package_dir = {
    "ttmlir_runtime": f"{ttmlir_build_dir}/python_packages/ttmlir_runtime",
    "ttmlir_runtime.runtime": f"{ttmlir_build_dir}/python_packages/ttmlir_runtime/runtime",
    "ttmlir_runtime.binary": f"{ttmlir_build_dir}/python_packages/ttmlir_runtime/binary",
    "ttmlir_runtime.utils": f"{ttmlir_build_dir}/python_packages/ttmlir_runtime/utils",
}

setup(
    name="ttmlir_runtime",
    version=__version__,
    author="Tenstorrent",
    author_email="info@tenstorrent.com",
    url="https://github.com/tenstorrent/tt-mlir",
    description="Python bindings for the TTMLIR runtime",
    long_description="",
    packages=packages,
    package_dir=package_dir,
    install_requires=install_requires,
    package_data={"ttmlir_runtime.runtime": dylibs + ["_ttmlir_runtime/*.pyi"]},
    zip_safe=False,
    python_requires=">=3.7",
)
