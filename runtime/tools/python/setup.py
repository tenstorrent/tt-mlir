# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
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
toolchain = os.environ.get("TTMLIR_TOOLCHAIN_DIR", "/opt/ttmlir-toolchain")
metallibdir = f"{src_dir}/third_party/tt-metal/src/tt-metal-build/lib"

os.environ["LDFLAGS"] = "-Wl,-rpath,'$ORIGIN'"
enable_runtime = os.environ.get("TTMLIR_ENABLE_RUNTIME", "OFF") == "ON"
enable_ttnn = os.environ.get("TT_RUNTIME_ENABLE_TTNN", "OFF") == "ON"
enable_ttmetal = os.environ.get("TT_RUNTIME_ENABLE_TTMETAL", "OFF") == "ON"
enable_perf = os.environ.get("TT_RUNTIME_ENABLE_PERF_TRACE", "OFF") == "ON"

ext_modules = [
    Pybind11Extension(
        "ttrt.binary._C",
        ["ttrt/binary/module.cpp"],
        include_dirs=[
            f"{toolchain}/include",
            f"{src_dir}/runtime/include",
            f"{src_dir}/build/include",
            f"{src_dir}/build/include/ttmlir/Target/Common",
        ],
        libraries=["TTBinary", "flatbuffers"],
        library_dirs=[
            f"{src_dir}/build/runtime/lib",
            f"{toolchain}/lib",
        ],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

dylibs = []
linklibs = ["TTBinary", "TTRuntimeSysDesc"]
if enable_ttnn:
    dylibs += ["_ttnn.so"]
    linklibs += ["TTRuntimeTTNN", ":_ttnn.so"]

if enable_ttmetal:
    dylibs += ["libtt_metal.so"]
    linklibs += ["TTRuntimeTTMetal", "tt_metal"]

if enable_perf:
    dylibs += ["libtracy.so.0.10.0"]

if enable_runtime:
    assert enable_ttmetal or enable_ttnn, "At least one runtime must be enabled"

    for dylib in dylibs:
        shutil.copy(
            f"{metallibdir}/{dylib}",
            f"{src_dir}/build/runtime/tools/python/ttrt/runtime",
        )
        command = [
            "patchelf",
            "--set-rpath",
            "$ORIGIN",
            f"{src_dir}/build/runtime/tools/python/ttrt/runtime/{dylib}",
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
    ext_modules.append(
        Pybind11Extension(
            "ttrt.runtime._C",
            ["ttrt/runtime/module.cpp"],
            include_dirs=[
                f"{toolchain}/include",
                f"{src_dir}/runtime/include",
                f"{src_dir}/build/include",
                f"{src_dir}/build/include/ttmlir/Target/Common",
            ],
            libraries=["TTRuntime"] + linklibs + ["flatbuffers"],
            library_dirs=[
                f"{src_dir}/build/runtime/lib",
                f"{src_dir}/build/runtime/lib/common",
                f"{src_dir}/build/runtime/lib/ttnn",
                f"{src_dir}/build/runtime/lib/ttmetal",
                f"{toolchain}/lib",
                f"{src_dir}/build/runtime/tools/python/ttrt/runtime",
                f"{metallibdir}",
            ],
            define_macros=[("VERSION_INFO", __version__)],
        )
    )

setup(
    name="ttrt",
    version=__version__,
    author="Nicholas Smith",
    author_email="nsmith@tenstorrent.com",
    url="https://github.com/tenstorrent/tt-mlir",
    description="Python bindings to runtime libraries",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["ttrt", "ttrt.common", "ttrt.binary", "ttrt.runtime"],
    install_requires=["pybind11"],
    entry_points={
        "console_scripts": ["ttrt = ttrt:main"],
    },
    package_data={"ttrt.runtime": dylibs},
    zip_safe=False,
    python_requires=">=3.7",
)
