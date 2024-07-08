# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import shutil

__version__ = "0.0.1"

src_dir = os.environ.get(
    "SOURCE_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
)
toolchain = os.environ.get("TTMLIR_TOOLCHAIN_DIR", "/opt/ttmlir-toolchain")
metallibdir = f"{src_dir}/third_party/tt-metal/src/tt-metal-build/lib"

os.environ["LDFLAGS"] = "-Wl,-rpath,'$ORIGIN'"
enable_runtime = os.environ.get("TTMLIR_ENABLE_RUNTIME", "OFF") == "ON"

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
        libraries=["TTRuntime", "flatbuffers"],
        library_dirs=[
            f"{src_dir}/build/runtime/lib",
            f"{toolchain}/lib",
        ],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

if enable_runtime:
    shutil.copy(
        f"{metallibdir}/_ttnn.so", f"{src_dir}/build/runtime/tools/python/ttrt/runtime"
    )
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
            libraries=["TTRuntime", "TTRuntimeTTNN", ":_ttnn.so", "flatbuffers"],
            library_dirs=[
                f"{src_dir}/build/runtime/lib",
                f"{toolchain}/lib",
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
    packages=["ttrt", "ttrt.binary", "ttrt.runtime"],
    install_requires=["pybind11"],
    entry_points={
        "console_scripts": ["ttrt = ttrt:main"],
    },
    package_data={"ttrt.runtime": [f"_ttnn.so"]},
    zip_safe=False,
    python_requires=">=3.7",
)
