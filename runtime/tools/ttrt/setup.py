# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import shutil
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

TTMLIR_VERSION_MAJOR = os.getenv("TTMLIR_VERSION_MAJOR", "0")
TTMLIR_VERSION_MINOR = os.getenv("TTMLIR_VERSION_MINOR", "0")
TTMLIR_VERSION_PATCH = os.getenv("TTMLIR_VERSION_PATCH", "0")

__version__ = f"{TTMLIR_VERSION_MAJOR}.{TTMLIR_VERSION_MINOR}.{TTMLIR_VERSION_PATCH}"

enable_perf = False

src_dir = os.environ.get(
    "SOURCE_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
)

# Use 'src_dir/build' as default location if TTMLIR_BINARY_DIR env variable is not available.
ttmlir_build_dir = os.environ.get(
    "TTMLIR_BINARY_DIR",
    os.path.join(src_dir, "build"),
)

print(ttmlir_build_dir)


class TTExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if "ttrt" in ext.name:
                self.build_(ext)
            else:
                raise Exception("Unknown extension")

    def build_(self, ext):
        build_lib = self.build_lib
        if not os.path.exists(build_lib):
            # Might be an editable install or something else
            return

        extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))

        print(f"Running cmake to install ttrt at {extension_path}")

        cwd = pathlib.Path().absolute()

        # Set it to install directly into the wheel
        install_dir = pathlib.Path(self.build_lib) / "ttrt" / "runtime"

        assert os.path.exists(
            ttmlir_build_dir
        ), "This script can only run with a preconfigured cmake build"
        # self.spawn(["cmake", "--build", str(ttmlir_build_dir)])

        # Install the ttrt Component
        self.spawn(
            [
                "cmake",
                "--install",
                str(ttmlir_build_dir),
                "--component",
                "SharedLib",
                "--prefix",
                str(install_dir),
            ]
        )


# Only the ttrt package relies on the CMake build process
cmake_ttrt = TTExtension("ttrt")


install_requires = []
install_requires += ["nanobind"]
if enable_perf:
    install_requires += ["loguru"]
    install_requires += ["pandas"]
    install_requires += ["seaborn"]
    install_requires += ["graphviz"]
    install_requires += ["pyyaml"]
    install_requires += ["click"]

setup(
    name="ttrt",
    version=__version__,
    author="Nicholas Smith",
    author_email="nsmith@tenstorrent.com",
    url="https://github.com/tenstorrent/tt-mlir",
    description="Python bindings to runtime libraries",
    long_description="",
    packages=["ttrt", "ttrt.common", "ttrt.binary", "ttrt.runtime"],
    package_dir={},
    install_requires=install_requires,
    entry_points={
        "console_scripts": ["ttrt = ttrt:main"],
    },
    ext_modules=[cmake_ttrt],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.7",
)
