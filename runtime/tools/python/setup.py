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
metaldir = f"{src_dir}/third_party/tt-metal/src/tt-metal-build"
ttmetalhome = os.environ.get("TT_METAL_HOME", "")

os.environ["LDFLAGS"] = "-Wl,-rpath,'$ORIGIN'"
enable_runtime = os.environ.get("TTMLIR_ENABLE_RUNTIME", "OFF") == "ON"
enable_ttnn = os.environ.get("TT_RUNTIME_ENABLE_TTNN", "OFF") == "ON"
enable_ttmetal = os.environ.get("TT_RUNTIME_ENABLE_TTMETAL", "OFF") == "ON"
enable_perf = os.environ.get("TT_RUNTIME_ENABLE_PERF_TRACE", "OFF") == "ON"
debug_runtime = os.environ.get("TT_RUNTIME_DEBUG", "OFF") == "ON"

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
runlibs = []
perflibs = []
metallibs = []
install_requires = []
install_requires += ["pybind11"]

linklibs = ["TTBinary", "TTRuntimeSysDesc"]
if enable_ttnn:
    runlibs += ["_ttnn.so"]
    linklibs += ["TTRuntimeTTNN", "TTRuntimeTTNNOps", ":_ttnn.so"]

if enable_ttmetal:
    runlibs += ["libtt_metal.so"]
    linklibs += ["TTRuntimeTTMetal", "tt_metal"]

if enable_ttnn or enable_ttmetal:
    runlibs += ["libdevice.so", "libnng.so.1", "libuv.so.1"]
    linklibs += ["TTRuntimeDebug"]

if enable_perf:
    runlibs += ["libtracy.so.0.10.0"]
    perflibs += ["capture-release"]
    perflibs += ["csvexport-release"]

if enable_runtime:
    assert enable_ttmetal or enable_ttnn, "At least one runtime must be enabled"

    for dylib in runlibs:
        shutil.copy(
            f"{metaldir}/lib/{dylib}",
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

    for dylib in perflibs:
        shutil.copy(
            f"{metaldir}/tools/profiler/bin/{dylib}",
            f"{src_dir}/build/runtime/tools/python/ttrt/runtime",
        )
        shutil.copy(
            f"{metaldir}/tools/profiler/bin/{dylib}",
            f"{ttmetalhome}/{dylib}",
        )

    # copy metal dir folder
    shutil.copytree(
        f"{ttmetalhome}/tt_metal",
        f"{src_dir}/build/runtime/tools/python/ttrt/runtime/tt_metal",
        dirs_exist_ok=True,
    )

    # copy runtime dir folder
    shutil.copytree(
        f"{ttmetalhome}/runtime",
        f"{src_dir}/build/runtime/tools/python/ttrt/runtime/runtime",
        dirs_exist_ok=True,
    )

    # copy kernels
    shutil.copytree(
        f"{ttmetalhome}/ttnn",
        f"{src_dir}/build/runtime/tools/python/ttrt/runtime/ttnn",
        dirs_exist_ok=True,
    )

    # copy test folder
    shutil.copytree(
        f"{ttmetalhome}/tests",
        f"{src_dir}/build/runtime/tools/python/ttrt/runtime/tests",
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
        f"{src_dir}/build/runtime/tools/python/ttrt/runtime/tt_metal/"
    )
    extra_files_runtime = package_files(
        f"{src_dir}/build/runtime/tools/python/ttrt/runtime/runtime/"
    )
    extra_files_ttnn = package_files(
        f"{src_dir}/build/runtime/tools/python/ttrt/runtime/ttnn/"
    )
    extra_files_tests = package_files(
        f"{src_dir}/build/runtime/tools/python/ttrt/runtime/tests/"
    )

    metallibs += extra_files_tt_metal
    metallibs += extra_files_runtime
    metallibs += extra_files_ttnn
    metallibs += extra_files_tests

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
                f"{src_dir}/build/runtime/lib/ttnn/operations",
                f"{src_dir}/build/runtime/lib/ttmetal",
                f"{toolchain}/lib",
                f"{src_dir}/build/runtime/tools/python/ttrt/runtime",
                f"{metaldir}/lib",
            ],
            define_macros=[
                ("VERSION_INFO", __version__),
                ("TT_RUNTIME_DEBUG", "1" if debug_runtime else "0"),
            ],
        )
    )

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
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
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
