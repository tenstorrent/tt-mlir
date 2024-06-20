import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

src_dir = os.environ.get(
    "SOURCE_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
)
toolchain = os.environ.get("TOOLCHAIN_ENV", "/opt/ttmlir-toolchain")

ext_modules = [
    Pybind11Extension(
        "ttrt.binary._C",
        ["ttrt/binary/module.cpp"],
        include_dirs=[
            f"{src_dir}/runtime/include",
            f"{src_dir}/build/include",
            f"{src_dir}/build/include/ttmlir/Target/Common",
        ],
        libraries=["TTRuntimeBinary", "flatbuffers"],
        library_dirs=[
            f"{src_dir}/build/runtime/lib",
            f"{toolchain}/lib",
        ],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

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
    packages=["ttrt", "ttrt.binary"],
    install_requires=["pybind11"],
    entry_points={
        "console_scripts": ["ttrt = ttrt:main"],
    },
    zip_safe=False,
    python_requires=">=3.7",
)
