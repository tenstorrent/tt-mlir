# Building

These are the steps required to get the TT-MLIR project running on your machine

Please refer to the [Dependencies](#dependencies) section before building the project.

## Environment setup

You only need to build this once, it builds llvm, flatbuffers and a python virtual environment.

```bash
cmake -B env/build env
cmake --build env/build
```

> - It is recommended to use the **system installation of python3** for the virtual environment.
>   Please ensure that you do not already have a venv activated before running the above command.
> - Please ensure the directory `/opt/ttmlir-toolchain` exist and its
>   owner is the current user, i.e. the one that executes the above `cmake` commands.
>   The commands create it and assign the proper ownership are:
>     ```bash
>     sudo mkdir -p /opt/ttmlir-toolchain
>     sudo chown -R $USER /opt/ttmlir-toolchain
>     ```


## Build

```bash
source env/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build
```

> - To enable the ttnn/metal runtime add `-DTTMLIR_ENABLE_RUNTIME=ON`
> - To accelerate the builds with ccache use `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`
> - To accelerate builds further, if python bindings aren't needed, `-DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF`. For some reason the python bindings link step is very slow.
> - TTNN build is automatically integrated / handled by tt-mlir cmake build system.  For debugging and further information regarding the TTNN backend build step, please refer to [TTNN Documentation](https://tenstorrent.github.io/tt-metal/latest/ttnn/ttnn/installing.html).
> - The runtime build  step depends on the `ARCH_NAME` environment variable, which is set in the `env/activate` script.
>   If you want to build the runtime for a different architecture, please set `ARCH_NAME` to the desired value
>   (one of `grayskull`, `wormhole_b0`, or `blackhole`).
>   Please note that the runtime is built only if `TTMLIR_ENABLE_RUNTIME=ON`.
> - In addition to `ARCH_NAME`, the runtime build depends on `TT_METAL_HOME` variable,
>   which is also set in `env/activate` script.
>   For more information, please refer to
>   [TT-NN and TT-Metailium installation documentation](https://tenstorrent.github.io/tt-metal/latest/ttnn/ttnn/installing.html#step-4-install-and-start-using-tt-nn-and-tt-metalium).

| OS | Offline Compiler Only | Runtime Enabled Build |
|----|-----------------------|-----------------------|
| Ubuntu 22.04  | ✅ | ❌ |
| Ubuntu 20.04  | ✅ | ✅ |
| MacOS         | ✅ | ❌ |

## Test

```bash
source env/activate
cmake --build build -- check-ttmlir
```

### llvm-lit

Under the hood the check-ttmlir cmake target is running `llvm-lit`. With it you
can:

```bash
# Query which tests are available
llvm-lit -sv ./build/test --show-tests

# Run an individual test:
llvm-lit -sv ./build/test/ttmlir/Dialect/TTIR/test_allocate.mlir

# Run a sub-suite:
llvm-lit -sv ./build/test/ttmlir/Dialect/TTIR
```

> See the full [llvm-lit documentation](https://llvm.org/docs/CommandGuide/lit.html) for more information.

## Lint

```bash
source env/activate
cmake --build build -- clang-tidy
```

### Pre-Commit
Pre-Commit applies a git hook to the local repository such that linting is checked and applied on every `git commit` action. Install from the root of the repository using:

```bash
source env/activate
pre-commit install
```

If you have already committed before installing the pre-commit hooks, you can run on all files to "catch up":

```bash
pre-commit run --all-files
```

For more information visit [pre-commit](https://pre-commit.com/)

## Docs

```bash
source env/activate
cmake --build build -- docs
mdbook serve build/docs/book
```

> - `mdbook` can be installed with the system's package manager.

## Dependencies

### Ubuntu Common

Make sure to have Git LFS installed. You can install it with the following command:

```bash
sudo apt-get install git-lfs
```

### Ubuntu 22.04

We need to install Ninja which can be done with the following command

```bash
sudo apt install ninja-build
```

### Ubuntu 20.04

On Ubuntu 20.04, we need to update the version because 3.20 is the minimum required for this project

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt remove cmake -y
pip3 install cmake --upgrade
hash -r
```

> Ensure cmake can by found in this path pip installed it to. E.g. `PATH=$PATH:$HOME/.local/bin`

Then run the following command to see the cmake version which should be later than 3.20

```bash
cmake --version
```

We also need to install Ninja which can be done with the following command

```bash
sudo apt install ninja-build
```

### MacOS

On MacOS we need to install the latest version of cmake and ninja which can be done using Homebrew with (Docs for installing Homebrew: https://brew.sh)

```bash
brew install cmake
brew install ninja
```

## Common Build Errors

### `TTMLIRPythonCAPI target requires changing an RPATH`

```
CMake Error at /opt/ttmlir-toolchain/lib/cmake/llvm/AddLLVM.cmake:594 (add_library):
  The install of the TTMLIRPythonCAPI target requires changing an RPATH from
  the build tree, but this is not supported with the Ninja generator unless
  on an ELF-based or XCOFF-based platform.  The
  CMAKE_BUILD_WITH_INSTALL_RPATH variable may be set to avoid this relinking
  step.
```

If you get the above error, it means you tried to build with an old version of cmake or ninja and there is a stale file. To fix this, `rm -rf` your build directory, install a newer version of cmake/ninja, and then rebuild. If you installed ninja via `sudo apt install ninja-build`, it might still be not up-to-date (v1.10.0). You may use ninja in the python virtual environment, or install it via `pip3 install -U ninja`, either way the version `1.11.1.git.kitware.jobserver-1` should work.

### `clang++ is not a full path and was not found in the PATH`

```
CMake Error at CMakeLists.txt:2 (project):
  The CMAKE_CXX_COMPILER:
    clang++
  is not a full path and was not found in the PATH.
  Tell CMake where to find the compiler by setting either the environment
  variable "CXX" or the CMake cache entry CMAKE_CXX_COMPILER to the full path
  to the compiler, or to the compiler name if it is in the PATH.
CMake Error at CMakeLists.txt:2 (project):
  The CMAKE_C_COMPILER:
    clang
  is not a full path and was not found in the PATH.
  Tell CMake where to find the compiler by setting either the environment
  variable "CC" or the CMake cache entry CMAKE_C_COMPILER to the full path to
  the compiler, or to the compiler name if it is in the PATH.
```

If you get the following error, it means you need to install clang which you can do with `sudo apt install clang` on Ubuntu.

### `sfpi`, `trisc`, `ncrisc` build failure

```
pybuda/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-g++: 1: version: not found
pybuda/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-g++: 2: oid: not found
size: '1961632': No such file
size: '1961632': No such file
size: '1961632': No such file
Always | FATAL | ncrisc build failed
```

If you got the above error, it means that SFPI or similar component build failed. First, make sure you have GIT LFS setup (e.g. sudo apt-get install git-lfs). Then, try to pull SFPI submodule manually:

```bash
cd third_party/tt-mlir/third_party/tt-metal/src/tt-metal
git submodule foreach 'git lfs fetch --all && git lfs pull’
```

Then, try to build again.
