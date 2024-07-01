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

## Build

```bash
source env/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build
```

> - To enable the ttnn/metal runtime add `-DTTMLIR_ENABLE_RUNTIME=ON`
> - To accelerate the builds with ccache use `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`
> - To accelerate builds further, if python bindings aren't needed, `-DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF`. For some reason the python bindings link step is very slow.
> - TTNN build is automatically integrated / handled by tt-mlir cmake build system.  For debugging and further information regarding the TTNN backend build step, please refer to [TTNN Documentation](https://tenstorrent.github.io/tt-metal/latest/ttnn/ttnn/installing.html)

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

## Lint

```bash
source env/activate
cmake --build build -- clang-tidy
```

## Docs

```bash
source env/activate
cmake --build build -- docs
mdbook serve build/docs/book
```

> - `mdbook` can be installed with the system's package manager.

## Dependencies

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

If you get the above error, it means you tried to build with an old version of cmake and there is a stale file. To fix this, `rm -rf` your build directory, install a newer version of cmake, and then rebuild.

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
