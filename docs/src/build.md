# Building

These are the steps required to get the TT-MLIR project running on your machine

## Environment setup

You only need to build this once, it builds a python virtual environment with the necessary dependencies, flatbuffers, and llvm.

```bash
cmake -B env/build env
```

Running this script will give you a CMake error that looks like
  CMake Error at CMakeLists.txt:8 (message):
  The directory /opt/ttmlir-toolchain does not exist.  Please create it
  before running this script.

Along with 2 commands that it says you need to run before this script that look like

```bash
sudo mkdir -p /opt/ttmlir-toolchain
sudo chown -R $USER /opt/ttmlir-toolchain
```

Run the 2 commands from the error message and then rerun

```bash
cmake -B env/build env
```

Then run

```bash
cmake --build env/build
```

## Build

Here is the current compatibility matrix for the different kind of builds supported

| OS | Offline Compiler Only | Runtime Enabled Build |
|----|-----------------------|-----------------------|
| Ubuntu 22.04  | ✅ | ❌ |
| Ubuntu 20.04  | ✅ | ✅ |
| MacOS         | ✅ | ❌ |

```bash
source env/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build
```

Note: To accelerate the builds with ccache use `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`

Note:
- TTNN build is automatically integrated / handled by tt-mlir cmake build system.  For debugging and further information regarding the TTNN backend build step, please refer to https://tenstorrent.github.io/tt-metal/latest/ttnn/ttnn/installing.html
- To enable the ttnn/metal runtime add `-DTTMLIR_ENABLE_RUNTIME=ON`

## Dependencies

### Ubuntu 22.04

We need to install Ninja which can be done with the following command

```bash
sudo apt install ninja-build
```

### Ubuntu 20.04

On Ubuntu 20.04, we need to update the version because 3.2 is the minimum required for this project

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt remove cmake -y
pip3 install cmake --upgrade
hash -r
```

Then run the following command to see the cmake version which should be later than 3.2

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

## Errors

If you get the following error, it means you tried to build with an old version of cmake and there is a stale file. To fix this, `rm -rf` your build directory and then rerun the two previous commands

```bash
CMake Error at /opt/ttmlir-toolchain/lib/cmake/llvm/AddLLVM.cmake:594 (add_library):
  The install of the TTMLIRPythonCAPI target requires changing an RPATH from
  the build tree, but this is not supported with the Ninja generator unless
  on an ELF-based or XCOFF-based platform.  The
  CMAKE_BUILD_WITH_INSTALL_RPATH variable may be set to avoid this relinking
  step.
 ```

If you get the following error, it means you need to install clang which you can do with `sudo apt install clang` on Ubuntu

```bash
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

### Test

```bash
source env/activate
cmake --build build -- check-ttmlir
```

## Lint

- `clang-tidy`: Run clang-tidy on the project
```bash
source env/activate
cmake --build build -- clang-tidy
```

## Docs

Doc dependencies: `mdbook`

```bash
source env/activate
cmake --build build -- docs
mdbook serve build/docs/book
```
