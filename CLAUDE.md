# Project Overview
tt-mlir is a compiler project aimed at defining MLIR dialects to abstract
compute on Tenstorrent AI accelerators. It is built on top of the MLIR and
targets tt-metal.

# Code Style
- Follow the style of the current file.
- LLVM style is preferred.

# Commands
- `source env/activate`: Activates the project's virtual environment.  Run this
  command before running any other commands.
- `cmake -G Ninja -B build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache`: Configures the
  project using CMake.
- `cmake --build build`: Builds and installs the project into venv using CMake.
- `ttrt query --save-artifacts`: Runs ttrt to query and save a system descriptor
  to location `ttrt-artifacts/system_desc.ttsys`.  Required before running
  python tests.
- `pre-commit run --all-files`: Runs pre-commit to lint code style.


# Testing
- `source env/activate`: Activates the project's virtual environment.  Run this
  command before running any other commands.
- `cmake --build ${BUILD_DIR} --target check-ttmlir`: Runs compiler tests.
- `cmake --build ${BUILD_DIR} --target check-perf`: Runs optimizer models perf tests (resnet, yolo_v8, segformer). Requires `-DTTMLIR_ENABLE_OPMODEL=ON`.
- `llvm-lit test/ttmlir/.../*.mlir`: Runs LLVM lit driven compiler tests.
- `pytest test/python`: Runs Python driven compiler tests and generates flatbuffers.
  Usually required to have run `ttrt query --save-artifacts` first and then to
  set the environment variable `SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys`.
- `ttrt run ...`: Runs a compiler generated flatbuffer on silicon.  Can either
  point to a flabuffer file or a directory containing flatbuffers.
