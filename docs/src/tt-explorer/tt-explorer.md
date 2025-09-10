# `tt-explorer`

Welcome to the `tt-explorer` wiki! The Wiki will serve as a source for documentation, examples, and general knowledge related to the TT-MLIR visualization project. The sidebar will provide navigation to relevant pages. If this is your first time hearing about the project, take a look at Project Architecture for an in-depth introduction to the tool and motivations behind it. 🙂

## Overview

Visualizer tool for `ttmlir`-powered compiler results. Visualizes from emitted `.mlir` files to display compiled model, attributes, performance results, and provides a platform for human-driven overrides to _gameify_ model tuning.

## Quick Start

`tt-explorer` comes packaged as a tool in the `tt-mlir` repo. If you haven't done so yet, please refer to ["Setting up the environment manually"](../getting-started.md#setting-up-the-environment-manually) section from the Getting Started Guide to build the environment manually.

Here is a summary of the steps needed:

1. Clone `tt-mlir` and build the environment
2. Run `source env/activate` to be in `tt-mlir` virtualenv for the following steps
3. Ensure `tt-mlir` is built with atleast these flags:
   - `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON`
   - `-DTTMLIR_ENABLE_RUNTIME=ON`
   - `-DTT_RUNTIME_DEBUG=ON`
   - `-DTTMLIR_ENABLE_STABLEHLO=ON`
4. Build `explorer` target in `tt-mlir` using `cmake --build build -- explorer`
5. Run `tt-explorer` in terminal to start `tt-explorer` instance. (Refer to CLI section in API for specifics)
   - **Note**: `tt-explorer` requires [Pandas](https://pypi.org/project/pandas/) in addition to the `tt-mlir` [System Dependencies](https://docs.tenstorrent.com/tt-mlir/getting-started.html#system-dependencies).
6. Ensure server has started in `tt-explorer` shell instance (check for message below)
   ```
   Starting Model Explorer server at:
   http://localhost:8080
   ```

## Building `tt-explorer`

To build `tt-explorer` you need first to clone and configure the environment for `tt-mlir`. Please refer to the [Getting Started Guide](../getting-started.md).

After building and activating the virtualenv, build `tt-mlir` and ensure the following flags are present, as they are needed for executing models in `tt-explorer` and without them it won't build.

Flags required:

- `-DTT_RUNTIME_ENABLE_PERF_TRACE=ON`
- `-DTTMLIR_ENABLE_RUNTIME=ON`
- `-DTT_RUNTIME_DEBUG=ON`
- `-DTTMLIR_ENABLE_STABLEHLO=ON`

Then build the `explorer` target by running the following command:

```sh
cmake --build build -- explorer
```

After it finishes building, start the `explorer` server by running the following command:

```sh
tt-explorer
```

The server should then start and show a message similar to this:

```
Starting Model Explorer server at:
http://localhost:8080
```

### Running `tt-explorer` CI Tests Locally

> **Note:** CI tests are ran like described below. Here we provide the steps needed to reproduce it and debug failing CI tests locally.

`tt-explorer` relies on tests that are present in the `tests/` directory as well as tests dynamically created through `llvm-lit`. Below are the steps to replicate the testing procedure seen in CI:

1. Make sure you're in the `tt-mlir` directory
2. You need to build the explorer target with `cmake --build build -- explorer`
3. Run and save the system descriptor `ttrt query --save-artifacts`
4. Save the system variable `export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys`
5. Run and generate ttnn + MLIR tests: `cmake --build build -- check-ttmlir`
6. Save the relevant test directories:
   - `export TT_EXPLORER_GENERATED_MLIR_TEST_DIRS=$(pwd)/build/test/python/golden/ttnn,$(pwd)/build/test/ttmlir/Silicon/TTNN/n150/perf`
   - `export TT_EXPLORER_GENERATED_TTNN_TEST_DIRS=$(pwd)/build/test/python/golden/ttnn`
7. Run the pytest for `tt-explorer` with `pytest tools/explorer/test/run_tests.py`

or in a concise shell script:

```sh
# Ensure you are present in the tt-mlir directory
source env/activate

# Build Tests
cmake --build build -- explorer
ttrt query --save-artifacts
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys
cmake --build build -- check-ttmlir

# Load Tests
export TT_EXPLORER_GENERATED_MLIR_TEST_DIRS=$(pwd)/build/test/python/golden/ttnn,$(pwd)/build/test/ttmlir/Silicon/TTNN/n150/perf
export TT_EXPLORER_GENERATED_TTNN_TEST_DIRS=$(pwd)/build/test/python/golden/ttnn

# Run Tests
pytest tools/explorer/test/run_tests.py
```
