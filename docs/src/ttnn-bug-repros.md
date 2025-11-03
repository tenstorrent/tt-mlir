# Creating Bug Repros for TTNN Using TT-MLIR Codegen

While developing in [tt-mlir](https://github.com/tenstorrent/tt-mlir), it's not uncommon to encounter bugs originating in the [TTNN](https://github.com/tenstorrent/tt-metal/tree/main/ttnn) library. To isolate and report such bugs, a practical approach is to use the C++ codegen feature (`EmitC`) to generate a minimal repro. This guide walks you through how to create such repros and integrate them into the [tt-metal](https://github.com/tenstorrent/tt-metal) repository, where TTNN is developed.

---

## Step-by-Step Guide

> **Note:** If you run into issues while following these steps, check the [Known Issues](#known-issues) section at the end of this guide for common problems and solutions.

### 1. Generate C++ Code from TT-MLIR

Use the `ttnn-standalone` tool to run the compiler and emit C++ code.

> 📖 See [`ttnn-standalone`](./ttnn-standalone.md) for instructions on how to generate C++ code from your MLIR input using EmitC.

### 2. Scope Down the Repro

Once you've generated the C++ code:
- Use the `ttnn-standalone` tool to run and debug it in isolation.
- Reduce the repro to the minimal example that still triggers the bug.
- Confirm the issue still reproduces reliably.

### 3. Clone the TT-Metal Repository

Clone the tt-metal repo:

```bash
git clone git@github.com:tenstorrent/tt-metal.git
cd tt-metal
```

### 4. Add the Repro to the GTest Infrastructure

Place your `.cpp` file in:

```
tests/ttnn/unit_tests/gtests/emitc/
```

and add it to the cmake file:

```
tests/ttnn/unit_tests/gtests/CMakeLists.txt
```

like so:

```cmake
set(EMITC_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/emitc/test_sanity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/emitc/your_test_name.cpp  # <<<===
)
```

Use the existing file `test_sanity.cpp` in that directory as a reference.


### 5. Modify the Repro for GTest

There are some modifications that need to be made in order to fit the GTest infra:

- Convert the `main()` function to a `TEST(...)` macro:

```cpp
TEST(EmitC, YourTestName) {
    // Your original main function body here
}
```

- Remove any `return` statements from the `TEST(...)` function body.
- Replace `#include "ttnn-precompiled.hpp"` with `#include "emitc.hpp"`

### 6. Build the TTNN EmitC Tests

First, activate the python virtual env, and set some env variables:
```bash
source python_env/bin/activate
export TT_METAL_RUNTIME_ROOT=$(pwd)
export PYTHONPATH=$(pwd)
```

Then, build the tests:

```bash
./build_metal.sh --build-ttnn-tests
```

Note: some unrelated gtests might fail here, we can ignore them.

### 7. Run the EmitC Unit Tests

To run all EmitC tests:

```bash
./build/test/ttnn/unit_tests_ttnn_emitc
```

To run a specific test:

```bash
./build/test/ttnn/unit_tests_ttnn_emitc --gtest_filter=EmitC.YourTestName
```

### 8. Share the Repro

- Create a branch with your changes.
- Open a GitHub issue or comment on an existing one.
- Link to your branch and include the instructions for running the repro

```bash
./build_metal.sh --build-ttnn-tests
./build/test/ttnn/unit_tests_ttnn_emitc
./build/test/ttnn/unit_tests_ttnn_emitc --gtest_filter=EmitC.YourTestName
```

## Known Issues

- **Missing `sfpi` compiler or other dependencies**
  If you encounter errors about a missing `sfpi` compiler or other system-level dependencies, refer to the [tt-metal installation guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md#install-system-level-dependencies) for instructions on installing the required packages.

- **TTNN test compilation failures**
  If the build fails when compiling TTNN tests, inspect the specific tests that caused the failure. If the failures are unrelated to EmitC tests, they can typically be ignored — this is a known issue.
