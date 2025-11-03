# EmitC testing

#### NOTE: This is a developer's guide on how to test EmitC as a feature. For usage of EmitC, please refer to [`ttnn-standalone`](./ttnn-standalone.md) docs.

## Prerequisites

* [Built ttmlir](./getting-started.md)
* [Built `ttrt`](./ttrt.md#building)
* Activated virtual environment:

  ```bash
  source env/activate
  ```

* Saved system descriptor file:

  ```bash
  ttrt query --save-artifacts
  ```

### Table of Contents
1. [Generate all EmitC tests and run them](#1-generate-all-emitc-tests-and-run-them)
2. [Generate a single EmitC test and run it](#2-generate-a-single-emitc-test-and-run-it)
3. [Generate EmitC tests with Builder](#3-generate-emitc-tests-with-builder)

## Generate all EmitC tests and run them

1. Generate flatbuffers and .cpp files for EmitC tests

    If you don't have SYSTEM_DESC_PATH environment variable exported, you can run:

    ```bash
    SYSTEM_DESC_PATH=/path/to/system_desc.ttsys llvm-lit -sv test/ttmlir/EmitC/TTNN
    ```

    Or if you have SYSTEM_DESC_PATH exported, you can omit it:

    ```bash
    llvm-lit -sv test/ttmlir/EmitC/TTNN
    ```

2. Compile generated .cpp files to shared objects

    ```python
    tools/ttnn-standalone/ci_compile_dylib.py
    ```

3. Run flatbuffers + shared objects and compare results

    ```bash
    ttrt run --emitc build/test/ttmlir/EmitC/TTNN
    ```

## Generate EmitC tests with Builder
Builder offers support for building EmitPy modules from ttir or stablehlo ops. Refer to [Builder documentation](./builder/ttir-builder.md).

## Generate a single EmitC test and run it

1. Generate flatbuffers and .cpp files for EmitC test

    ```bash
    SYSTEM_DESC_PATH=/path/to/system_desc.ttsys llvm-lit -sv test/ttmlir/EmitC/TTNN/eltwise_binary/add.mlir
    ```

2. Compile generated .cpp files to shared objects

    Assuming default build directory path:
    ```bash
    tools/ttnn-standalone/ci_compile_dylib.py --file build/test/ttmlir/EmitC/TTNN/eltwise_binary/add.mlir.cpp
    ```

3. Run the flatbuffer + shared object and compare results

    ```bash
    ttrt emitc build/test/ttmlir/EmitC/TTNN/eltwise_binary/add.mlir.so --flatbuffer build/test/ttmlir/EmitC/TTNN/eltwise_binary/add.mlir.ttnn
    ```
