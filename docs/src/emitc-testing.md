# EmitC testing

#### NOTE: This is a developer's guide on how to test EmitC as a feature. For usage of EmitC, please refer to [`ttnn-standalone`](./ttnn-standalone.md) docs.

## Prerequisites

* [Built ttmlir](./getting-started.md)
* [Built `ttrt`](./ttrt.md#building)
* Saved system descriptor file:

  ```bash
  ttrt query --save-artifacts
  ```
* Activated virtual environment:

  ```bash
  source env/activate
  ```

## Generate EmitC tests and run it

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
