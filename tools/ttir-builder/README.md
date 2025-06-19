# `ttir-builder`

`ttir-builder` is a tool for creating TTIR operations from pure python. It provides support for MLIR modules to be generated from user-constructed ops, lowered into TTNN or TTMetal backends, and translated into flatbuffers executable by `ttrt`.

## Build
1. Build [ttmlir](./getting-started.md)
2. Build [`ttrt`](./ttrt.md#building)
3. Generate `.ttsys` file from the system you want to compile for using `ttrt` (Shown below). This will create a `ttrt-artifacts` folder containing a `system_desc.ttsys` file.
```bash
ttrt query --save-artifacts
```
4. Export this file in your environment using `export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys`. `ttir_builder.utils` uses the `system_desc.ttsys` file as it runs a pass over an MLIR module to lower into the TTNN or TTMetal backend.

## Usage
`TTIRBuilder` is implemented through python packages.
```bash
import ttir_builder
import ttir_builder.utils
```
To get started, refer to `ttir-builder` [documentation](../../docs/src/ttir-builder.md#getting-started).
