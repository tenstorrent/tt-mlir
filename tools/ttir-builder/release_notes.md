# `ttir-builder`

## Overview

Release Date: June 30th, 2025

Version: v1.0.0

### Summary
`ttir-builder` is a tool for creating TTIR operations. It provides support for MLIR modules to be generated from user-constructed ops, lowered into TTNN or TTMetal backends, and finally translated into executable flatbuffers.

### Quick links
[README.md](./README.md)
[Documentation](../../docs/src/ttir-builder.md)

### Features

#### Creating TTIR operations
The `ttir-builder` tool provides functions through the builder class `TTIRBuilder` to create TTIR ops. Refer to the corresponding documentation [section](../../docs/src/ttir-builder.md#creating-a-ttir-module) for an example of using the class.

#### Build a TTIR module
The python package `ttir_builder` exposes the class `TTIRBuilder`, its APIs, and utility function `build_mlir_module` that converts the object into a TTIR module. Refer to the corresponding documentation [section](../../docs/src/ttir-builder.md#creating-a-ttir-module) for usage instructions.

#### Run pipelines
There are a few pipeline passes that can be run over TTIR modules through the utility function `run_pipeline`. Passes pybound in the package `ttmlir.passes` include functionality to lower TTIR modules into TTNN and TTMetal backends. Refer to the corresponding documentation [section](../../docs/src/ttir-builder.md#running-a-pipeline) for usage instructions.

#### Compiling into flatbuffer
Utility function `compile_to_flatbuffer` compiles an function containing `TTIRBuilder` op creations through the aforementioned steps into an executable flatbuffer. Refer to the corresponding documentation [section](../../docs/src/ttir-builder.md#compiling-into-flatbuffer) for usage instructions.

#### Golden accuracy checks
`ttir-builder` provides support to code golden tensors into flatbuffers to be used for comparison with TT device output in `ttrt` runtime for accuracy verification. Refer to the corresponding documentation [section](../../docs/src/ttir-builder.md#golden-mode) for usage instructions.

## License
SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
