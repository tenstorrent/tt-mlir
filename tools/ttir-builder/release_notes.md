# `ttir-builder`

## Overview

### Release date
June 30th, 2025

### Summary
`ttir-builder` is a tool for creating TTIR operations. It provides support for MLIR modules to be generated from user-constructed ops, lowered into TTNN or TTMetal backends, and finally translated into executable flatbuffers.

### Quick links

[README.md](./README.md)
[Documentation](../../docs/src/ttir-builder.md)

### Features

#### Creating TTIR operations

The `ttir-builder` tool provides functions through the builder dataclass `TTIRBuilder` to create TTIR ops. Refer to the corresponding documentation [section](../../docs/src/ttir-builder.md#creating-a-ttir-module) for an example of using the class.

#### Op coverage

TODO: Once created, link the documented list of supported ops.

#### Build a TTIR module

The python package `ttir_builder` exposes the dataclass `TTIRBuilder`, its APIs, and utility function `build_mlir_module` that converts the object into a TTIR module. Refer to the corresponding documentation [section](../../docs/src/ttir-builder.md#creating-a-ttir-module) for usage instructions.

#### Run pipelines

There are a few pipeline passes that can be run over TTIR modules through the utility function `run_pipeline`. Passes pybound in the package `ttmlir.passes` include functionality to lower TTIR modules into TTNN and TTMetal backends. Refer to the corresponding documentation [section](../../docs/src/ttir-builder.md#running-a-pipeline) for usage instructions.

#### Compiling into flatbuffer

Utility function `compile_to_flatbuffer` compiles an function containing `TTIRBuilder` op creations through the aforementioned steps into an executable flatbuffer. Refer to the corresponding documentation [section](../../docs/src/ttir-builder.md#compiling-into-flatbuffer) for usage instructions.

#### Golden accuracy checks

`ttir-builder` provides support to code golden tensors into flatbuffers to be used for comparison with TT device output in `ttrt` runtime for accuracy verification. Refer to the corresponding documentation [section](../../docs/src/ttir-builder.md#golden-mode) for usage instructions.

## Upcoming features

TODO: Decide whether to include planned Q3 features (and cleanup/explanations):

integration with module splitter + auto generated ttir builder ops
auto generate unit tests from model tests
deprecate silicon directory and use builder as compiler + silicon CI driver
op level golden testing for CCL ops
currently we only check graph level output, but jax has a way to determine multi-device goldens per device
optimizer tests + overrides
stableHLO integration?
shardy related test sweeps
ttrt/ttir-builder can dump tons of data - we should find a way to visualize all of this in superset

## Support

TODO: Potentially add information on FAQs and where to seek customer-support

## Contributing and feedback

TODO: Potentially add information on community engagement/contributions/feedback

## License

SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
