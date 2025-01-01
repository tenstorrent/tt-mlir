// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTKERNEL_PIPELINES_TTKERNELPIPELINES_H
#define TTMLIR_DIALECT_TTKERNEL_PIPELINES_TTKERNELPIPELINES_H

#include "mlir/Pass/PassOptions.h"

namespace mlir::tt::ttkernel {
// Options for the TTKernel to EmitC pipeline.
//
// struct TTIRToTTMetalBackendPipelineOptions
//     : public PassPipelineOptions<TTIRToTTMetalBackendPipelineOptions> {
//   ListOption<int64_t> meshShape{
//       *this, "mesh-shape", llvm::cl::desc("Set the multi-device mesh
//       shape.")};

//   // Option to provide a system descriptor flatbuffer file to compile
//   // against.
//   //
//   Option<std::string> systemDescPath{
//       *this, "system-desc-path",
//       llvm::cl::desc(
//           "Pass in a system descriptor flatbuffer to compile against."),
//       llvm::cl::init("")};
// };

void createTTKernelToEmitCPipeline(OpPassManager &pm);

void registerTTKernelPipelines();
} // namespace mlir::tt::ttkernel

#endif
