// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_GPU_GPUTOFLATBUFFER_H
#define TTMLIR_TARGET_GPU_GPUTOFLATBUFFER_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

namespace mlir::tt::gpu {

std::string translateToPTX(Operation *op);

// Convert a GPU LLVMIR operation to a flatbuffer
std::shared_ptr<void> gpuToFlatbuffer(Operation *op);

// Convert a GPU LLVMIR operation to a flatbuffer
// This function signature is required in order to register the conversion in
// mlir translation framework
LogicalResult translateGPUToFlatbuffer(Operation *op, llvm::raw_ostream &os);
} // namespace mlir::tt::gpu

#endif
