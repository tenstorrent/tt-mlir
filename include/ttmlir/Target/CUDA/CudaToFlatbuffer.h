// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_CUDA_CUDATOFLATBUFFER_H
#define TTMLIR_TARGET_CUDA_CUDATOFLATBUFFER_H

#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::cuda {

// Compile a GPU operation to PTX.
llvm::Expected<std::string> translateToPTX(Operation *op,
                                           const std::string &mcpu = "sm_50");

// Convert a GPU LLVMIR operation to a flatbuffer.
std::shared_ptr<void> cudaToFlatbuffer(Operation *op);

// Convert a GPU LLVMIR operation to a flatbuffer.
// This function signature is required in order to register the conversion in
// mlir translation framework.
LogicalResult translateCudaToFlatbuffer(Operation *op, llvm::raw_ostream &os);
} // namespace mlir::tt::cuda

#endif
