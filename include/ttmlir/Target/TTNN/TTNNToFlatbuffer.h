// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTNN_TTNNTOFLATBUFFER_H
#define TTMLIR_TARGET_TTNN_TTNNTOFLATBUFFER_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

namespace mlir::tt::ttnn {

// Convert a TTNNIR operation to a flatbuffer
std::shared_ptr<void>
ttnnToFlatbuffer(Operation *op,
                 std::unordered_map<std::string, GoldenTensor> goldenMap = {});

// Convert a TTNNIR operation to a flatbuffer
// This function signature is required in order to register the conversion in
// mlir translation framework
LogicalResult translateTTNNToFlatbuffer(
    Operation *op, llvm::raw_ostream &os,
    std::unordered_map<std::string, GoldenTensor> goldenMap = {});
} // namespace mlir::tt::ttnn

#endif
