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
std::shared_ptr<void> ttnnToFlatbuffer(
    Operation *op,
    const std::unordered_map<std::string, GoldenTensor> &goldenMap = {},
    const std::vector<std::pair<std::string, std::string>> &moduleCache = {});

// Convert a TTNNIR operation to a flatbuffer
// This function signature is required in order to register the conversion in
// mlir translation framework
LogicalResult translateTTNNToFlatbuffer(
    Operation *op, llvm::raw_ostream &os,
    const std::unordered_map<std::string, GoldenTensor> &goldenMap = {},
    const std::vector<std::pair<std::string, std::string>> &moduleCache = {});
} // namespace mlir::tt::ttnn

#endif
