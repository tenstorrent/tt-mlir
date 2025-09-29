// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTMETAL_TTMETALTOFLATBUFFER_H
#define TTMLIR_TARGET_TTMETAL_TTMETALTOFLATBUFFER_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

namespace mlir::tt::ttmetal {

// Translates a TTMetal operation to a flatbuffer and writes it to the given
// stream.
LogicalResult translateTTMetalToFlatbuffer(
    Operation *op, llvm::raw_ostream &os,
    /* goldenMap has following structure
    {
      loc: {
        device_id: GoldenTensor
      }
    }
    */
    const std::unordered_map<std::string,
                             std::unordered_map<std::uint32_t, GoldenTensor>>
        &goldenMap = {},
    const std::vector<std::pair<std::string, std::string>> &moduleCache = {});

// Translates a TTMetal operation to a flatbuffer and returns a pointer to
// in-memory blob.
std::shared_ptr<void> translateTTMetalToFlatbuffer(
    Operation *op,
    const std::unordered_map<std::string,
                             std::unordered_map<std::uint32_t, GoldenTensor>>
        &goldenMap = {},
    const std::vector<std::pair<std::string, std::string>> &moduleCache = {});

} // namespace mlir::tt::ttmetal

#endif
