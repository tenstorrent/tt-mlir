// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTMETAL_TTMETALTOFLATBUFFER_H
#define TTMLIR_TARGET_TTMETAL_TTMETALTOFLATBUFFER_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttmetal {

// In a shared header so that TTNN to flatbuffer can share it
struct GoldenTensor {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::uint8_t *data;
};

// Translates a TTMetal operation to a flatbuffer and writes it to the given
// stream.
LogicalResult
translateTTMetalToFlatbuffer(Operation *op, llvm::raw_ostream &os,
                             std::unordered_map<std::string, GoldenTensor> goldenMap = {});
} // namespace mlir::tt::ttmetal

#endif
