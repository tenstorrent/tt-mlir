// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_UTILS_OVERRIDEPARAMS_H
#define TTMLIR_DIALECT_TT_UTILS_OVERRIDEPARAMS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include <cstdint>

namespace mlir::tt {

struct InputLayoutOverrideParams {
  SmallVector<int64_t> operandIdxes;
};

struct OutputLayoutOverrideParams {
  SmallVector<int64_t, 2> grid;
  MemorySpace memorySpace;
  TensorMemoryLayout memoryLayout;
};

} // namespace mlir::tt

#endif
