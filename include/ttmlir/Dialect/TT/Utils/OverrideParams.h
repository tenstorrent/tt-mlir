// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_OVERRIDE_PARAMS_H
#define TTMLIR_DIALECT_TT_OVERRIDE_PARAMS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::tt {

struct LayoutOverrideParams {
  SmallVector<int64_t, 2> grid;
  MemorySpace memorySpace;
  TensorMemoryLayout memoryLayout;
};

} // namespace mlir::tt

#endif
