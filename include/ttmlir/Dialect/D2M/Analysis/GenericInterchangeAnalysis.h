// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_GENERICINTERCHANGEANALYSIS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_GENERICINTERCHANGEANALYSIS_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

namespace mlir::tt::d2m {

struct InterchangeOptions {
  ArrayRef<int64_t> matmulInterchange = {};
};

// Returns the optimal interchange for the given GenericOp or std::nullopt if no
// interchange is beneficial / possible.
std::optional<SmallVector<int64_t>>
calculateInterchange(GenericOp op, const InterchangeOptions &options = {});

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_GENERICINTERCHANGEANALYSIS_H
