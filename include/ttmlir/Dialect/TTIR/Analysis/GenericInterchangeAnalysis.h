// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_GENERICINTERCHANGEANALYSIS_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_GENERICINTERCHANGEANALYSIS_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

namespace mlir::tt::ttir {

struct InterchangeOptions {
  ArrayRef<int64_t> matmulInterchange = {};
};

// Returns the optimal interchange for the given GenericOp or std::nullopt if no
// interchange is beneficial / possible.
std::optional<SmallVector<int64_t>>
calculateInterchange(GenericOp op, const InterchangeOptions &options = {});

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_GENERICINTERCHANGEANALYSIS_H
