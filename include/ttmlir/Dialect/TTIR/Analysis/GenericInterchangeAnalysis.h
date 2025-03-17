// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_GENERICINTERCHANGEANALYSIS_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_GENERICINTERCHANGEANALYSIS_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

namespace mlir::tt::ttir {

SmallVector<int64_t> calculateOptimalInterchange(GenericOp op);

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_GENERICINTERCHANGEANALYSIS_H
