// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_TTIROPSINTERFACES_H
#define TTMLIR_DIALECT_TTIR_IR_TTIROPSINTERFACES_H

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

namespace mlir {
namespace tt {
namespace ttir {
namespace detail {
mlir::LogicalResult verifyElementwiseOp(mlir::Operation *op);
} // namespace detail
} // namespace ttir
} // namespace tt
} // namespace mlir

#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h.inc"

#endif
