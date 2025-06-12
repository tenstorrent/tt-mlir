// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_TTIROPSINTERFACES_H
#define TTMLIR_DIALECT_TTIR_IR_TTIROPSINTERFACES_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

namespace mlir::tt::ttir {

namespace detail {
mlir::LogicalResult verifyGenericParent(mlir::Operation *op);
} // namespace detail

std::pair<mlir::MemRefType, mlir::AffineMap> applyViews(mlir::Operation *op);

} // namespace mlir::tt::ttir

#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h.inc"

#endif
