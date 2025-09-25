// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_IR_D2MOPSINTERFACES_H
#define TTMLIR_DIALECT_D2M_IR_D2MOPSINTERFACES_H

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

namespace mlir::tt::d2m {

namespace detail {
mlir::LogicalResult verifyGenericParent(mlir::Operation *op);
} // namespace detail

std::pair<mlir::MemRefType, mlir::AffineMap> applyViews(mlir::Operation *op);

} // namespace mlir::tt::d2m

#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h.inc"

#endif
