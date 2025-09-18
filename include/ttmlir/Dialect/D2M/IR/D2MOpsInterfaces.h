// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_IR_D2MOPSINTERFACES_H
#define TTMLIR_DIALECT_D2M_IR_D2MOPSINTERFACES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

namespace mlir::tt::d2m {

namespace detail {
mlir::LogicalResult verifyGenericParent(mlir::Operation *op);
}

// Expose applyViews for default interface implementations
std::pair<mlir::MemRefType, mlir::AffineMap> applyViews(mlir::Operation *op);

} // namespace mlir::tt::d2m

#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h.inc"

#endif // TTMLIR_DIALECT_D2M_IR_D2MOPSINTERFACES_H
