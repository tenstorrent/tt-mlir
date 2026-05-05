// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_IR_D2MGENERICREGIONOPSINTERFACES_H
#define TTMLIR_DIALECT_D2M_IR_D2MGENERICREGIONOPSINTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::tt::d2m::detail {

// Verifies that the op is nested inside a d2m.generic or func.func
LogicalResult verifyGenericParent(Operation *op);

} // namespace mlir::tt::d2m::detail

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOpsInterfaces.h.inc"

#endif // TTMLIR_DIALECT_D2M_IR_D2MGENERICREGIONOPSINTERFACES_H
