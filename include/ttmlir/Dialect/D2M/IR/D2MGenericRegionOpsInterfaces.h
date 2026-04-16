// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_IR_D2MGENERICREGIONOPSINTERFACES_H
#define TTMLIR_DIALECT_D2M_IR_D2MGENERICREGIONOPSINTERFACES_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOpsInterfaces.h.inc"

namespace mlir::tt::d2m {
struct CBUsageInfo {
  SmallVector<Operation *> producers;
  SmallVector<Operation *> consumers;
};

llvm::DenseMap<Value, CBUsageInfo> getCBUsageInfo(Region &region);
} // namespace mlir::tt::d2m

#endif
