// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/TTTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::tt {

MlirAttribute ttmlirTTOperandConstraintAttrGet(MlirContext ctx,
                                               uint32_t operandConstraint) {
  return wrap(OperandConstraintAttr::get(
      unwrap(ctx), static_cast<OperandConstraint>(operandConstraint)));
}

} // namespace mlir::tt
