// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_IR_D2MTRAITS_H
#define TTMLIR_DIALECT_D2M_IR_D2MTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::tt::d2m {

// Minimal trait stubs for D2M generic region ops. We can add verification
// later.
template <typename ConcreteType>
struct D2MGenericRegionComputeOpTrait
    : public OpTrait::TraitBase<ConcreteType, D2MGenericRegionComputeOpTrait> {
  static mlir::LogicalResult verifyTrait(mlir::Operation *) {
    return mlir::success();
  }
};

template <typename ConcreteType>
struct D2MGenericRegionDatamovementOpTrait
    : public OpTrait::TraitBase<ConcreteType,
                                D2MGenericRegionDatamovementOpTrait> {
  static mlir::LogicalResult verifyTrait(mlir::Operation *) {
    return mlir::success();
  }
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_IR_D2MTRAITS_H
