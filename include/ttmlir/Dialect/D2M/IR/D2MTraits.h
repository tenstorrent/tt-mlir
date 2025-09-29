// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_IR_D2MTRAITS_H
#define TTMLIR_DIALECT_D2M_IR_D2MTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::tt::d2m {
namespace impl {
mlir::LogicalResult verifyGenericRegionComputeOp(mlir::Operation *op);
mlir::LogicalResult verifyGenericRegionDatamovementOp(mlir::Operation *op);
} // namespace impl

// Trait for operations that must be in compute regions
template <typename ConcreteType>
struct D2MGenericRegionComputeOpTrait
    : public OpTrait::TraitBase<ConcreteType, D2MGenericRegionComputeOpTrait> {
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return impl::verifyGenericRegionComputeOp(op);
  }
};

// Trait for operations that must be in datamovement regions
template <typename ConcreteType>
struct D2MGenericRegionDatamovementOpTrait
    : public OpTrait::TraitBase<ConcreteType,
                                D2MGenericRegionDatamovementOpTrait> {
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return impl::verifyGenericRegionDatamovementOp(op);
  }
};

template <typename ConcreteType>
struct D2MSkipOpEltWiseFusionTrait
    : public OpTrait::TraitBase<ConcreteType, D2MSkipOpEltWiseFusionTrait> {
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_IR_D2MTRAITS_H
