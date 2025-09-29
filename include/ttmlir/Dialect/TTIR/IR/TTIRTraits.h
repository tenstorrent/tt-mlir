// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_TTIRTRAITS_H
#define TTMLIR_DIALECT_TTIR_IR_TTIRTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::tt::ttir {
namespace impl {
bool verifyInvolution(mlir::Operation *op);
bool verifyIdempotence(mlir::Operation *op);
bool verifyBinaryIdempotence(mlir::Operation *op);
mlir::LogicalResult verifyGenericRegionComputeOp(mlir::Operation *op);
mlir::LogicalResult verifyGenericRegionDatamovementOp(mlir::Operation *op);
mlir::LogicalResult verifyBroadcastable(mlir::Operation *op);
mlir::OpFoldResult foldInvolution(mlir::Operation *op);
mlir::OpFoldResult foldIdempotence(mlir::Operation *op);
mlir::OpFoldResult foldBinaryIdempotence(mlir::Operation *op);
} // namespace impl

template <typename ConcreteType>
class TTIRInvolution
    : public mlir::OpTrait::TraitBase<ConcreteType, TTIRInvolution> {
public:
  static mlir::LogicalResult foldTrait(mlir::Operation *op, ArrayRef<Attribute>,
                                       SmallVectorImpl<OpFoldResult> &results) {
    if (!impl::verifyInvolution(op)) {
      return mlir::failure();
    }

    results.push_back(impl::foldInvolution(op));
    return mlir::success();
  }
};

template <typename ConcreteType>
class TTIRIdempotence
    : public mlir::OpTrait::TraitBase<ConcreteType, TTIRIdempotence> {
public:
  static mlir::LogicalResult foldTrait(mlir::Operation *op, ArrayRef<Attribute>,
                                       SmallVectorImpl<OpFoldResult> &results) {
    if (!impl::verifyIdempotence(op)) {
      return mlir::failure();
    }

    results.push_back(impl::foldIdempotence(op));
    return mlir::success();
  }
};

template <typename ConcreteType>
class TTIRBinaryIdempotence
    : public mlir::OpTrait::TraitBase<ConcreteType, TTIRBinaryIdempotence> {
public:
  static mlir::LogicalResult foldTrait(mlir::Operation *op, ArrayRef<Attribute>,
                                       SmallVectorImpl<OpFoldResult> &results) {
    if (!impl::verifyBinaryIdempotence(op)) {
      return mlir::failure();
    }

    results.push_back(impl::foldBinaryIdempotence(op));
    return mlir::success();
  }
};

template <typename ConcreteType>
struct TTIRGenericRegionComputeOpTrait
    : public OpTrait::TraitBase<ConcreteType, TTIRGenericRegionComputeOpTrait> {
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return impl::verifyGenericRegionComputeOp(op);
  }
};

template <typename ConcreteType>
struct TTIRGenericRegionDatamovementOpTrait
    : public OpTrait::TraitBase<ConcreteType,
                                TTIRGenericRegionDatamovementOpTrait> {
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return impl::verifyGenericRegionDatamovementOp(op);
  }
};

template <typename ConcreteType>
struct Broadcastable : public OpTrait::TraitBase<ConcreteType, Broadcastable> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return impl::verifyBroadcastable(op);
  }
};

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_IR_TTIRTRAITS_H
