// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_TTIRTRAITS_H
#define TTMLIR_DIALECT_TTIR_IR_TTIRTRAITS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace tt {
namespace ttir {
namespace OpTrait {

namespace impl {
bool verifyInvolution(mlir::Operation *op);
bool verifyIdempotence(mlir::Operation *op);
bool verifyBinaryIdempotence(mlir::Operation *op);
mlir::OpFoldResult foldInvolution(mlir::Operation *op);
mlir::OpFoldResult foldIdempotence(mlir::Operation *op);
mlir::OpFoldResult foldBinaryIdempotence(mlir::Operation *op);
} // namespace impl

template <typename ConcreteType>
class TTIRInvolution
    : public mlir::TypeTrait::TraitBase<ConcreteType, TTIRInvolution> {
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
    : public mlir::TypeTrait::TraitBase<ConcreteType, TTIRIdempotence> {
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
    : public mlir::TypeTrait::TraitBase<ConcreteType, TTIRBinaryIdempotence> {
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
class TTIRGenericRegionOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTIRGenericRegionOpTrait> {
};

namespace named_op_group {

struct elementwise {
  using named_op_group_type = elementwise;
};
struct reduction {
  using named_op_group_type = reduction;
};
struct contraction {
  using named_op_group_type = contraction;
}; // matmul-like

} // namespace named_op_group

template <typename ConcreteType>
struct TTIRNamedElementwise
    : named_op_group::elementwise,
      mlir::TypeTrait::TraitBase<ConcreteType, TTIRNamedElementwise> {};

template <typename ConcreteType>
struct TTIRNamedReduction
    : named_op_group::reduction,
      mlir::TypeTrait::TraitBase<ConcreteType, TTIRNamedReduction> {};

template <typename ConcreteType>
struct TTIRNamedContraction
    : named_op_group::contraction,
      mlir::TypeTrait::TraitBase<ConcreteType, TTIRNamedContraction> {};

} // namespace OpTrait
} // namespace ttir
} // namespace tt
} // namespace mlir

#endif // TTMLIR_DIALECT_TTIR_IR_TTIRTRAITS_H
