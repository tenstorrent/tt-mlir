// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTKERNEL_IR_TTKERNELTRAITS_H
#define TTMLIR_DIALECT_TTKERNEL_IR_TTKERNELTRAITS_H

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::tt::ttkernel {

template <typename ConcreteType>
class TTKernelFPUOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelFPUOpTrait> {};

template <typename ConcreteType>
class TTKernelSFPUOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelSFPUOpTrait> {};

template <typename ConcreteType>
class TTKernelInitOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelInitOpTrait> {};

template <typename ConcreteType>
class TTKernelUnaryOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelUnaryOpTrait> {
public:
  static constexpr int arity = 1;
};

template <typename ConcreteType>
class TTKernelBinaryOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelBinaryOpTrait> {
public:
  static constexpr int arity = 2;
};

template <typename ConcreteType>
class TTKernelTernaryOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelTernaryOpTrait> {
public:
  static constexpr int arity = 3;
};

template <typename ConcreteType>
class TTKernelDstAccumulatingOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      TTKernelDstAccumulatingOpTrait> {};

template <typename ConcreteType>
class TTKernelDeviceZoneOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelDeviceZoneOpTrait> {
};

template <typename ConcreteType>
class TTKernelTridNocOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelTridNocOpTrait> {
public:
  static constexpr int32_t kMaxTrid = 15;
  static constexpr int32_t kNumNocs = 2;

  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    auto concreteOp = mlir::cast<ConcreteType>(op);

    auto tridValue = getConstantIntValue(concreteOp.getTrid());
    if (tridValue && (*tridValue < 0 || *tridValue > kMaxTrid)) {
      return op->emitOpError() << "trid must be in [0, " << kMaxTrid << "].";
    }

    mlir::Value noc = concreteOp.getNoc();
    if (noc) {
      auto nocValue = getConstantIntValue(noc);
      if (nocValue && (*nocValue < 0 || *nocValue >= kNumNocs)) {
        return op->emitOpError()
               << "noc must be in [0, " << (kNumNocs - 1) << "].";
      }
    }

    return mlir::success();
  }
};

} // namespace mlir::tt::ttkernel

#endif
