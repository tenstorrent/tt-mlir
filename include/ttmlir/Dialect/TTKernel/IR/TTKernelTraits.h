// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTKERNEL_IR_TTKERNELTRAITS_H
#define TTMLIR_DIALECT_TTKERNEL_IR_TTKERNELTRAITS_H

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
class TTKernelDeviceZoneOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelDeviceZoneOpTrait> {
};

} // namespace mlir::tt::ttkernel

#endif
