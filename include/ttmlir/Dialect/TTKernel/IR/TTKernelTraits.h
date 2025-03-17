// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_TTKERNELTRAITS_H
#define TTMLIR_DIALECT_TTIR_IR_TTKERNELTRAITS_H

namespace mlir {
namespace tt {
namespace ttkernel {
namespace OpTrait {

template <typename ConcreteType>
class TTKernelFPUOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelFPUOpTrait> {};

template <typename ConcreteType>
class TTKernelSFPUOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelSFPUOpTrait> {};

template <typename ConcreteType>
class TTKernelInitOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, TTKernelInitOpTrait> {};

} // namespace OpTrait
} // namespace ttkernel
} // namespace tt
} // namespace mlir

#endif
