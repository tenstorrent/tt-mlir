// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPI_IR_SFPITRAITS_H
#define TTMLIR_DIALECT_SFPI_IR_SFPITRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::tt::sfpi::Trait {

// Trait for unary SFPI operations
template <typename ConcreteType>
class SFPIUnaryOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, SFPIUnaryOpTrait> {};

// Trait for binary SFPI operations  
template <typename ConcreteType>
class SFPIBinaryOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, SFPIBinaryOpTrait> {};

// Trait for ternary SFPI operations
template <typename ConcreteType>
class SFPITernaryOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, SFPITernaryOpTrait> {};

// Trait for SFPI comparison operations
template <typename ConcreteType>
class SFPIComparisonOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, SFPIComparisonOpTrait> {};

// Trait for SFPI type conversion operations
template <typename ConcreteType>
class SFPIConversionOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, SFPIConversionOpTrait> {};

} // namespace mlir::tt::sfpi::Trait

#endif