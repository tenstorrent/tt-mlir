// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_IR_TTCORETRAITS_H
#define TTMLIR_DIALECT_TTCORE_IR_TTCORETRAITS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tt::ttcore::Trait {
template <typename Ty>
class TTCoreCreationOpTrait
    : public mlir::OpTrait::TraitBase<Ty, TTCoreCreationOpTrait> {};

template <typename Ty>
class TTCoreDuplicateConstEvalTrait
    : public mlir::OpTrait::TraitBase<Ty, TTCoreDuplicateConstEvalTrait> {};

template <typename Ty>
class TTCoreNonCacheableTrait
    : public mlir::OpTrait::TraitBase<Ty, TTCoreNonCacheableTrait> {};
} // namespace mlir::tt::ttcore::Trait

#endif
