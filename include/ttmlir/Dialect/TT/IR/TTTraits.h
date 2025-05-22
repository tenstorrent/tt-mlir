// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_IR_TTTRAITS_H
#define TTMLIR_DIALECT_TT_IR_TTTRAITS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tt::Trait {
template <typename Ty>
class TTCreationOpTrait
    : public mlir::OpTrait::TraitBase<Ty, TTCreationOpTrait> {};

template <typename Ty>
class TTDuplicateConstEvalTrait
    : public mlir::OpTrait::TraitBase<Ty, TTDuplicateConstEvalTrait> {};
} // namespace mlir::tt::Trait

#endif
