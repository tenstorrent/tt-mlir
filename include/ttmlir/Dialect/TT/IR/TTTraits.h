// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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
class TTIgnoreConstEvalTrait
    : public mlir::OpTrait::TraitBase<Ty, TTIgnoreConstEvalTrait> {};

template <typename Ty>
class TTForkConstEvalTrait
    : public mlir::OpTrait::TraitBase<Ty, TTForkConstEvalTrait> {};
} // namespace mlir::tt::Trait

#endif
