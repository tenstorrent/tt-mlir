// SPDX-FileCopyrightText: (c) 20245 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_UTILS_H
#define TTMLIR_DIALECT_TTIR_IR_UTILS_H

#include "mlir/IR/ValueRange.h"

#include <type_traits>

namespace mlir::tt::ttir {

// detect the presence of 'getOutputsMutable()' in 'Op':
template <typename Op, typename = void>
inline constexpr bool has_variadic_outputs_v = false;

template <typename Op>
inline constexpr bool has_variadic_outputs_v<
    Op, std::void_t<decltype(std::declval<Op>().getOutputsMutable())>> = true;

// A helper for simplifying DPS tablegen derivations with 'arguments' of any
// form in {AnyRankedTensor:$output, Variadic<AnyRankedTensor>:$outputs}.
//
// If a base tablegen 'class' adds this extra class declaration, derived 'def's
// don't need to override it just to switch from single to variadic type of
// '$outputs' (or vice versa):
// ...
// clang-format off
//   let extraClassDeclaration = [{
//     MutableOperandRange getDpsInitsMutable() { return ttir::getDpsOutputs(this); }
//   }]
// clang-format on
template <typename Op>
mlir::MutableOperandRange getDpsOutputs(Op *op) {
  if constexpr (has_variadic_outputs_v<Op>) {
    return op->getOutputsMutable();
  } else {
    return op->getOutputMutable();
  }
}

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_IR_UTILS_H
