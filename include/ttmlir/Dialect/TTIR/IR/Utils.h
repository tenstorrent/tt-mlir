// SPDX-FileCopyrightText: (c) 20245 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_UTILS_H
#define TTMLIR_DIALECT_TTIR_IR_UTILS_H

#include <mlir/IR/ValueRange.h>

#include <type_traits>

namespace mlir::tt::ttir {

// detect the presence of 'getOutputsMutable()' in 'Op':
template <typename Op, typename = void>
inline constexpr bool has_variadic_outputs = false;

template <typename Op>
inline constexpr bool has_variadic_outputs<
    Op, std::void_t<decltype(std::declval<Op>().getOutputsMutable())>> = true;

namespace impl {

template <typename Op, typename = void>
struct getDpsOutputs {
  static mlir::MutableOperandRange evaluate(Op *op) {
    return op->getOutputMutable();
  }
};

template <typename Op>
struct getDpsOutputs<Op, std::enable_if_t<has_variadic_outputs<Op>>> {
  static mlir::MutableOperandRange evaluate(Op *op) {
    return op->getOutputsMutable();
  }
};

} // namespace impl

// A helper for simplifying DPS tablegen derivations with 'arguments' of any
// form in {AnyRankedTensor:$output, Variadic<AnyRankedTensor>:$outputs}.
//
// If a base tablegen 'class' adds this extra class declaration, derived 'def's
// don't need to overrride it just to switch from single to variadic type of
// '$outputs' (or vice versa):
// ...
// clang-format off
//   let extraClassDeclaration = [{
//     MutableOperandRange getDpsInitsMutable() { return ttir::getDpsOutputs(this); }
//   }]
// clang-format on
template <typename Op>
mlir::MutableOperandRange getDpsOutputs(Op *op) {
  return impl::getDpsOutputs<Op>::evaluate(op);
}

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_IR_UTILS_H
