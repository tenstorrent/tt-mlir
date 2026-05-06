// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_SDPAUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_SDPAUTILS_H

#include "mlir/IR/Value.h"

#include <optional>
#include <utility>

namespace mlir::tt::ttnn::utils {

// Returns the float value of a tensor that is constant-broadcast from a
// scalar. Looks through one outer ttnn.typecast on `v`. The constant must
// originate from either a `ttnn.full` op (FillValue is a FloatAttr) or from
// a `ttcore.load_cached` op whose callee returns a `ttnn.full` value.
//
// Returns std::nullopt for any other producer.
std::optional<float> extractScalarConstant(Value v);

// If `v` is the result of a `ttnn.multiply` whose other input is a scalar
// constant (per `extractScalarConstant`), returns `{nonScalarInput, scalar}`.
// Otherwise returns `{v, nullopt}`.
//
// `nonScalarInput` is the multiply's other input. Substituting it for the
// multiply's result is shape-safe whenever the constant tensor broadcasts to
// `nonScalarInput`'s shape (e.g. when it is `1x...x1`); callers that rely on
// that property must ensure it themselves — the helper does not check.
std::pair<Value, std::optional<float>>
extractMultiplyWithScalarConstant(Value v);

} // namespace mlir::tt::ttnn::utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_SDPAUTILS_H
