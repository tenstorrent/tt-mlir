// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header re-exports ttir builders under the torch backend's namespace and adds
// the ATen-typed conveniences the torch kernels use.

#include <tuple>
#include <type_traits>
#include <utility>

#include <ATen/core/ScalarType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TypeProperties.h>

#include "engine/ttir_module_builder.hpp"

namespace tt::crank::torch_backend {

using ::tt::crank::ModuleBuilder;
using ::tt::crank::TensorTypeSpec;

// Spec for a TTIR function input, carrying the tensor's logical dtype.
// The TTIR→TTNN rewriter demotes unsupported wide types (f64, i64, ...)
// to their hardware alias at lowering time.
TensorTypeSpec spec_for(const at::Tensor &t);

// Torch scalar type → MLIR element type via the logical runtime dtype. May
// return a type the hardware doesn't support directly; the rewriter handles it.
mlir::Type mlir_element_type_for(c10::ScalarType torch_dtype);

// Kernel-side convenience for native binary/ternary ops. Computes the
// PyTorch-promoted dtype across `tensors` (using `at::result_type` semantics —
// wrapped-scalar handling, etc.) and emits a `ttir.typecast` for each builder
// arg that doesn't already match. Returns the promoted dtype followed by the
// cast values, for structured binding:
//
//     auto [promoted, lhs, rhs] = promote_inputs(mb, a, b);
//
// `tensors` must be in the same order they were passed to `ModuleBuilder::init`.
template <typename... Tensors> auto promote_inputs(ModuleBuilder &mb, const Tensors &...tensors) {
    static_assert(sizeof...(Tensors) > 0, "promote_inputs: at least one input required");
    static_assert((std::is_same_v<std::remove_cvref_t<Tensors>, at::Tensor> && ...),
                  "promote_inputs: all arguments must be at::Tensor");

    at::native::ResultTypeState state{};
    ((state = at::native::update_result_type_state(tensors, state)), ...);
    const c10::ScalarType promoted = at::native::result_type(state);
    const auto promoted_mlir = mlir_element_type_for(promoted);

    auto args = mb.args();
    TORCH_CHECK(args.size() == sizeof...(Tensors), "promote_inputs: ModuleBuilder has ", args.size(),
                " arg(s), expected ", sizeof...(Tensors));

    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::tuple{promoted, mb.insert_typecast(args[Is], promoted_mlir)...};
    }(std::make_index_sequence<sizeof...(Tensors)>{});
}

using ::tt::crank::broadcast_shape;
using ::tt::crank::build_add;
using ::tt::crank::build_addmm;
using ::tt::crank::build_all;
using ::tt::crank::build_all_gather;
using ::tt::crank::build_all_reduce;
using ::tt::crank::build_any;
using ::tt::crank::build_arange;
using ::tt::crank::build_argmax;
using ::tt::crank::build_bitwise_and;
using ::tt::crank::build_bitwise_not;
using ::tt::crank::build_bitwise_or;
using ::tt::crank::build_bn_inference;
using ::tt::crank::build_broadcast;
using ::tt::crank::build_cat;
using ::tt::crank::build_clamp;
using ::tt::crank::build_conv1d;
using ::tt::crank::build_conv2d;
using ::tt::crank::build_conv3d;
using ::tt::crank::build_cos;
using ::tt::crank::build_cumsum;
using ::tt::crank::build_div;
using ::tt::crank::build_embedding;
using ::tt::crank::build_embedding_backward;
using ::tt::crank::build_eq;
using ::tt::crank::build_exp;
using ::tt::crank::build_floor_divide;
using ::tt::crank::build_full;
using ::tt::crank::build_gather;
using ::tt::crank::build_ge;
using ::tt::crank::build_gelu;
using ::tt::crank::build_gt;
using ::tt::crank::build_index_copy;
using ::tt::crank::build_isneginf;
using ::tt::crank::build_layer_norm;
using ::tt::crank::build_layer_norm_with_stats;
using ::tt::crank::build_le;
using ::tt::crank::build_linear;
using ::tt::crank::build_linear_backward;
using ::tt::crank::build_log;
using ::tt::crank::build_log1p;
using ::tt::crank::build_logical_and;
using ::tt::crank::build_logical_not;
using ::tt::crank::build_logical_or;
using ::tt::crank::build_lt;
using ::tt::crank::build_matmul;
using ::tt::crank::build_matmul_backward;
using ::tt::crank::build_max_pool2d;
using ::tt::crank::build_mean;
using ::tt::crank::build_mm;
using ::tt::crank::build_mse_loss;
using ::tt::crank::build_mse_loss_backward;
using ::tt::crank::build_mul;
using ::tt::crank::build_ne;
using ::tt::crank::build_neg;
using ::tt::crank::build_ones;
using ::tt::crank::build_pad;
using ::tt::crank::build_permute;
using ::tt::crank::build_pow;
using ::tt::crank::build_reciprocal;
using ::tt::crank::build_reduce;
using ::tt::crank::build_reduce_scatter;
using ::tt::crank::build_relu;
using ::tt::crank::build_reshape;
using ::tt::crank::build_rsqrt;
using ::tt::crank::build_scalar;
using ::tt::crank::build_sdpa;
using ::tt::crank::build_sigmoid;
using ::tt::crank::build_silu;
using ::tt::crank::build_sin;
using ::tt::crank::build_slice;
using ::tt::crank::build_softmax;
using ::tt::crank::build_sqrt;
using ::tt::crank::build_squeeze;
using ::tt::crank::build_sub;
using ::tt::crank::build_sum;
using ::tt::crank::build_sum_to;
using ::tt::crank::build_t;
using ::tt::crank::build_tanh;
using ::tt::crank::build_threshold_backward;
using ::tt::crank::build_transpose;
using ::tt::crank::build_tril;
using ::tt::crank::build_unsqueeze;
using ::tt::crank::build_vector_norm;
using ::tt::crank::build_where;
using ::tt::crank::build_zeros;
using ::tt::crank::scale_tensor;

} // namespace tt::crank::torch_backend
