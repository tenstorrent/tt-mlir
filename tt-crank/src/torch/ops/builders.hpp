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

namespace tt::kurbla::torch_backend {

using ::tt::kurbla::ModuleBuilder;
using ::tt::kurbla::TensorTypeSpec;

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

using ::tt::kurbla::broadcast_shape;
using ::tt::kurbla::build_add;
using ::tt::kurbla::build_addmm;
using ::tt::kurbla::build_all;
using ::tt::kurbla::build_all_gather;
using ::tt::kurbla::build_all_reduce;
using ::tt::kurbla::build_any;
using ::tt::kurbla::build_arange;
using ::tt::kurbla::build_argmax;
using ::tt::kurbla::build_bitwise_and;
using ::tt::kurbla::build_bitwise_not;
using ::tt::kurbla::build_bitwise_or;
using ::tt::kurbla::build_bn_inference;
using ::tt::kurbla::build_broadcast;
using ::tt::kurbla::build_cat;
using ::tt::kurbla::build_clamp;
using ::tt::kurbla::build_conv2d;
using ::tt::kurbla::build_cos;
using ::tt::kurbla::build_cumsum;
using ::tt::kurbla::build_div;
using ::tt::kurbla::build_embedding;
using ::tt::kurbla::build_eq;
using ::tt::kurbla::build_exp;
using ::tt::kurbla::build_floor_divide;
using ::tt::kurbla::build_full;
using ::tt::kurbla::build_gather;
using ::tt::kurbla::build_ge;
using ::tt::kurbla::build_gelu;
using ::tt::kurbla::build_gt;
using ::tt::kurbla::build_index_copy;
using ::tt::kurbla::build_isneginf;
using ::tt::kurbla::build_le;
using ::tt::kurbla::build_linear;
using ::tt::kurbla::build_linear_backward;
using ::tt::kurbla::build_log;
using ::tt::kurbla::build_log1p;
using ::tt::kurbla::build_logical_and;
using ::tt::kurbla::build_logical_not;
using ::tt::kurbla::build_logical_or;
using ::tt::kurbla::build_lt;
using ::tt::kurbla::build_matmul;
using ::tt::kurbla::build_matmul_backward;
using ::tt::kurbla::build_max_pool2d;
using ::tt::kurbla::build_mean;
using ::tt::kurbla::build_mm;
using ::tt::kurbla::build_mse_loss;
using ::tt::kurbla::build_mse_loss_backward;
using ::tt::kurbla::build_mul;
using ::tt::kurbla::build_ne;
using ::tt::kurbla::build_neg;
using ::tt::kurbla::build_ones;
using ::tt::kurbla::build_pad;
using ::tt::kurbla::build_permute;
using ::tt::kurbla::build_pow;
using ::tt::kurbla::build_reduce;
using ::tt::kurbla::build_reduce_scatter;
using ::tt::kurbla::build_relu;
using ::tt::kurbla::build_reshape;
using ::tt::kurbla::build_rsqrt;
using ::tt::kurbla::build_scalar;
using ::tt::kurbla::build_sdpa;
using ::tt::kurbla::build_sigmoid;
using ::tt::kurbla::build_silu;
using ::tt::kurbla::build_sin;
using ::tt::kurbla::build_slice;
using ::tt::kurbla::build_softmax;
using ::tt::kurbla::build_sqrt;
using ::tt::kurbla::build_squeeze;
using ::tt::kurbla::build_sub;
using ::tt::kurbla::build_sum;
using ::tt::kurbla::build_sum_to;
using ::tt::kurbla::build_t;
using ::tt::kurbla::build_threshold_backward;
using ::tt::kurbla::build_transpose;
using ::tt::kurbla::build_tril;
using ::tt::kurbla::build_unsqueeze;
using ::tt::kurbla::build_where;
using ::tt::kurbla::build_zeros;
using ::tt::kurbla::scale_tensor;

} // namespace tt::kurbla::torch_backend
