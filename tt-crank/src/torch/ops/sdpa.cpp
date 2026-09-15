// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SDPA integration for the tt (PrivateUse1) backend.
//
// `scaled_dot_product_attention` is CompositeImplicitAutograd: it asks the
// `_fused_sdp_choice_stub` DispatchStub (keyed on `query.device().type()`) which
// fused backend to use, then dispatches to the matching op. The stub ships no
// PrivateUse1 kernel, so `is_device_supported(PrivateUse1)` is false and the
// composite falls back to MATH, decomposing to matmul+softmax during tracing
// (see PyTorch issue #162989) - there is no atomic op left for the compile path
// to lower to a single `ttir.sdpa`, and under DTensor the head-sharded Q/K/V get
// all-gathered.
//
// Registering the *stub* for PrivateUse1 (the composite consults the stub directly,
// not the `aten::_fused_sdp_choice` operator) makes the composite pick OVERRIDEABLE
// and emit the atomic `_scaled_dot_product_fused_attention_overrideable`. The compile
// path lowers that (see _compile.py); eager runs it through the kernels below. This
// mirrors PyTorch's in-tree `openreg` backend.
//
// Inference uses `ttir.scaled_dot_product_attention`; training the ttml `sdpa_fw`/`sdpa_bw`
// composites, or the MATH decomposition where ttml can't run (see ttml_sdpa_supported).

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "mlir/IR/Value.h"
#include <ATen/ATen.h>
#include <ATen/SDPBackend.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <torch/library.h>

#include "cast.hpp"
#include "torch/backend.hpp"
#include "torch/ops/builders.hpp"
#include "torch/tensor.hpp"

namespace tt::crank::torch_backend {

namespace {

// The sdpa composite turns a bool mask into a 0/-inf float one before calling the op; only the choice stub
// sees the original, so it records here whether the mask the op receives next came from a bool one.
thread_local bool g_sdpa_mask_from_bool = false;

// ttml's `arbitrary` mask is one [1, 1, S, S] keep-mask for all batches/heads (HF's padded causal mask at
// batch 1); genuine float masks and per-batch/head masks decompose.
bool ttml_sdpa_supported(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                         const std::optional<at::Tensor> &attn_mask, bool mask_from_bool) {
    if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4 || query.scalar_type() != at::kBFloat16) {
        return false;
    }
    const int64_t seq = query.size(2);
    if (attn_mask.has_value() && attn_mask->defined() &&
        (!mask_from_bool || attn_mask->dim() < 2 || attn_mask->dim() > 4 || attn_mask->size(-1) != seq ||
         attn_mask->size(-2) != seq || attn_mask->numel() != seq * seq)) {
        return false;
    }
    return seq % 32 == 0 && key.size(2) == seq && value.size(2) == seq && query.size(0) == key.size(0) &&
           query.size(0) == value.size(0) && query.size(3) == key.size(3) && key.size(1) == value.size(1) &&
           key.size(1) > 0 && query.size(1) % key.size(1) == 0;
}

int64_t tt_fused_sdp_choice(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                            const std::optional<at::Tensor> &attn_mask, double dropout_p, bool, std::optional<double>,
                            bool) {
    const bool training =
        at::GradMode::is_enabled() && (query.requires_grad() || key.requires_grad() || value.requires_grad());
    TORCH_CHECK_NOT_IMPLEMENTED(dropout_p == 0.0, "tt-crank sdpa: dropout is not supported, got dropout_p=", dropout_p);
    g_sdpa_mask_from_bool = attn_mask.has_value() && attn_mask->defined() && attn_mask->scalar_type() == at::kBool;
    if (training && !ttml_sdpa_supported(query, key, value, attn_mask, g_sdpa_mask_from_bool)) {
        TORCH_WARN_ONCE("tt-crank sdpa: training call is outside what the ttml sdpa_fw/sdpa_bw kernels support "
                        "(non-bf16, float or per-batch mask, non-4-D, S % 32 != 0 or Sq != Sk); using the math "
                        "decomposition instead.");
        return as<int64_t>(at::SDPBackend::math);
    }
    return as<int64_t>(at::SDPBackend::overrideable);
}

// Emit the SDPA subgraph and run it, returning the attention output. Q/K/V are
// promoted to a common element type; the mask (when present) is passed through
// unchanged, matching the compile-path lowering in _compile.py.
at::Tensor run_sdpa(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                    const std::optional<at::Tensor> &attn_mask, bool is_causal, std::optional<double> scale) {
    const std::optional<float> scale_f = scale.has_value() ? std::optional<float>(as<float>(*scale)) : std::nullopt;
    // build_sdpa types the result after Q; the query and value head_dim must match.
    TORCH_CHECK(query.size(3) == value.size(3), "tt-crank sdpa: query head_dim (", query.size(3),
                ") must equal value head_dim (", value.size(3), ")");
    const std::vector<int64_t> out_shape{query.size(0), query.size(1), query.size(2), value.size(3)};

    if (attn_mask.has_value() && attn_mask->defined()) {
        const auto [q, k, v, mask] = align_on_tt(query, key, value, *attn_mask);
        auto mb = ModuleBuilder::init({spec_for(q), spec_for(k), spec_for(v), spec_for(mask)});
        auto args = mb.args();
        const c10::ScalarType promoted = at::promote_types(at::result_type(q, k), v.scalar_type());
        const auto promoted_mlir = mlir_element_type_for(promoted);
        auto result =
            build_sdpa(mb, mb.insert_typecast(args[0], promoted_mlir), mb.insert_typecast(args[1], promoted_mlir),
                       mb.insert_typecast(args[2], promoted_mlir), is_causal, scale_f, args[3]);
        auto module_op = std::move(mb).finalize({result});
        auto outputs = compile_and_run(std::move(module_op), {q, k, v, mask});
        return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
    }

    const auto [q, k, v] = align_on_tt(query, key, value);
    auto mb = ModuleBuilder::init({spec_for(q), spec_for(k), spec_for(v)});
    auto [promoted, qv, kv, vv] = promote_inputs(mb, q, k, v);
    auto result = build_sdpa(mb, qv, kv, vv, is_causal, scale_f, mlir::Value{});
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {q, k, v});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

// Torch already requires one dtype across q/k/v/out/grad_out, so no promotion here.
template <typename Build>
std::vector<tt::runtime::Tensor> run_ttml(Build build, const std::vector<at::Tensor> &tensors) {
    const std::vector<at::Tensor> aligned = align_on_tt(tensors);
    std::vector<TensorTypeSpec> specs;
    for (const at::Tensor &t : aligned) {
        specs.push_back(spec_for(t));
    }
    auto mb = ModuleBuilder::init(specs);
    auto module_op = std::move(mb).finalize(build(mb));
    return compile_and_run(std::move(module_op), aligned);
}

std::vector<at::Tensor> with_mask(std::vector<at::Tensor> tensors, const std::optional<at::Tensor> &mask) {
    if (mask.has_value() && mask->defined()) {
        tensors.push_back(*mask);
    }
    return tensors;
}

mlir::Value mask_arg(ModuleBuilder &mb, std::size_t index) {
    return mb.args().size() > index ? mb.args()[index] : mlir::Value{};
}

// Slots 0 (output) and 1 (logsumexp) carry values; 6/7 are the dropout RNG state autograd saves.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
tt_sdpa_overrideable(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                     const std::optional<at::Tensor> &attn_bias, double dropout_p, bool is_causal,
                     bool return_debug_mask, std::optional<double> scale) {
    TORCH_CHECK(!return_debug_mask, "tt-crank sdpa: return_debug_mask=True is not supported");
    TORCH_CHECK(dropout_p == 0.0, "tt-crank sdpa: dropout_p must be 0, got ", dropout_p);
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
                "tt-crank sdpa: query/key/value must be 4-D [batch, heads, seq, dim]");

    const int64_t b = query.size(0), h = query.size(1), s_q = query.size(2), s_kv = key.size(2);
    at::Tensor output, logsumexp;
    // Gated on grad mode, not requires_grad: DTensor re-dispatches on local tensors with requires_grad stripped.
    const bool mask_from_bool = std::exchange(g_sdpa_mask_from_bool, false);
    if (at::GradMode::is_enabled() && ttml_sdpa_supported(query, key, value, attn_bias, mask_from_bool)) {
        auto outputs = run_ttml(
            [&](ModuleBuilder &mb) {
                auto a = mb.args();
                auto [out, lse] = build_sdpa_fw(mb, a[0], a[1], a[2], is_causal, scale, mask_arg(mb, 3));
                return std::vector{out, lse};
            },
            with_mask({query, key, value}, attn_bias));
        output = wrap_tt_tensor(std::move(outputs[0]), {b, h, s_q, value.size(3)}, query.scalar_type());
        logsumexp = wrap_tt_tensor(std::move(outputs[1]), {b, h, s_q}, at::kFloat);
    } else {
        output = run_sdpa(query, key, value, attn_bias, is_causal, scale);
        logsumexp = at::empty({b, h, s_q}, query.options().dtype(at::kFloat));
    }
    return std::make_tuple(std::move(output), std::move(logsumexp), at::Tensor(), at::Tensor(), c10::SymInt(s_q),
                           c10::SymInt(s_kv), at::empty({}, at::dtype(at::kLong)), at::empty({}, at::dtype(at::kLong)),
                           at::Tensor());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
tt_sdpa_overrideable_backward(const at::Tensor &grad_out, const at::Tensor &query, const at::Tensor &key,
                              const at::Tensor &value, const at::Tensor &attn_bias, std::array<bool, 4> grad_input_mask,
                              const at::Tensor &out, const at::Tensor &logsumexp, const at::Tensor &,
                              const at::Tensor &, c10::SymInt, c10::SymInt, double dropout_p, bool is_causal,
                              const at::Tensor &, const at::Tensor &, std::optional<double> scale) {
    TORCH_CHECK(!grad_input_mask[3], "tt-crank sdpa backward: a gradient w.r.t. attn_bias is not supported");
    TORCH_CHECK(ttml_sdpa_supported(query, key, value, attn_bias, /*mask_from_bool=*/true) && logsumexp.dim() == 3,
                "tt-crank sdpa backward: this call is outside what the ttml sdpa_bw kernel supports");
    auto grads = run_ttml(
        [&](ModuleBuilder &mb) {
            auto a = mb.args();
            auto [dq, dk, dv] =
                build_sdpa_bw(mb, a[0], a[1], a[2], a[3], a[4], a[5], is_causal, scale, mask_arg(mb, 6));
            return std::vector{dq, dk, dv};
        },
        with_mask({grad_out, out, query, key, value, logsumexp}, attn_bias));
    auto wrap = [&](int i, const at::Tensor &like) {
        return grad_input_mask[i] ? wrap_tt_tensor(std::move(grads[i]), like.sizes(), like.scalar_type())
                                  : at::Tensor();
    };
    return std::make_tuple(wrap(0, query), wrap(1, key), wrap(2, value), at::Tensor());
}

} // namespace

// The composite consults `_fused_sdp_choice_stub` (a DispatchStub keyed on device
// type), not the `aten::_fused_sdp_choice` operator, so registering the stub for
// PrivateUse1 is the whole mechanism; no operator kernel is needed (a direct
// `aten::_fused_sdp_choice` call on a tt tensor is unused by the SDPA path and would
// just take the CPU fallback). Needs `at::native` in scope so the stub symbol and
// macro resolve, as in openreg's OpenRegExtra.cpp.
using namespace at::native;
REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &tt_fused_sdp_choice);

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_scaled_dot_product_fused_attention_overrideable", TORCH_FN(tt_sdpa_overrideable));
    m.impl("_scaled_dot_product_fused_attention_overrideable_backward", TORCH_FN(tt_sdpa_overrideable_backward));
}

} // namespace tt::crank::torch_backend
