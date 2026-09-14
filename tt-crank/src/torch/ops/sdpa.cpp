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
// path lowers that to one `ttir.sdpa` (see _compile.py); eager runs it through the
// kernel below. This mirrors PyTorch's in-tree `openreg` backend. The choice function
// still returns MATH while autograd is live, since only the decomposition is
// differentiable.

#include <array>
#include <cstdint>
#include <optional>
#include <tuple>
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

int64_t tt_fused_sdp_choice(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                            const std::optional<at::Tensor> &, double, bool, std::optional<double>, bool) {
    // Choosing OVERRIDEABLE makes autograd record
    // ScaledDotProductFusedAttentionOverrideableBackward0, whose backward we don't
    // implement; the compile path can't even trace the joint, since that backward op
    // has no meta kernel. MATH decomposes to differentiable primitives instead, so
    // training keeps working at the cost of the single fused op.
    if (at::GradMode::is_enabled() && (query.requires_grad() || key.requires_grad() || value.requires_grad())) {
        TORCH_WARN_ONCE("tt-crank sdpa: inputs require grad, so SDPA is using the math decomposition "
                        "instead of the fused tt kernel. For inference, run under torch.no_grad() or "
                        "torch.inference_mode() to get the fused op.");
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

// Only `output` (slot 0) is a real value: the composite unpacks it and the
// forward-inference path never reads the rest. The other eight slots are
// backward/debug bookkeeping, minimal placeholders mirroring openreg. Backward is
// intentionally not implemented (see the explicit stub below): inference only.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
tt_sdpa_overrideable(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                     const std::optional<at::Tensor> &attn_bias, double dropout_p, bool is_causal,
                     bool return_debug_mask, std::optional<double> scale) {
    TORCH_CHECK(!return_debug_mask, "tt-crank sdpa: return_debug_mask=True is not supported "
                                    "(the debug_attn_mask output slot is left undefined)");
    TORCH_CHECK(dropout_p == 0.0, "tt-crank sdpa: dropout_p must be 0 (inference only), got ", dropout_p);
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
                "tt-crank sdpa: query/key/value must be 4-D [batch, heads, seq, dim]");

    at::Tensor output = run_sdpa(query, key, value, attn_bias, is_causal, scale);

    const int64_t b = query.size(0), h = query.size(1), s_q = query.size(2), s_kv = key.size(2);
    auto logsumexp = at::empty({b, h, s_q}, query.options().dtype(at::kFloat));
    auto philox_seed = at::empty({}, at::dtype(at::kLong));
    auto philox_offset = at::empty({}, at::dtype(at::kLong));
    return std::make_tuple(std::move(output), std::move(logsumexp), at::Tensor(), at::Tensor(), c10::SymInt(s_q),
                           c10::SymInt(s_kv), std::move(philox_seed), std::move(philox_offset), at::Tensor());
}

// Overrideable backward sdpa op - throws not implemented.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
tt_sdpa_overrideable_backward(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
                              const at::Tensor &, std::array<bool, 4>, const at::Tensor &, const at::Tensor &,
                              const at::Tensor &, const at::Tensor &, c10::SymInt, c10::SymInt, double, bool,
                              const at::Tensor &, const at::Tensor &, std::optional<double>) {
    TORCH_CHECK(false, "tt-crank sdpa: backward is not implemented (inference only)");
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
