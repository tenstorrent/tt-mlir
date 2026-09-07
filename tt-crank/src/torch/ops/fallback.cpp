// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Catch-all CPU fallback for the tt backend. Any aten op without an explicit
// PrivateUse1 kernel routes through here: tensors get materialized to CPU, the
// op runs on CPU, results are copied back to tt. Slow, but correct.
//
// For debugging, TT_KURBLA_LOG_FALLBACK_ENABLED=1 environment variable can be used
// which causes us to log every operation that triggers a fallback.

#include <atomic>
#include <cstdlib>
#include <cstring>

#include <ATen/native/CPUFallback.h>
#include <c10/util/Exception.h>
#include <torch/library.h>
#include <tt-logger/tt-logger.hpp>

#include "config.hpp"
#include "torch/ops/fallback.hpp"

namespace tt::kurbla::torch_backend {

namespace {

// Process-wide strict flag. When set, the fallback raises instead of running —
// used by `strict_no_fallback()` in tests to assert "this code path stays on
// the native tt kernels".
std::atomic<bool> g_fallback_strict{false};

void tt_cpu_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack) {
    TORCH_CHECK(!g_fallback_strict, "tt-kurbla strict-fallback: op `", op.operator_name(),
                "` would fall back to CPU but strict mode is enabled");

    if (log_fallback_enabled()) {
        log_info(tt::LogAlways, "tt-kurbla fallback: {}", c10::toString(op.operator_name()));
    }

    // error_on_views=true: a view op's contract is that the result shares
    // storage with the source, but cpu_fallback physically can't honor that
    // across the device boundary — the CPU result and the tt source live in
    // different storages. Fail if fallback happens on a view-like op.
    at::native::cpu_fallback(op, stack, /*error_on_views=*/true);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&tt_cpu_fallback>());
}

} // namespace

void set_fallback_strict(bool strict) {
    g_fallback_strict = strict;
}

bool fallback_strict() {
    return g_fallback_strict;
}

} // namespace tt::kurbla::torch_backend
