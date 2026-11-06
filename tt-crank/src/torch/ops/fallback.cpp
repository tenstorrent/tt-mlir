// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Catch-all CPU fallback for the tt backend. Any aten op without an explicit
// PrivateUse1 kernel routes through here: tensors get materialized to CPU, the
// op runs on CPU, results are copied back to tt. Slow, but correct.
//
// On a multi-chip mesh a tt tensor spans every chip (see TensorStorage), and its
// `.cpu()` is only chip 0's slab -- the `to_local` view DTensor expects. torch's
// `at::native::cpu_fallback` would therefore compute on chip 0's data alone and
// upload the result replicated, silently collapsing a *sharded* tensor onto chip
// 0's shard. So on a mesh the op runs once per shard, on that chip's slab, and the
// per-shard results are reassembled into one multi-device tensor; replicated
// operands (identical slabs) are recognised and run once.
//
// For debugging, TT_CRANK_LOG_FALLBACK_ENABLED=1 environment variable can be used
// which causes us to log every operation that triggers a fallback.

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <ATen/core/List.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/Exception.h>
#include <torch/library.h>
#include <tt-logger/tt-logger.hpp>

#include "cast.hpp"
#include "engine/device.hpp"
#include "torch/tensor.hpp"

#include "config.hpp"
#include "torch/ops/fallback.hpp"

namespace tt::crank::torch_backend {

namespace {

// Process-wide strict flag. When set, the fallback raises instead of running —
// used by `strict_no_fallback()` in tests to assert "this code path stays on
// the native tt kernels".
std::atomic<bool> g_fallback_strict{false};

bool is_tt_tensor(const c10::IValue &v) {
    return v.isTensor() && v.toTensor().defined() && is_tt(v.toTensor());
}

// Host copies of every tt tensor argument on the stack, one CPU tensor per chip.
// `shards[k]` belongs to stack position k (empty for non-tt entries). Tensor lists are
// flattened into `list_shards[k][j]` for element j of the list at position k.
struct HostArgs {
    std::vector<std::vector<at::Tensor>> shards;
    std::vector<std::vector<std::vector<at::Tensor>>> list_shards;
};

HostArgs host_args_of(const torch::jit::Stack &stack) {
    HostArgs args;
    args.shards.resize(stack.size());
    args.list_shards.resize(stack.size());
    for (std::size_t k = 0; k < stack.size(); ++k) {
        const c10::IValue &v = stack[k];
        if (is_tt_tensor(v)) {
            args.shards[k] = host_shards_of(v.toTensor());
        } else if (v.isTensorList()) {
            for (const at::Tensor &t : v.toTensorVector()) {
                args.list_shards[k].push_back(t.defined() && is_tt(t) ? host_shards_of(t) : std::vector<at::Tensor>{t});
            }
        }
    }
    return args;
}

// True iff every chip holds the same bytes, i.e. the tensor is replicated and one CPU
// run covers all chips.
bool shards_identical(const std::vector<at::Tensor> &shards) {
    for (std::size_t i = 1; i < shards.size(); ++i) {
        if (shards[i].scalar_type() != shards[0].scalar_type() || shards[i].sizes() != shards[0].sizes() ||
            std::memcmp(shards[i].data_ptr(), shards[0].data_ptr(), shards[0].nbytes()) != 0) {
            return false;
        }
    }
    return true;
}

bool all_replicated(const HostArgs &args) {
    for (const auto &shards : args.shards) {
        if (!shards_identical(shards)) {
            return false;
        }
    }
    for (const auto &list : args.list_shards) {
        for (const auto &shards : list) {
            if (!shards_identical(shards)) {
                return false;
            }
        }
    }
    return true;
}

// The stack as seen from chip `i`: tt tensors swapped for their chip-i CPU slab,
// device arguments pointed at CPU (what `_to_copy(..., device=tt)` and friends carry).
torch::jit::Stack stack_for_shard(const torch::jit::Stack &stack, const HostArgs &args, std::size_t i) {
    torch::jit::Stack cpu_stack;
    cpu_stack.reserve(stack.size());
    for (std::size_t k = 0; k < stack.size(); ++k) {
        const c10::IValue &v = stack[k];
        if (!args.shards[k].empty()) {
            const auto &shards = args.shards[k];
            cpu_stack.emplace_back(shards[shards.size() == 1 ? 0 : i]);
        } else if (v.isTensorList()) {
            c10::List<at::Tensor> list;
            for (const auto &shards : args.list_shards[k]) {
                list.push_back(shards[shards.size() == 1 ? 0 : i]);
            }
            cpu_stack.emplace_back(std::move(list));
        } else if (v.isDevice() && is_tt(v.toDevice())) {
            cpu_stack.emplace_back(c10::Device(c10::DeviceType::CPU));
        } else {
            cpu_stack.push_back(v);
        }
    }
    return cpu_stack;
}

// Multi-chip fallback: run the op on CPU once per shard and reassemble. Mirrors what
// `at::native::cpu_fallback` does for the single-device case -- mutable (`Tensor(a!)`)
// arguments are written back into the tt originals and aliasing returns hand back the
// original argument -- but per chip.
void per_shard_cpu_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack, const HostArgs &args,
                            std::size_t num_shards) {
    const c10::FunctionSchema &schema = op.schema();
    const std::size_t num_args = schema.arguments().size();
    const std::size_t num_returns = schema.returns().size();
    TORCH_CHECK(stack->size() >= num_args, "tt-crank fallback: stack shorter than the schema of ", schema.name());
    const std::size_t base = stack->size() - num_args;
    const torch::jit::Stack originals(stack->begin() + as<std::ptrdiff_t>(base), stack->end());
    HostArgs host_args;
    host_args.shards.assign(args.shards.begin() + as<std::ptrdiff_t>(base), args.shards.end());
    host_args.list_shards.assign(args.list_shards.begin() + as<std::ptrdiff_t>(base), args.list_shards.end());

    // One CPU run per chip. `runs[i]` is chip i's stack after the call: its arguments
    // (possibly mutated in place) followed by its returns.
    std::vector<torch::jit::Stack> runs;
    runs.reserve(num_shards);
    for (std::size_t i = 0; i < num_shards; ++i) {
        torch::jit::Stack cpu_stack = stack_for_shard(originals, host_args, i);
        const torch::jit::Stack cpu_args = cpu_stack;
        op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), &cpu_stack);
        TORCH_CHECK(cpu_stack.size() == num_returns, "tt-crank fallback: ", schema.name(), " returned ",
                    cpu_stack.size(), " values, schema says ", num_returns);
        torch::jit::Stack run = cpu_args;
        run.insert(run.end(), cpu_stack.begin(), cpu_stack.end());
        runs.push_back(std::move(run));
    }

    // Write mutated tt arguments back, chip by chip.
    for (std::size_t k = 0; k < num_args; ++k) {
        const c10::AliasInfo *alias = schema.arguments()[k].alias_info();
        if (alias == nullptr || !alias->isWrite() || host_args.shards[k].empty()) {
            continue;
        }
        const at::Tensor &orig = originals[k].toTensor();
        std::vector<at::Tensor> mutated;
        mutated.reserve(num_shards);
        for (std::size_t i = 0; i < num_shards; ++i) {
            mutated.push_back(runs[i][k].toTensor());
        }
        TORCH_CHECK(mutated[0].sizes() == orig.sizes() && mutated[0].scalar_type() == orig.scalar_type(),
                    "tt-crank fallback: ", schema.name(), " resized or retyped its in-place argument #", k, " on CPU (",
                    orig.sizes(), " ", orig.scalar_type(), " -> ", mutated[0].sizes(), " ", mutated[0].scalar_type(),
                    "), which the tt tensor cannot follow");
        storage_of(orig).replace(storage_of(tt_from_host_shards(mutated)).tensor());
    }

    // Returns: an aliasing return (`-> Tensor(a!)`) is the original tt argument it aliases;
    // anything else tensor-valued is reassembled from the per-chip results.
    torch::jit::drop(*stack, num_args);
    for (std::size_t j = 0; j < num_returns; ++j) {
        const c10::IValue &first = runs[0][num_args + j];
        const c10::AliasInfo *alias = schema.returns()[j].alias_info();
        if (alias != nullptr && first.isTensor()) {
            std::optional<std::size_t> aliased;
            for (std::size_t k = 0; k < num_args && !aliased; ++k) {
                const c10::AliasInfo *arg_alias = schema.arguments()[k].alias_info();
                if (arg_alias != nullptr && arg_alias->beforeSets() == alias->beforeSets()) {
                    aliased = k;
                }
            }
            TORCH_CHECK(aliased.has_value(), "tt-crank fallback: ", schema.name(), " return #", j,
                        " aliases no argument");
            stack->push_back(originals[*aliased]);
            continue;
        }
        if (first.isTensor() && first.toTensor().defined()) {
            std::vector<at::Tensor> outs;
            outs.reserve(num_shards);
            for (std::size_t i = 0; i < num_shards; ++i) {
                outs.push_back(runs[i][num_args + j].toTensor());
            }
            stack->push_back(tt_from_host_shards(outs));
            continue;
        }
        TORCH_CHECK(!first.isTensorList(), "tt-crank fallback: ", schema.name(),
                    " returns a tensor list, which the per-shard fallback does not reassemble yet");
        stack->push_back(first);
    }
}

void tt_cpu_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack) {
    TORCH_CHECK(!g_fallback_strict, "tt-crank strict-fallback: op `", op.operator_name(),
                "` would fall back to CPU but strict mode is enabled");

    if (log_fallback_enabled()) {
        log_info(tt::LogAlways, "tt-crank fallback: {}", c10::toString(op.operator_name()));
    }

    // error_on_views=true: a view op's contract is that the result shares
    // storage with the source, but cpu_fallback physically can't honor that
    // across the device boundary — the CPU result and the tt source live in
    // different storages. Fail if fallback happens on a view-like op.
    const std::size_t mesh_size = ::tt::crank::runtime_device_mesh_size();
    if (mesh_size <= 1) {
        at::native::cpu_fallback(op, stack, /*error_on_views=*/true);
        return;
    }

    // On a mesh, decide per call: replicated operands (every chip holds the same
    // slab) take torch's single-device fallback, whose `.cpu()`/upload round-trip is
    // exact for them; anything sharded runs per chip.
    const HostArgs args = host_args_of(*stack);
    if (all_replicated(args)) {
        at::native::cpu_fallback(op, stack, /*error_on_views=*/true);
        return;
    }
    for (const c10::Argument &ret : op.schema().returns()) {
        TORCH_CHECK(ret.alias_info() == nullptr || ret.alias_info()->isWrite(), "tt-crank fallback: view op `",
                    op.operator_name(), "` reached the CPU fallback; a view cannot cross the device boundary");
    }
    per_shard_cpu_fallback(op, stack, args, mesh_size);
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

} // namespace tt::crank::torch_backend
