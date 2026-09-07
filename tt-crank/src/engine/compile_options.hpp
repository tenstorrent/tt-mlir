// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include <tt/runtime/types.h>

#include "tt_kurbla_export.hpp"

// Forward-declared so this header stays free of the heavy tt-mlir TTNN pipeline
// header; set_compile_options only takes a reference to it.
namespace mlir::tt::ttnn {
struct TTIRToTTNNRuntimePipelineOptions;
} // namespace mlir::tt::ttnn

namespace tt::kurbla {

// Options for TTIR-starting pipelines.
// Default values are only set for options that we want to override by default.
// For other values, we will take default mlir values.
struct TT_KURBLA_API CompileOptions {
    // Block-float dtype for weight / KV-cache conversion. Maps to tt-mlir's
    // ttnn::BFPDtype.
    enum class BfpDtype { BfpBf8, BfpBf4 };

    // Math fidelity for ops exposing a compute-kernel config. Maps to tt-mlir's
    // ttnn::OptionalMathFidelity.
    enum class MathFidelity { LoFi, HiFi2, HiFi3, HiFi4 };

    // 0=all optimizer passes off (fastest), 1=optimizer on without sharding,
    // 2=optimizer on with memory layout analysis (sharding).
    std::optional<int> optimization_level = 0;

    // Target dtype for weight conversion in matmul/linear ops.
    std::optional<BfpDtype> experimental_weight_dtype;

    // Experimental KV-cache dtype override.
    std::optional<BfpDtype> experimental_kv_cache_dtype;

    // Math fidelity override for all ops exposing a compute-kernel config.
    std::optional<MathFidelity> math_fidelity;

    // fp32 destination accumulation override.
    std::optional<bool> fp32_dest_acc_en;

    // Fuse conv2d + multiply in the TTNN fusing pass.
    std::optional<bool> experimental_enable_fusing_conv2d_with_multiply_pattern;

    // Fuse transpose + matmul/linear.
    // TODO: This should be default constructed, once mlir set it's value to true by default.
    std::optional<bool> experimental_enable_permute_matmul_fusion = true;

    // Hoist repeated op sequences into a TTNN trace (eliminates host dispatch
    // overhead). Requires all non-consteval ops on device. -> enableTrace
    std::optional<bool> enable_trace;

    // Generate const-eval subgraphs for weight-only computation.
    std::optional<bool> enable_const_eval;

    // Hoist const-eval subgraphs to the CPU module (32-bit precision).
    std::optional<bool> enable_const_eval_on_cpu;

    // Annotate const-eval inputs as system memory.
    std::optional<bool> enable_const_eval_inputs_to_system_memory;

    // Run the DRAM space-saving optimization pass (TTNNMemoryManagement).
    std::optional<bool> experimental_enable_dram_space_saving_optimization;

    // Create D2M subgraphs for elementwise fusion (only effective at
    // optimization_level >= 1).
    std::optional<bool> enable_create_d2m_subgraphs;

    // Collect TTNN performance metrics during execution.
    std::optional<bool> ttnn_perf_metrics_enabled;

    // Output file for TTNN performance metrics (empty -> default location).
    std::optional<std::string> ttnn_perf_metrics_output_file;

    ///////////
    // Utils //
    ///////////

    std::string to_string() const;

    // Translates kurbla CompileOptions into the tt-mlir TTIR->TTNN runtime pipeline
    // options (enum conversions + the has_value()-gated overrides).
    void set_options_on(mlir::tt::ttnn::TTIRToTTNNRuntimePipelineOptions &opts) const;
};

} // namespace tt::kurbla
