#include "engine/compile_options.hpp"

#include <optional>
#include <sstream>
#include <string>

#include <ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h>
#include <ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h>
#include <ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h>

namespace tt::kurbla {

namespace {

mlir::tt::ttcore::Arch to_ttcore_arch(CompileOptions::MockArch arch) {
    switch (arch) {
        case CompileOptions::MockArch::WormholeB0:
            return mlir::tt::ttcore::Arch::WormholeB0;
        case CompileOptions::MockArch::Blackhole:
            return mlir::tt::ttcore::Arch::Blackhole;
    }
    return mlir::tt::ttcore::Arch::WormholeB0;
}

// Maps the kurbla-facing block-float dtype enum to tt-mlir's BFPDtype.
mlir::tt::ttnn::BFPDtype to_bfp_dtype(CompileOptions::BfpDtype dtype) {
    switch (dtype) {
        case CompileOptions::BfpDtype::BfpBf8:
            return mlir::tt::ttnn::BFPDtype::BFP_BFloat8;
        case CompileOptions::BfpDtype::BfpBf4:
            return mlir::tt::ttnn::BFPDtype::BFP_BFloat4;
    }
    return mlir::tt::ttnn::BFPDtype::None;
}

// Maps the kurbla-facing math fidelity enum to tt-mlir's OptionalMathFidelity.
// TtnnDefault maps to Undefined (defer the per-op choice to ttnn).
mlir::tt::ttnn::OptionalMathFidelity to_math_fidelity(CompileOptions::MathFidelity fidelity) {
    switch (fidelity) {
        case CompileOptions::MathFidelity::LoFi:
            return mlir::tt::ttnn::OptionalMathFidelity::LoFi;
        case CompileOptions::MathFidelity::HiFi2:
            return mlir::tt::ttnn::OptionalMathFidelity::HiFi2;
        case CompileOptions::MathFidelity::HiFi3:
            return mlir::tt::ttnn::OptionalMathFidelity::HiFi3;
        case CompileOptions::MathFidelity::HiFi4:
            return mlir::tt::ttnn::OptionalMathFidelity::HiFi4;
    }
    return mlir::tt::ttnn::OptionalMathFidelity::Undefined;
}

std::string to_string(CompileOptions::BfpDtype v) {
    switch (v) {
        case CompileOptions::BfpDtype::BfpBf8:
            return "bfp_bf8";
        case CompileOptions::BfpDtype::BfpBf4:
            return "bfp_bf4";
    }
    return "unknown";
}

std::string to_string(CompileOptions::MathFidelity v) {
    switch (v) {
        case CompileOptions::MathFidelity::LoFi:
            return "lofi";
        case CompileOptions::MathFidelity::HiFi2:
            return "hifi2";
        case CompileOptions::MathFidelity::HiFi3:
            return "hifi3";
        case CompileOptions::MathFidelity::HiFi4:
            return "hifi4";
    }
    return "unknown";
}

} // namespace

std::string CompileOptions::to_string() const {
    const auto opt_int = [](const std::optional<int> &v) { return v ? std::to_string(*v) : "none"; };
    const auto opt_str = [](const std::optional<std::string> &v) { return v ? *v : "none"; };
    const auto opt_bool = [](const std::optional<bool> &v) { return v ? (*v ? "true" : "false") : "none"; };
    const auto opt_bfp = [](const std::optional<BfpDtype> &v) { return v ? tt::kurbla::to_string(*v) : "none"; };
    const auto opt_fidelity = [](const std::optional<MathFidelity> &v) {
        return v ? tt::kurbla::to_string(*v) : "none";
    };

    std::stringstream ss;
    ss << "{ ";
    ss << "optimization_level: " << opt_int(optimization_level);
    ss << ", experimental_weight_dtype: " << opt_bfp(experimental_weight_dtype);
    ss << ", experimental_kv_cache_dtype: " << opt_bfp(experimental_kv_cache_dtype);
    ss << ", math_fidelity: " << opt_fidelity(math_fidelity);
    ss << ", fp32_dest_acc_en: " << opt_bool(fp32_dest_acc_en);
    ss << ", experimental_enable_fusing_conv2d_with_multiply_pattern: "
       << opt_bool(experimental_enable_fusing_conv2d_with_multiply_pattern);
    ss << ", experimental_enable_permute_matmul_fusion: " << opt_bool(experimental_enable_permute_matmul_fusion);
    ss << ", enable_trace: " << opt_bool(enable_trace);
    ss << ", enable_const_eval: " << opt_bool(enable_const_eval);
    ss << ", enable_const_eval_on_cpu: " << opt_bool(enable_const_eval_on_cpu);
    ss << ", enable_const_eval_inputs_to_system_memory: " << opt_bool(enable_const_eval_inputs_to_system_memory);
    ss << ", experimental_enable_dram_space_saving_optimization: "
       << opt_bool(experimental_enable_dram_space_saving_optimization);
    ss << ", enable_create_d2m_subgraphs: " << opt_bool(enable_create_d2m_subgraphs);
    ss << ", ttnn_perf_metrics_enabled: " << opt_bool(ttnn_perf_metrics_enabled);
    ss << ", ttnn_perf_metrics_output_file: " << opt_str(ttnn_perf_metrics_output_file);
    ss << ", all_reduce_workaround_enabled: " << opt_bool(all_reduce_workaround_enabled);
    ss << " }";

    return ss.str();
}

void CompileOptions::set_options_on(mlir::tt::ttnn::TTIRToTTNNRuntimePipelineOptions &opts) const {
    if (mock_arch.has_value()) {
        opts.mockSystemDescArch = to_ttcore_arch(*mock_arch);
    }
    if (system_desc_path.has_value()) {
        opts.systemDescPath = *system_desc_path;
    }
    if (optimization_level.has_value()) {
        opts.optimizationLevel = *optimization_level;
    }
    if (experimental_weight_dtype.has_value()) {
        opts.experimentalWeightDtype = to_bfp_dtype(*experimental_weight_dtype);
    }
    if (experimental_kv_cache_dtype.has_value()) {
        opts.experimentalKVCacheDtype = to_bfp_dtype(*experimental_kv_cache_dtype);
    }
    if (math_fidelity.has_value()) {
        opts.computeCfgMathFidelity = to_math_fidelity(*math_fidelity);
    }
    if (fp32_dest_acc_en.has_value()) {
        opts.computeCfgFp32DestAccEn = *fp32_dest_acc_en;
    }
    if (experimental_enable_fusing_conv2d_with_multiply_pattern.has_value()) {
        opts.enableFusingConv2dWithMultiplyPattern = *experimental_enable_fusing_conv2d_with_multiply_pattern;
    }
    if (experimental_enable_permute_matmul_fusion.has_value()) {
        opts.enablePermuteMatmulFusion = *experimental_enable_permute_matmul_fusion;
    }
    if (enable_trace.has_value()) {
        opts.enableTrace = *enable_trace;
    }
    if (enable_const_eval.has_value()) {
        opts.enableConstEval = *enable_const_eval;
    }
    if (enable_const_eval_on_cpu.has_value()) {
        opts.enableCPUHoistedConstEval = *enable_const_eval_on_cpu;
    }
    if (enable_const_eval_inputs_to_system_memory.has_value()) {
        opts.enableConstEvalInputsToSystemMemory = *enable_const_eval_inputs_to_system_memory;
    }
    if (experimental_enable_dram_space_saving_optimization.has_value()) {
        opts.dramSpaceSavingOptimizationEnabled = *experimental_enable_dram_space_saving_optimization;
    }
    if (enable_create_d2m_subgraphs.has_value()) {
        opts.enableCreateD2MSubgraphs = *enable_create_d2m_subgraphs;
    }
    if (ttnn_perf_metrics_enabled.has_value()) {
        opts.ttnnPerfMetricsEnabled = *ttnn_perf_metrics_enabled;
    }
    if (ttnn_perf_metrics_output_file.has_value()) {
        opts.ttnnPerfMetricsOutputFile = *ttnn_perf_metrics_output_file;
    }
    if (all_reduce_workaround_enabled.has_value()) {
        opts.allReduceWorkaroundEnabled = *all_reduce_workaround_enabled;
    }
}

} // namespace tt::kurbla
