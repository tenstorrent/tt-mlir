// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "options.hpp"

#include <optional>
#include <string>

#include "assert.hpp"
#include "ort.hpp"

// Compile options. These need to be implemented properly. Hacking them for now.
// Find a better way to pass them or ORT, or move these parsing functions to CompileOptions.

namespace tt::crank::onnx {

namespace {

// A session config entry's string value by its full key, or nullopt when unset.
std::optional<std::string> config_entry(const OrtSessionOptions *opts, const std::string &key) {
    int est = 0;
    check_call(ort_api().HasSessionConfigEntry(opts, key.c_str(), &est));
    if (est == 0) {
        return std::nullopt;
    }
    std::size_t size = 0;
    check_call(ort_api().GetSessionConfigEntry(opts, key.c_str(), nullptr, &size));
    std::string value(size, '\0');
    check_call(ort_api().GetSessionConfigEntry(opts, key.c_str(), value.data(), &size));
    value.resize(size - 1); // drop the null terminator
    return value;
}

// Our compile options are namespaced under the EP's prefix; ORT's own keys
// (e.g. ep.context_enable) are not.
std::optional<std::string> option_entry(const OrtSessionOptions *opts, const char *name) {
    return config_entry(opts, std::string{"ep.ttcrankexecutionprovider."} + name);
}

int parse_int(const char *name, const std::string &value) {
    try {
        std::size_t consumed = 0;
        int parsed = std::stoi(value, &consumed);
        TT_FATAL(consumed == value.size(), "trailing characters");
        return parsed;
    } catch (const std::exception &) {
        TT_THROW("compile option '{}': expected an integer, got '{}'", name, value);
    }
}

bool parse_bool(const char *name, const std::string &value) {
    if (value == "1" || value == "true") {
        return true;
    }
    if (value == "0" || value == "false") {
        return false;
    }
    TT_THROW("compile option '{}': expected 0/1/true/false, got '{}'", name, value);
}

CompileOptions::MathFidelity parse_math_fidelity(const char *name, const std::string &value) {
    if (value == "LoFi") {
        return CompileOptions::MathFidelity::LoFi;
    }
    if (value == "HiFi2") {
        return CompileOptions::MathFidelity::HiFi2;
    }
    if (value == "HiFi3") {
        return CompileOptions::MathFidelity::HiFi3;
    }
    if (value == "HiFi4") {
        return CompileOptions::MathFidelity::HiFi4;
    }
    TT_THROW("compile option '{}': expected LoFi/HiFi2/HiFi3/HiFi4, got '{}'", name, value);
}

CompileOptions::BfpDtype parse_bfp_dtype(const char *name, const std::string &value) {
    if (value == "BfpBf8") {
        return CompileOptions::BfpDtype::BfpBf8;
    }
    if (value == "BfpBf4") {
        return CompileOptions::BfpDtype::BfpBf4;
    }
    TT_THROW("compile option '{}': expected BfpBf8/BfpBf4, got '{}'", name, value);
}

} // namespace

CompileOptions parse_compile_options(const OrtSessionOptions *session_options) {
    CompileOptions options; // engine defaults

    if (auto value = option_entry(session_options, "optimization_level")) {
        options.optimization_level = parse_int("optimization_level", *value);
    }
    if (auto value = option_entry(session_options, "experimental_weight_dtype")) {
        options.experimental_weight_dtype = parse_bfp_dtype("experimental_weight_dtype", *value);
    }
    if (auto value = option_entry(session_options, "experimental_kv_cache_dtype")) {
        options.experimental_kv_cache_dtype = parse_bfp_dtype("experimental_kv_cache_dtype", *value);
    }
    if (auto value = option_entry(session_options, "math_fidelity")) {
        options.math_fidelity = parse_math_fidelity("math_fidelity", *value);
    }
    if (auto value = option_entry(session_options, "fp32_dest_acc_en")) {
        options.fp32_dest_acc_en = parse_bool("fp32_dest_acc_en", *value);
    }
    if (auto value = option_entry(session_options, "experimental_enable_fusing_conv2d_with_multiply_pattern")) {
        options.experimental_enable_fusing_conv2d_with_multiply_pattern =
            parse_bool("experimental_enable_fusing_conv2d_with_multiply_pattern", *value);
    }
    if (auto value = option_entry(session_options, "experimental_enable_permute_matmul_fusion")) {
        options.experimental_enable_permute_matmul_fusion =
            parse_bool("experimental_enable_permute_matmul_fusion", *value);
    }
    if (auto value = option_entry(session_options, "enable_trace")) {
        options.enable_trace = parse_bool("enable_trace", *value);
    }
    if (auto value = option_entry(session_options, "enable_const_eval")) {
        options.enable_const_eval = parse_bool("enable_const_eval", *value);
    }
    if (auto value = option_entry(session_options, "enable_const_eval_on_cpu")) {
        options.enable_const_eval_on_cpu = parse_bool("enable_const_eval_on_cpu", *value);
    }
    if (auto value = option_entry(session_options, "enable_const_eval_inputs_to_system_memory")) {
        options.enable_const_eval_inputs_to_system_memory =
            parse_bool("enable_const_eval_inputs_to_system_memory", *value);
    }
    if (auto value = option_entry(session_options, "experimental_enable_dram_space_saving_optimization")) {
        options.experimental_enable_dram_space_saving_optimization =
            parse_bool("experimental_enable_dram_space_saving_optimization", *value);
    }
    if (auto value = option_entry(session_options, "enable_create_d2m_subgraphs")) {
        options.enable_create_d2m_subgraphs = parse_bool("enable_create_d2m_subgraphs", *value);
    }
    if (auto value = option_entry(session_options, "ttnn_perf_metrics_enabled")) {
        options.ttnn_perf_metrics_enabled = parse_bool("ttnn_perf_metrics_enabled", *value);
    }
    if (auto value = option_entry(session_options, "ttnn_perf_metrics_output_file")) {
        options.ttnn_perf_metrics_output_file = *value;
    }
    return options;
}

bool parse_ep_ctx_enabled(const OrtSessionOptions *session_options) {
    auto value = config_entry(session_options, "ep.context_enable");
    return value.has_value() && parse_bool("ep.context_enable", *value);
}

} // namespace tt::crank::onnx
