// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "partition.hpp"

#include "assert.hpp"
#include "builder.hpp"
#include "graph.hpp"
#include "onnxruntime_c_api.h"

#include <tt/runtime/runtime.h>

#include <cstdint>
#include <cstring>
#include <ranges>
#include <unordered_set>
#include <vector>

namespace tt::crank::onnx {

namespace {

std::vector<std::uint32_t> to_u32_shape(const std::vector<std::int64_t> &shape) {
    std::vector<std::uint32_t> out;
    out.reserve(shape.size());
    for (auto dim : shape) {
        out.push_back(as<std::uint32_t>(dim));
    }
    return out;
}

} // namespace

// Iterates over all fused node and graph inputs and saves their info in usable form.
// Fused node represents 'node that can be executed', hence its inputs are only input tensors, and all parameters are
// excluded from it. In order to get graph parameters (weights), we must iterate over all graph nodes and collect them.
// For parameters, ORT will give us physical tensor data in input node, that we must extract.
Args::Args(const OrtGraph *graph, const OrtNode *fused_node) {
    // Collect inputs.
    for (const OrtValueInfo *input : node_inputs(fused_node)) {
        TT_FATAL(input != nullptr && !value_is_param(input), "Invalid program input.");
        specs.push_back({value_shape(input), to_runtime_dtype(value_elem_type(input))});
        roles.push_back(mlir::tt::ttcore::ArgumentType::Input);
        names.push_back(value_name(input));
        param_tensors.emplace_back(std::nullopt);
        ++inputs_count;
    }

    // Collect parameters.
    std::unordered_set<std::string> seen;
    for (const OrtNode *node : graph_nodes(graph)) {
        auto inputs = node_inputs(node);
        for (std::size_t idx = 0; idx < inputs.size(); ++idx) {
            const OrtValueInfo *input = inputs[idx];
            if (input == nullptr || !value_is_param(input) || is_metadata_input(node, idx)) {
                continue;
            }
            auto name = value_name(input);
            if (!seen.insert(name).second) {
                continue;
            }
            auto shape = value_shape(input);
            auto dtype = to_runtime_dtype(value_elem_type(input));
            ::tt::runtime::TensorDesc desc(to_u32_shape(shape), dtype);

            specs.push_back({shape, dtype});
            roles.push_back(mlir::tt::ttcore::ArgumentType::Parameter);
            names.push_back(name);

            // Create owned tensor from provided parameter tensor data.
            param_tensors.emplace_back(
                ::tt::runtime::createOwnedHostTensor(tensor_data(value_initializer(input)), desc));
            ++params_count;
        }
    }
}

Partition::Partition(const OrtGraph *graph, const OrtNode *fused_node)
    : OrtNodeComputeInfo{.ort_version_supported = ORT_API_VERSION,
                         .CreateState = CreateStateImpl,
                         .Compute = ComputeImpl,
                         .ReleaseState = ReleaseStateImpl},
      m_graph{graph}, m_fused_node{fused_node}, m_args{m_graph, m_fused_node} {}

void Partition::compile(const CompileOptions &options) {
    if (graph_ctx_serialized(m_graph)) {
        return deserialize();
    }

    OnnxModuleBuilder mb{m_args};
    for (const OrtNode *node : graph_nodes(m_graph)) {
        mb.build_node(node);
    }
    auto module_op = std::move(mb).finalize(m_fused_node);

    m_program = compile_ttir_to_ttnn_flatbuffer(*module_op, options).program;
}

// Serializes partition (compiled program and weights) to string.
// Format: [program_len][program][params]
std::string Partition::serialize() const {
    std::vector<std::uint8_t> program;
    m_program->binary.storeToMemory(program);

    std::string blob;
    const std::uint64_t program_len = program.size();
    blob.append(as<const char *>(&program_len), sizeof(program_len));
    blob.append(as<const char *>(program.data()), program.size());
    for (std::size_t i = inputs_count(); i < args_size(); ++i) {
        const auto bytes = ::tt::runtime::getTensorDataBuffer(param_tensor(i));
        blob.append(as<const char *>(bytes.data()), bytes.size());
    }
    return blob;
}

void Partition::deserialize() {
    TT_FATAL(graph_ctx_serialized(m_graph), "Ctx not serialized.");
    const std::string blob = node_attr_string(graph_nodes(m_graph)[0], "ep_cache_context", "");

    std::uint64_t program_len = 0;
    TT_FATAL(blob.size() >= sizeof(program_len), "context blob truncated");

    std::memcpy(&program_len, blob.data(), sizeof(program_len));
    TT_FATAL(sizeof(program_len) + program_len <= blob.size(), "context blob truncated");

    m_program =
        new CompiledProgram(::tt::runtime::Binary::loadFromMemory(blob.data() + sizeof(program_len), program_len));

    // We don't have parameters in graph, since we are loaded from .onnx file, so we must recreate them.
    // Don't botter attaching desc to args, since args are only used for params upload.
    m_args.params_count = m_program->num_inputs - m_args.inputs_count;
    std::size_t offset = sizeof(program_len) + program_len;
    for (const auto &desc : m_program->input_descs | std::views::drop(m_args.inputs_count)) {
        TT_FATAL(offset + desc.sizeBytes() <= blob.size(), "context blob truncated");
        m_args.param_tensors.emplace_back(::tt::runtime::createOwnedHostTensor(blob.data() + offset, desc));
        offset += desc.sizeBytes();
    }
}

} // namespace tt::crank::onnx
