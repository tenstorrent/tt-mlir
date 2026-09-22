// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <utility>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

#include "engine/ttir_module_builder.hpp"
#include "onnxruntime_c_api.h"
#include "partition.hpp"

namespace tt::crank::onnx {

// Module builder for onnx graph.
// It is constructed from (partition) args, which sets initial mapping values (see m_values).
// For each graph node, build_node should be called, which will emit TTIR (using engine module builder) for provided
// node. After whole graph is traversed, user should call finalize, which will return built module.
class OnnxModuleBuilder {
public:
    explicit OnnxModuleBuilder(const Args &args);

    void build_node(const OrtNode *node);

    mlir::OwningOpRef<mlir::ModuleOp> finalize(const OrtNode *fused_node) &&;

private:
    mlir::Value operand(const OrtNode *node, std::size_t index);
    mlir::Value required_operand(const OrtNode *node, std::size_t index);

    void set_output(const OrtNode *node, mlir::Value result);

    std::pair<mlir::Value, mlir::Value> align_ranks(mlir::Value lhs, mlir::Value rhs);

    template <mlir::Value (*BuildFn)(ModuleBuilder &, mlir::Value, mlir::Value)> void build_binary(const OrtNode *node);

    template <mlir::Value (*BuildFn)(ModuleBuilder &, mlir::Value)> void build_unary(const OrtNode *node);

    void build_add(const OrtNode *node);
    void build_sub(const OrtNode *node);
    void build_div(const OrtNode *node);
    void build_softmax(const OrtNode *node);
    void build_matmul(const OrtNode *node);
    void build_gemm(const OrtNode *node);
    void build_conv(const OrtNode *node);
    void build_max_pool(const OrtNode *node);
    void build_global_average_pool(const OrtNode *node);
    void build_reduce_mean(const OrtNode *node);
    void build_batch_norm(const OrtNode *node);
    void build_flatten(const OrtNode *node);
    void build_reshape(const OrtNode *node);
    void build_transpose(const OrtNode *node);
    void build_concat(const OrtNode *node);
    void build_identity(const OrtNode *node);
    void build_cast(const OrtNode *node);
    void build_clip(const OrtNode *node);
    void build_slice(const OrtNode *node);
    void build_gather(const OrtNode *node);

    // Engine module builder used for emitting TTIR for graph nodes.
    ModuleBuilder m_mb;

    // node_name -> mlir::Value mappings used for building nodes.
    // At the start, it holds only program args, and each time new node is created, it's output is stored in values.
    // This allows us to get mlir values directly from graph nodes (by node name) while traversing graph in compile.
    std::unordered_map<std::string, mlir::Value> m_values;
};

} // namespace tt::crank::onnx
