// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt/runtime/types.h>

#include "onnxruntime_c_api.h"

namespace tt::crank::onnx {

// Various helpers for ORT graph.

// ---- graph level ------------------------------------------------------------

// Nodes of `graph` in a valid topological order (ORT guarantees the order).
std::vector<const OrtNode *> graph_nodes(const OrtGraph *graph);
// Graph inputs, in declaration order. May include initializer-backed inputs.
std::vector<const OrtValueInfo *> graph_inputs(const OrtGraph *graph);
// Graph outputs, in declaration order.
std::vector<const OrtValueInfo *> graph_outputs(const OrtGraph *graph);
// Constant initializers (weights), including outer-scope ones for subgraphs.
std::vector<const OrtValueInfo *> graph_initializers(const OrtGraph *graph);
// Whether context node for this graph is serialized.
bool graph_ctx_serialized(const std::vector<const OrtNode *> &);
bool graph_ctx_serialized(const OrtGraph *graph);

// ---- node level -------------------------------------------------------------

std::string node_op_type(const OrtNode *node);
std::string node_name(const OrtNode *node);
// Empty string means the default ONNX domain.
std::string node_domain(const OrtNode *node);
// The opset version whose semantics this node follows (schema since_version).
int node_since_version(const OrtNode *node);
// Entries may be nullptr for omitted optional inputs.
std::vector<const OrtValueInfo *> node_inputs(const OrtNode *node);
std::vector<const OrtValueInfo *> node_outputs(const OrtNode *node);
// True for a build-time constant input the builder reads directly and excludes from program params (see Args):
// Reshape shape, ReduceMean axes (opset>=18, as input), Resize roi/scales/sizes, etc.
bool is_metadata_input(const OrtNode *node, std::size_t index);

// ---- value level ------------------------------------------------------------

const char *value_name_ref(const OrtValueInfo *value);
std::string value_name(const OrtValueInfo *value);
// False for non-tensor values (sequences, maps, optionals).
bool value_is_tensor(const OrtValueInfo *value);
// ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED when the value is not a tensor.
ONNXTensorElementDataType value_elem_type(const OrtValueInfo *value);
// Tensor dims; symbolic/unknown dims are -1. Empty for rank-0 tensors AND for
// non-tensor values — check value_is_tensor first when that matters.
std::vector<std::int64_t> value_shape(const OrtValueInfo *value);
bool value_is_param(const OrtValueInfo *value);
// The initializer's data as an OrtValue (borrowed; mmapped on first access
// for external data), or nullptr when the value is not an initializer.
const OrtValue *value_initializer(const OrtValueInfo *value);

// ---- kernel context level ----------------------------------------------------
std::size_t ctx_inputs_count(const OrtKernelContext *ctx);
const OrtValue *ctx_input_at(const OrtKernelContext *ctx, std::size_t idx);
OrtValue *ctx_output_at(OrtKernelContext *ctx, std::size_t idx, const std::vector<std::int64_t> &shape);

// ---- node attributes ---------------------------------------------------------

bool node_has_attr(const OrtNode *node, const char *name);
// Scalar attribute readers: `fallback` when the attribute is absent; throws
// when it exists with a different type.
std::int64_t node_attr_i64(const OrtNode *node, const char *name, std::int64_t fallback);
float node_attr_f32(const OrtNode *node, const char *name, float fallback);
std::string node_attr_string(const OrtNode *node, const char *name, const std::string &fallback);
std::vector<std::int64_t> node_attr_i64s(const OrtNode *node, const char *name, std::vector<std::int64_t> fallback);

// ---- tensor data --------------------------------------------------------------

// Raw host bytes of an OrtValue tensor (borrowed; valid while the value is).
const void *tensor_data(const OrtValue *tensor);

// Raw host bytes of an OrtValue tensor (borrowed; valid while the value is).
void *tensor_mutable_data(OrtValue *tensor);

// Element type / dims of an OrtValue tensor — the OrtValue counterparts of
// value_elem_type / value_shape.
ONNXTensorElementDataType tensor_elem_type(const OrtValue *tensor);
std::vector<std::int64_t> tensor_shape(const OrtValue *tensor);

// The values of a constant int64 initializer (e.g. a Reshape shape or
// ReduceMean axes input). Throws if `value` is not an int64 constant.
std::vector<std::int64_t> constant_i64s(const OrtValueInfo *value);

// The values of a constant float32 initializer (e.g. a Resize scales input).
// Throws if `value` is not a float32 constant.
std::vector<float> constant_f32s(const OrtValueInfo *value);

// The memory device the OrtValue's buffer lives on.
const OrtMemoryDevice *tensor_device(const OrtValue *tensor);

// ONNX type to runtime data type.
// Throws for unsupported data types.
::tt::target::DataType to_runtime_dtype(ONNXTensorElementDataType elem_type);

bool is_supported_dtype(ONNXTensorElementDataType elem_type);

// ---- op kinds -----------------------------------------------------------------

enum class OpKind : std::size_t { // NOLINT
    Add,
    Sub,
    Mul,
    Div,
    Relu,
    Sigmoid,
    Exp,
    Log,
    Neg,
    Softmax,
    MatMul,
    Gemm,
    Conv,
    MaxPool,
    GlobalAveragePool,
    ReduceMean,
    BatchNormalization,
    Flatten,
    Reshape,
    Transpose,
    Concat,
    Identity,
    Cast,
    Clip,
    Slice,
    Gather,
    Unsupported,
};

// OpKind for this node. Unsupported for unknown op types and for ops outside
// the default ONNX domain.
OpKind op_kind(const OrtNode *node);

// Returns whether this node is supported.
bool is_node_supported(const OrtNode *node);

// True for an EPContext node this EP produced (a precompiled partition to
// load rather than compile).
bool is_ep_context_node(const OrtNode *node);

}; // namespace tt::crank::onnx
