// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "graph.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <optional>
#include <string>
#include <unordered_map>

#include <tt-logger/tt-logger.hpp>

#include "assert.hpp"
#include "cast.hpp"
#include "config.hpp"
#include "ort.hpp"

#include "onnxruntime_c_api.h"

namespace tt::crank::onnx {

namespace {

template <typename Handle, typename Item>
std::vector<Item> fetch_items(Handle handle, OrtStatus *(ORT_API_CALL *count_fn)(Handle, std::size_t *)NO_EXCEPTION,
                              OrtStatus *(ORT_API_CALL *get_fn)(Handle, Item *, std::size_t)NO_EXCEPTION) {
    std::size_t count = 0;
    check_call(count_fn(handle, &count));

    std::vector<Item> items(count);
    if (count > 0) {
        check_call(get_fn(handle, items.data(), count));
    }
    return items;
}

const OrtTensorTypeAndShapeInfo *tensor_info_of(const OrtValueInfo *value) {
    const OrtTypeInfo *type_info = nullptr;
    check_call(ort_api().GetValueInfoTypeInfo(value, &type_info));

    ONNXType onnx_type = ONNX_TYPE_UNKNOWN;
    check_call(ort_api().GetOnnxTypeFromTypeInfo(type_info, &onnx_type));
    if (onnx_type != ONNX_TYPE_TENSOR) {
        return nullptr;
    }

    const OrtTensorTypeAndShapeInfo *tensor_info = nullptr;
    check_call(ort_api().CastTypeInfoToTensorInfo(type_info, &tensor_info));
    return tensor_info;
}

const std::unordered_map<std::string_view, OpKind> op_kinds = {
    {"Add", OpKind::Add},
    {"Sub", OpKind::Sub},
    {"Mul", OpKind::Mul},
    {"Div", OpKind::Div},
    {"Relu", OpKind::Relu},
    {"Sigmoid", OpKind::Sigmoid},
    {"Exp", OpKind::Exp},
    {"Log", OpKind::Log},
    {"Neg", OpKind::Neg},
    {"Softmax", OpKind::Softmax},
    {"MatMul", OpKind::MatMul},
    {"Gemm", OpKind::Gemm},
    {"Conv", OpKind::Conv},
    {"MaxPool", OpKind::MaxPool},
    {"GlobalAveragePool", OpKind::GlobalAveragePool},
    {"ReduceMean", OpKind::ReduceMean},
    {"BatchNormalization", OpKind::BatchNormalization},
    {"Flatten", OpKind::Flatten},
    {"Reshape", OpKind::Reshape},
    {"Transpose", OpKind::Transpose},
    {"Concat", OpKind::Concat},
    {"Identity", OpKind::Identity},
    {"Cast", OpKind::Cast},
    {"Clip", OpKind::Clip},
    {"Slice", OpKind::Slice},
    {"Gather", OpKind::Gather},
};

std::optional<::tt::target::DataType> dtype_or_none(ONNXTensorElementDataType elem_type) {
    switch (elem_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return ::tt::target::DataType::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return ::tt::target::DataType::Float64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return ::tt::target::DataType::BFloat16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return ::tt::target::DataType::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return ::tt::target::DataType::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return ::tt::target::DataType::Bool;
        default:
            return std::nullopt;
    }
}

bool is_static_supported(const OrtValueInfo *value) {
    if (!is_supported_dtype(value_elem_type(value))) {
        return false;
    }
    auto shape = value_shape(value);
    return std::ranges::all_of(shape, [](std::int64_t dim) { return dim > 0; });
}

} // namespace

::tt::target::DataType to_runtime_dtype(ONNXTensorElementDataType elem_type) {
    auto dtype = dtype_or_none(elem_type);
    TT_FATAL(dtype.has_value(), "unsupported ONNX element type {}", as<int>(elem_type));
    return *dtype;
}

bool is_supported_dtype(ONNXTensorElementDataType elem_type) {
    return dtype_or_none(elem_type).has_value();
}

std::vector<const OrtNode *> graph_nodes(const OrtGraph *graph) {
    return fetch_items(graph, ort_api().Graph_GetNumNodes, ort_api().Graph_GetNodes);
}

std::vector<const OrtValueInfo *> graph_inputs(const OrtGraph *graph) {
    return fetch_items(graph, ort_api().Graph_GetNumInputs, ort_api().Graph_GetInputs);
}

std::vector<const OrtValueInfo *> graph_outputs(const OrtGraph *graph) {
    return fetch_items(graph, ort_api().Graph_GetNumOutputs, ort_api().Graph_GetOutputs);
}

std::vector<const OrtValueInfo *> graph_initializers(const OrtGraph *graph) {
    return fetch_items(graph, ort_api().Graph_GetNumInitializers, ort_api().Graph_GetInitializers);
}

bool graph_ctx_serialized(const std::vector<const OrtNode *> &nodes) {
    return nodes.size() == 1 && is_ep_context_node(nodes[0]);
}

bool graph_ctx_serialized(const OrtGraph *graph) {
    return graph_ctx_serialized(graph_nodes(graph));
}

std::string node_op_type(const OrtNode *node) {
    const char *op_type = nullptr;
    check_call(ort_api().Node_GetOperatorType(node, &op_type));
    return op_type;
}

std::string node_name(const OrtNode *node) {
    const char *name = nullptr;
    check_call(ort_api().Node_GetName(node, &name));
    return name;
}

std::string node_domain(const OrtNode *node) {
    const char *domain = nullptr;
    check_call(ort_api().Node_GetDomain(node, &domain));
    return domain;
}

int node_since_version(const OrtNode *node) {
    int since_version = 0;
    check_call(ort_api().Node_GetSinceVersion(node, &since_version));
    return since_version;
}

std::vector<const OrtValueInfo *> node_inputs(const OrtNode *node) {
    return fetch_items(node, ort_api().Node_GetNumInputs, ort_api().Node_GetInputs);
}

std::vector<const OrtValueInfo *> node_outputs(const OrtNode *node) {
    return fetch_items(node, ort_api().Node_GetNumOutputs, ort_api().Node_GetOutputs);
}

// Build-time constant operands that the builder reads directly.
bool is_metadata_input(const OrtNode *node, std::size_t index) {
    std::string op = node_op_type(node);
    if (index == 0) {
        return false; // input 0 is always the runtime tensor
    }
    if (op == "Resize" || op == "Slice") {
        return true;
    }
    if (op == "Clip") {
        return index == 1 || index == 2;
    }
    if (op == "Reshape" || op == "ReduceMean" || op == "Gather") {
        return index == 1;
    }
    return false;
}

const char *value_name_ref(const OrtValueInfo *value) {
    const char *name = nullptr;
    check_call(ort_api().GetValueInfoName(value, &name));
    return name;
}

std::string value_name(const OrtValueInfo *value) {
    return value_name_ref(value);
}

bool value_is_tensor(const OrtValueInfo *value) {
    return tensor_info_of(value) != nullptr;
}

ONNXTensorElementDataType value_elem_type(const OrtValueInfo *value) {
    const OrtTensorTypeAndShapeInfo *tensor_info = tensor_info_of(value);
    if (tensor_info == nullptr) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    check_call(ort_api().GetTensorElementType(tensor_info, &elem_type));
    return elem_type;
}

std::vector<std::int64_t> value_shape(const OrtValueInfo *value) {
    const OrtTensorTypeAndShapeInfo *tensor_info = tensor_info_of(value);
    if (tensor_info == nullptr) {
        return {};
    }
    std::size_t rank = 0;
    check_call(ort_api().GetDimensionsCount(tensor_info, &rank));

    std::vector<std::int64_t> shape(rank);
    if (rank > 0) {
        check_call(ort_api().GetDimensions(tensor_info, shape.data(), rank));
    }
    return shape;
}

bool value_is_param(const OrtValueInfo *value) {
    bool is_constant = false;
    check_call(ort_api().ValueInfo_IsConstantInitializer(value, &is_constant));
    return is_constant;
}

const OrtValue *value_initializer(const OrtValueInfo *value) {
    const OrtValue *initializer = nullptr;
    check_call(ort_api().ValueInfo_GetInitializerValue(value, &initializer));
    return initializer;
}

std::size_t ctx_inputs_count(const OrtKernelContext *ctx) {
    std::size_t inputs_count = 0;
    check_call(ort_api().KernelContext_GetInputCount(ctx, &inputs_count));
    return inputs_count;
}

const OrtValue *ctx_input_at(const OrtKernelContext *ctx, std::size_t idx) {
    const OrtValue *value = nullptr;
    check_call(ort_api().KernelContext_GetInput(ctx, idx, &value));
    return value;
}

OrtValue *ctx_output_at(OrtKernelContext *ctx, std::size_t idx, const std::vector<std::int64_t> &shape) {
    OrtValue *out_value = nullptr;
    check_call(ort_api().KernelContext_GetOutput(ctx, idx, shape.data(), shape.size(), &out_value));
    return out_value;
}

namespace {

// Attribute handle by name, or nullptr when absent.
// Throws on a type mismatch. A failure status from Node_GetAttributeByName means "no such
// attribute", not an error.
const OrtOpAttr *find_attr(const OrtNode *node, const char *name, OrtOpAttrType expected) {
    const OrtOpAttr *attr = nullptr;
    OrtStatus *status = ort_api().Node_GetAttributeByName(node, name, &attr);
    if (status != nullptr) {
        ort_api().ReleaseStatus(status);
        return nullptr;
    }
    if (attr == nullptr) {
        return nullptr;
    }
    OrtOpAttrType type = ORT_OP_ATTR_UNDEFINED;
    check_call(ort_api().OpAttr_GetType(attr, &type));
    TT_FATAL(type == expected, "attribute '{}' has type {}, expected {}", name, as<int>(type), as<int>(expected));
    return attr;
}

// Two-call ReadOpAttr pattern for array-valued attributes. ReadOpAttr's size
// out-param is in BYTES (both the required size reported by the probing call
// and the amount written on success) — not elements.
template <typename T> std::vector<T> read_attr_array(const OrtOpAttr *attr, OrtOpAttrType type) {
    std::size_t bytes = 0;
    // The probing call fails by design; it reports the required byte count.
    OrtStatus *status = ort_api().ReadOpAttr(attr, type, nullptr, 0, &bytes);
    if (status != nullptr) {
        ort_api().ReleaseStatus(status);
    }
    std::vector<T> values(bytes / sizeof(T));
    if (!values.empty()) {
        check_call(ort_api().ReadOpAttr(attr, type, values.data(), bytes, &bytes));
    }
    return values;
}

} // namespace

bool node_has_attr(const OrtNode *node, const char *name) {
    const OrtOpAttr *attr = nullptr;
    OrtStatus *status = ort_api().Node_GetAttributeByName(node, name, &attr);
    if (status != nullptr) {
        ort_api().ReleaseStatus(status);
        return false;
    }
    return attr != nullptr;
}

std::int64_t node_attr_i64(const OrtNode *node, const char *name, std::int64_t fallback) {
    const OrtOpAttr *attr = find_attr(node, name, ORT_OP_ATTR_INT);
    if (attr == nullptr) {
        return fallback;
    }
    std::int64_t value = 0;
    std::size_t size = 0;
    check_call(ort_api().ReadOpAttr(attr, ORT_OP_ATTR_INT, &value, sizeof(value), &size));
    return value;
}

float node_attr_f32(const OrtNode *node, const char *name, float fallback) {
    const OrtOpAttr *attr = find_attr(node, name, ORT_OP_ATTR_FLOAT);
    if (attr == nullptr) {
        return fallback;
    }
    float value = 0.0F;
    std::size_t size = 0;
    check_call(ort_api().ReadOpAttr(attr, ORT_OP_ATTR_FLOAT, &value, sizeof(value), &size));
    return value;
}

std::string node_attr_string(const OrtNode *node, const char *name, const std::string &fallback) {
    const OrtOpAttr *attr = find_attr(node, name, ORT_OP_ATTR_STRING);
    if (attr == nullptr) {
        return fallback;
    }
    auto chars = read_attr_array<char>(attr, ORT_OP_ATTR_STRING);
    std::string value(chars.begin(), chars.end());
    while (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

std::vector<std::int64_t> node_attr_i64s(const OrtNode *node, const char *name, std::vector<std::int64_t> fallback) {
    const OrtOpAttr *attr = find_attr(node, name, ORT_OP_ATTR_INTS);
    if (attr == nullptr) {
        return fallback;
    }
    return read_attr_array<std::int64_t>(attr, ORT_OP_ATTR_INTS);
}

const void *tensor_data(const OrtValue *tensor) {
    const void *data = nullptr;
    check_call(ort_api().GetTensorData(tensor, &data));
    return data;
}

void *tensor_mutable_data(OrtValue *tensor) {
    void *data = nullptr;
    check_call(ort_api().GetTensorMutableData(tensor, &data));
    return data;
}

ONNXTensorElementDataType tensor_elem_type(const OrtValue *tensor) {
    OrtTensorTypeAndShapeInfo *info = nullptr;
    check_call(ort_api().GetTensorTypeAndShape(tensor, &info));
    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    OrtStatus *status = ort_api().GetTensorElementType(info, &elem_type);
    ort_api().ReleaseTensorTypeAndShapeInfo(info);
    check_call(status);
    return elem_type;
}

std::vector<std::int64_t> tensor_shape(const OrtValue *tensor) {
    OrtTensorTypeAndShapeInfo *info = nullptr;
    check_call(ort_api().GetTensorTypeAndShape(tensor, &info));
    std::size_t dim_count = 0;
    OrtStatus *status = ort_api().GetDimensionsCount(info, &dim_count);
    if (status == nullptr) {
        std::vector<std::int64_t> dims(dim_count);
        status = ort_api().GetDimensions(info, dims.data(), dim_count);
        if (status == nullptr) {
            ort_api().ReleaseTensorTypeAndShapeInfo(info);
            return dims;
        }
    }
    ort_api().ReleaseTensorTypeAndShapeInfo(info);
    check_call(status);
    return {};
}

std::vector<std::int64_t> constant_i64s(const OrtValueInfo *value) {
    const OrtValue *init = value_initializer(value);
    TT_FATAL(init != nullptr, "expected a constant initializer");
    TT_FATAL(tensor_elem_type(init) == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "expected an int64 constant");
    std::int64_t count = 1;
    for (std::int64_t dim : tensor_shape(init)) {
        count *= dim;
    }
    const auto *data = static_cast<const std::int64_t *>(tensor_data(init));
    return {data, data + count};
}

std::vector<float> constant_f32s(const OrtValueInfo *value) {
    const OrtValue *init = value_initializer(value);
    TT_FATAL(init != nullptr, "expected a constant initializer");
    TT_FATAL(tensor_elem_type(init) == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "expected a float32 constant");
    std::int64_t count = 1;
    for (std::int64_t dim : tensor_shape(init)) {
        count *= dim;
    }
    const auto *data = static_cast<const float *>(tensor_data(init));
    return {data, data + count};
}

const OrtMemoryDevice *tensor_device(const OrtValue *tensor) {
    const OrtMemoryInfo *info = nullptr;
    check_call(ort_api().GetTensorMemoryInfo(tensor, &info));
    return ep_api().MemoryInfo_GetMemoryDevice(info);
}

OpKind op_kind(const OrtNode *node) {
    if (auto domain = node_domain(node); !domain.empty() && domain != "ai.onnx") {
        return OpKind::Unsupported;
    }
    auto it = op_kinds.find(node_op_type(node));
    return it == op_kinds.end() ? OpKind::Unsupported : it->second;
}

bool is_node_supported(const OrtNode *node) {
    auto declined = [&](const std::string &reason) {
        if (log_fallback_enabled()) {
            log_error(tt::LogAlways, "declining node '{}' ({}): {}", node_name(node), node_op_type(node), reason);
        }
        return false;
    };

    OpKind kind = op_kind(node);
    if (kind == OpKind::Unsupported) {
        return declined("op not supported");
    }
    auto outputs = node_outputs(node);
    if (outputs.size() != 1) {
        return declined(std::format("{} outputs (only single-output ops are supported)", outputs.size()));
    }
    if (!is_static_supported(outputs[0])) {
        return declined("output has a dynamic shape or unsupported element type");
    }
    auto inputs = node_inputs(node);
    if (!std::ranges::all_of(inputs, [](const OrtValueInfo *input) {
            return input == nullptr || is_static_supported(input); // nullptr = omitted optional input
        })) {
        return declined("an input has a dynamic shape or unsupported element type");
    }
    // Per-op attr/dtype limits are builder TT_FATALs; here we screen op kind, single output, and static shapes/dtypes.
    return true;
}

bool is_ep_context_node(const OrtNode *node) {
    return node_op_type(node) == "EPContext" && node_domain(node) == "com.microsoft" &&
           node_attr_string(node, "source", "") == ep_name;
}

}; // namespace tt::crank::onnx
