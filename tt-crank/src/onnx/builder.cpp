// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

#include "assert.hpp"
#include "cast.hpp"
#include "graph.hpp"

namespace tt::crank::onnx {

namespace ttc = ::tt::crank;

namespace {

mlir::RankedTensorType tensor_type_of(mlir::Value value) {
    return mlir::cast<mlir::RankedTensorType>(value.getType());
}

llvm::ArrayRef<std::int64_t> shape_of(mlir::Value value) {
    return tensor_type_of(value).getShape();
}

bool is_float_tensor(mlir::Value value) {
    return mlir::isa<mlir::FloatType>(tensor_type_of(value).getElementType());
}

} // namespace

OnnxModuleBuilder::OnnxModuleBuilder(const Args &args) : m_mb{ModuleBuilder::init(args.specs, args.roles)} {
    for (std::size_t i = 0; i < args.names.size(); ++i) {
        m_values[args.names[i]] = m_mb.args()[i];
    }
}

mlir::Value OnnxModuleBuilder::operand(const OrtNode *node, std::size_t index) {
    auto inputs = node_inputs(node);
    if (index >= inputs.size() || inputs[index] == nullptr) {
        return {};
    }
    auto name = value_name(inputs[index]);
    auto it = m_values.find(name);
    TT_FATAL(it != m_values.end(), "{} node '{}': input '{}' has no emitted value", node_op_type(node), node_name(node),
             name);
    return it->second;
}

mlir::Value OnnxModuleBuilder::required_operand(const OrtNode *node, std::size_t index) {
    mlir::Value value = operand(node, index);
    TT_FATAL(value, "{} node '{}': missing required input #{}", node_op_type(node), node_name(node), index);
    return value;
}

void OnnxModuleBuilder::set_output(const OrtNode *node, mlir::Value result) {
    auto outputs = node_outputs(node);
    TT_FATAL(!outputs.empty(), "{} node '{}': no outputs", node_op_type(node), node_name(node));
    m_values[value_name(outputs[0])] = result;
}

std::pair<mlir::Value, mlir::Value> OnnxModuleBuilder::align_ranks(mlir::Value lhs, mlir::Value rhs) {
    auto lhs_shape = shape_of(lhs);
    auto rhs_shape = shape_of(rhs);
    if (lhs_shape.size() == rhs_shape.size()) {
        return {lhs, rhs};
    }
    auto pad_rank = [this](mlir::Value value, std::size_t rank) {
        auto shape = shape_of(value);
        llvm::SmallVector<std::int64_t> padded(rank - shape.size(), 1);
        padded.append(shape.begin(), shape.end());
        return ttc::build_reshape(m_mb, value, padded);
    };
    if (lhs_shape.size() < rhs_shape.size()) {
        return {pad_rank(lhs, rhs_shape.size()), rhs};
    }
    return {lhs, pad_rank(rhs, lhs_shape.size())};
}

// ---- elementwise ------------------------------------------------------------

template <mlir::Value (*BuildFn)(ModuleBuilder &, mlir::Value, mlir::Value)>
void OnnxModuleBuilder::build_binary(const OrtNode *node) {
    auto [lhs, rhs] = align_ranks(required_operand(node, 0), required_operand(node, 1));
    set_output(node, BuildFn(m_mb, lhs, rhs));
}

template <mlir::Value (*BuildFn)(ModuleBuilder &, mlir::Value)>
void OnnxModuleBuilder::build_unary(const OrtNode *node) {
    set_output(node, BuildFn(m_mb, required_operand(node, 0)));
}

// Add/Sub get named node builders instead of going through build_binary:
// the engine's build_add/build_sub carry torch's defaulted `alpha` parameter,
// and defaults don't survive into the template's function-pointer argument.
void OnnxModuleBuilder::build_add(const OrtNode *node) {
    auto [lhs, rhs] = align_ranks(required_operand(node, 0), required_operand(node, 1));
    set_output(node, ttc::build_add(m_mb, lhs, rhs));
}

void OnnxModuleBuilder::build_sub(const OrtNode *node) {
    auto [lhs, rhs] = align_ranks(required_operand(node, 0), required_operand(node, 1));
    set_output(node, ttc::build_sub(m_mb, lhs, rhs));
}

void OnnxModuleBuilder::build_div(const OrtNode *node) {
    auto [lhs, rhs] = align_ranks(required_operand(node, 0), required_operand(node, 1));
    // ttir.div lacks integer trunc semantics; ONNX permits integer operands, we don't.
    TT_FATAL(is_float_tensor(lhs), "Div node '{}': integer operands not supported", node_name(node));
    set_output(node, ttc::build_div(m_mb, lhs, rhs));
}

void OnnxModuleBuilder::build_softmax(const OrtNode *node) {
    // Pre-13 Softmax has flattened-2D semantics — a different op.
    TT_FATAL(node_since_version(node) >= 13, "Softmax node '{}': opset {} < 13 not supported", node_name(node),
             node_since_version(node));
    set_output(node, ttc::build_softmax(m_mb, required_operand(node, 0), node_attr_i64(node, "axis", -1)));
}

// ---- matmul family --------------------------------------------------------------

void OnnxModuleBuilder::build_matmul(const OrtNode *node) {
    mlir::Value lhs = required_operand(node, 0);
    mlir::Value rhs = required_operand(node, 1);
    auto lhs_shape = shape_of(lhs);
    auto rhs_shape = shape_of(rhs);
    TT_FATAL(lhs_shape.size() >= 2 && rhs_shape.size() >= 2, "MatMul node '{}': 1-D operands not supported",
             node_name(node));
    // ttnn's matmul is float-only; ONNX permits integer operands, we don't.
    TT_FATAL(is_float_tensor(lhs), "MatMul node '{}': integer operands not supported", node_name(node));

    // ONNX MatMul broadcasts batch dims numpy-style, but ttnn's matmul kernel
    // can't do that at run time (it requires equal batches, or a plain 2-D
    // rhs). Materialize the broadcast explicitly and hand the engine
    // equal-batch operands; a 2-D rhs stays 2-D (native, and it keeps the
    // engine's k-flatten fast path).
    auto batch = ttc::broadcast_shape(lhs_shape.drop_back(2), rhs_shape.drop_back(2));
    auto with_batch = [&batch](llvm::ArrayRef<std::int64_t> shape) {
        llvm::SmallVector<std::int64_t> out{batch.begin(), batch.end()};
        out.append(shape.end() - 2, shape.end());
        return out;
    };
    if (lhs_shape.drop_back(2) != llvm::ArrayRef<std::int64_t>(batch)) {
        lhs = ttc::build_broadcast(m_mb, lhs, with_batch(lhs_shape));
    }
    if (rhs_shape.size() > 2 && rhs_shape.drop_back(2) != llvm::ArrayRef<std::int64_t>(batch)) {
        rhs = ttc::build_broadcast(m_mb, rhs, with_batch(rhs_shape));
    }
    set_output(node, ttc::build_matmul(m_mb, lhs, rhs));
}

void OnnxModuleBuilder::build_gemm(const OrtNode *node) {
    double alpha = as<double>(node_attr_f32(node, "alpha", 1.0F));
    double beta = as<double>(node_attr_f32(node, "beta", 1.0F));
    bool trans_a = node_attr_i64(node, "transA", 0) != 0;
    bool trans_b = node_attr_i64(node, "transB", 0) != 0;

    mlir::Value a = required_operand(node, 0);
    mlir::Value b = required_operand(node, 1);
    mlir::Value c = operand(node, 2);
    TT_FATAL(shape_of(a).size() == 2 && shape_of(b).size() == 2, "Gemm node '{}': operands must be 2-D",
             node_name(node));
    // ttnn's matmul is float-only; ONNX permits integer operands, we don't.
    TT_FATAL(is_float_tensor(a), "Gemm node '{}': integer operands not supported", node_name(node));

    bool c_is_1d = c && shape_of(c).size() == 1;
    if (!trans_a && trans_b && alpha == 1.0 && (!c || (beta == 1.0 && c_is_1d))) {
        set_output(node, ttc::build_linear(m_mb, a, b, c));
        return;
    }

    if (trans_a) {
        a = ttc::build_transpose(m_mb, a, 0, 1);
    }
    if (trans_b) {
        b = ttc::build_transpose(m_mb, b, 0, 1);
    }
    mlir::Value y = ttc::build_mm(m_mb, a, b);
    auto elem = tensor_type_of(y).getElementType();
    if (alpha != 1.0) {
        y = ttc::build_mul(m_mb, y, ttc::build_scalar(m_mb, elem, alpha));
    }
    // beta == 0 means C is ignored entirely (BLAS convention, matched by ORT
    // CPU) — never computed as y + 0*C, which would propagate NaN/Inf from C.
    if (c && beta != 0.0) {
        if (beta != 1.0) {
            c = ttc::build_mul(m_mb, c, ttc::build_scalar(m_mb, elem, beta));
        }
        auto [yy, cc] = align_ranks(y, c);
        y = ttc::build_add(m_mb, yy, cc);
    }
    set_output(node, y);
}

// ---- conv / pooling ----------------------------------------------------------------

namespace {

struct SpatialPads {
    llvm::SmallVector<std::int64_t> begins;
    llvm::SmallVector<std::int64_t> ends;
    [[nodiscard]] bool symmetric() const { return begins == ends; }
};

// Resolves explicit pads / auto_pad against the operand's static spatial
// dims. ONNX pads are [x1_begin, x2_begin, ..., x1_end, x2_end].
SpatialPads resolve_pads(const OrtNode *node, llvm::ArrayRef<std::int64_t> input_shape,
                         llvm::ArrayRef<std::int64_t> kernel, llvm::ArrayRef<std::int64_t> strides,
                         llvm::ArrayRef<std::int64_t> dilations) {
    std::size_t spatial = kernel.size();
    auto auto_pad = node_attr_string(node, "auto_pad", "NOTSET");
    auto pads = node_attr_i64s(node, "pads", {});
    SpatialPads out;
    if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        TT_FATAL(std::ranges::all_of(pads, [](std::int64_t p) { return p == 0; }),
                 "{} node '{}': both auto_pad and explicit pads", node_op_type(node), node_name(node));
        for (std::size_t i = 0; i < spatial; ++i) {
            std::int64_t size = input_shape[2 + i];
            std::int64_t effective = ((kernel[i] - 1) * dilations[i]) + 1;
            std::int64_t out_size = (size + strides[i] - 1) / strides[i];
            std::int64_t total = std::max<std::int64_t>(((out_size - 1) * strides[i]) + effective - size, 0);
            std::int64_t small = total / 2;
            std::int64_t large = total - small;
            out.begins.push_back(auto_pad == "SAME_UPPER" ? small : large);
            out.ends.push_back(auto_pad == "SAME_UPPER" ? large : small);
        }
        return out;
    }
    if (auto_pad == "VALID" || pads.empty()) {
        out.begins.assign(spatial, 0);
        out.ends.assign(spatial, 0);
        return out;
    }
    TT_FATAL(auto_pad == "NOTSET", "{} node '{}': auto_pad '{}' not supported", node_op_type(node), node_name(node),
             auto_pad);
    TT_FATAL(pads.size() == 2 * spatial, "{} node '{}': pads length {} doesn't match {} spatial dims",
             node_op_type(node), node_name(node), pads.size(), spatial);
    out.begins.assign(pads.begin(), pads.begin() + as<std::ptrdiff_t>(spatial));
    out.ends.assign(pads.begin() + as<std::ptrdiff_t>(spatial), pads.end());
    return out;
}

} // namespace

void OnnxModuleBuilder::build_conv(const OrtNode *node) {
    mlir::Value input = required_operand(node, 0);
    mlir::Value weight = required_operand(node, 1);
    mlir::Value bias = operand(node, 2);
    auto input_shape = shape_of(input);
    auto weight_shape = shape_of(weight);
    TT_FATAL(input_shape.size() == 4 && weight_shape.size() == 4, "Conv node '{}': only 2-D convolution is supported",
             node_name(node));

    llvm::SmallVector<std::int64_t> kernel{weight_shape[2], weight_shape[3]};
    auto kernel_attr = node_attr_i64s(node, "kernel_shape", {kernel.begin(), kernel.end()});
    TT_FATAL(kernel_attr.size() == 2 && kernel_attr[0] == kernel[0] && kernel_attr[1] == kernel[1],
             "Conv node '{}': kernel_shape doesn't match the weight", node_name(node));

    auto strides = node_attr_i64s(node, "strides", {1, 1});
    auto dilations = node_attr_i64s(node, "dilations", {1, 1});
    std::int64_t groups = node_attr_i64(node, "group", 1);

    auto pads = resolve_pads(node, input_shape, kernel, strides, dilations);
    llvm::SmallVector<std::int64_t> symmetric{pads.begins.begin(), pads.begins.end()};
    if (!pads.symmetric()) {
        // ttir.conv2d takes symmetric padding; fold asymmetric pads into an
        // explicit zero-pad of the spatial dims first.
        input = ttc::build_pad(m_mb, input, {0, 0, pads.begins[0], pads.begins[1]}, {0, 0, pads.ends[0], pads.ends[1]},
                               0.0);
        symmetric = {0, 0};
    }
    set_output(node, ttc::build_conv2d(m_mb, input, weight, bias, strides, symmetric, dilations, groups));
}

void OnnxModuleBuilder::build_max_pool(const OrtNode *node) {
    TT_FATAL(node_attr_i64(node, "ceil_mode", 0) == 0,
             "MaxPool node '{}': ceil_mode=1 not supported (TTIR's ceil output formula lacks ONNX's end-padding "
             "adjustment)",
             node_name(node));
    TT_FATAL(node_attr_i64(node, "storage_order", 0) == 0, "MaxPool node '{}': storage_order=1 not supported",
             node_name(node));

    mlir::Value input = required_operand(node, 0);
    auto input_shape = shape_of(input);
    TT_FATAL(input_shape.size() == 4, "MaxPool node '{}': only 2-D pooling is supported", node_name(node));

    auto kernel = node_attr_i64s(node, "kernel_shape", {});
    TT_FATAL(kernel.size() == 2, "MaxPool node '{}': kernel_shape must be 2-D", node_name(node));
    auto strides = node_attr_i64s(node, "strides", {1, 1});
    auto dilations = node_attr_i64s(node, "dilations", {1, 1});

    auto pads = resolve_pads(node, input_shape, kernel, strides, dilations);
    // Explicit -inf pre-pad (padding never wins a max) whenever the pool op
    // can't express the pads itself: asymmetric, or beyond tt-metal's
    // pad <= kernel/2 limit — past that, tt-mlir decomposes the excess into a
    // ZERO-fill pad, silently corrupting maxima of negative regions.
    bool beyond_half_kernel = pads.begins[0] > kernel[0] / 2 || pads.begins[1] > kernel[1] / 2 ||
                              pads.ends[0] > kernel[0] / 2 || pads.ends[1] > kernel[1] / 2;
    llvm::SmallVector<std::int64_t> symmetric{pads.begins.begin(), pads.begins.end()};
    if (!pads.symmetric() || beyond_half_kernel) {
        input = ttc::build_pad(m_mb, input, {0, 0, pads.begins[0], pads.begins[1]}, {0, 0, pads.ends[0], pads.ends[1]},
                               -std::numeric_limits<double>::infinity());
        symmetric = {0, 0};
    }
    set_output(node, ttc::build_max_pool2d(m_mb, input, kernel, strides, symmetric, dilations, /*ceil_mode=*/false));
}

void OnnxModuleBuilder::build_global_average_pool(const OrtNode *node) {
    mlir::Value input = required_operand(node, 0);
    auto rank = shape_of(input).size();
    TT_FATAL(rank >= 3, "GlobalAveragePool node '{}': input rank {} < 3", node_name(node), rank);
    llvm::SmallVector<std::int64_t> dims;
    for (std::size_t i = 2; i < rank; ++i) {
        dims.push_back(as<std::int64_t>(i));
    }
    set_output(node, ttc::build_mean(m_mb, input, dims, /*keepdim=*/true));
}

void OnnxModuleBuilder::build_reduce_mean(const OrtNode *node) {
    mlir::Value input = required_operand(node, 0);
    auto rank = as<std::int64_t>(shape_of(input).size());

    // Axes: an int64 constant input since opset 18, else the `axes` attribute.
    auto inputs = node_inputs(node);
    std::vector<std::int64_t> axes =
        inputs.size() > 1 && inputs[1] != nullptr ? constant_i64s(inputs[1]) : node_attr_i64s(node, "axes", {});
    if (axes.empty()) {
        // Empty axes: identity when noop_with_empty_axes is set, else reduce all.
        if (node_attr_i64(node, "noop_with_empty_axes", 0) != 0) {
            set_output(node, input);
            return;
        }
        for (std::int64_t i = 0; i < rank; ++i) {
            axes.push_back(i);
        }
    }

    llvm::SmallVector<std::int64_t> dims;
    for (std::int64_t axis : axes) {
        dims.push_back(axis < 0 ? axis + rank : axis);
    }
    std::ranges::sort(dims);
    set_output(node, ttc::build_mean(m_mb, input, dims, node_attr_i64(node, "keepdims", 1) != 0));
}

void OnnxModuleBuilder::build_batch_norm(const OrtNode *node) {
    TT_FATAL(node_attr_i64(node, "training_mode", 0) == 0, "BatchNormalization node '{}': training_mode=1",
             node_name(node));
    TT_FATAL(node_attr_i64(node, "spatial", 1) == 1, "BatchNormalization node '{}': spatial=0 not supported",
             node_name(node));
    mlir::Value input = required_operand(node, 0);
    auto rank = shape_of(input).size();
    // ttir.batch_norm_inference's verifier requires rank 2..5.
    TT_FATAL(rank >= 2 && rank <= 5, "BatchNormalization node '{}': input rank {} outside the supported 2..5",
             node_name(node), rank);
    set_output(node, ttc::build_bn_inference(m_mb, input, required_operand(node, 1), required_operand(node, 2),
                                             required_operand(node, 3), required_operand(node, 4),
                                             node_attr_f32(node, "epsilon", 1e-5F)));
}

// ---- shape ops -------------------------------------------------------------------

void OnnxModuleBuilder::build_flatten(const OrtNode *node) {
    mlir::Value input = required_operand(node, 0);
    auto shape = shape_of(input);
    std::int64_t rank = as<std::int64_t>(shape.size());
    std::int64_t axis = node_attr_i64(node, "axis", 1);
    if (axis < 0) {
        axis += rank;
    }
    // build_reshape needs fully-resolved dims (no -1).
    std::int64_t leading = 1;
    std::int64_t trailing = 1;
    for (std::int64_t i = 0; i < rank; ++i) {
        (i < axis ? leading : trailing) *= shape[as<std::size_t>(i)];
    }
    set_output(node, ttc::build_reshape(m_mb, input, {leading, trailing}));
}

void OnnxModuleBuilder::build_reshape(const OrtNode *node) {
    mlir::Value input = required_operand(node, 0);
    auto in_shape = shape_of(input);
    auto inputs = node_inputs(node);
    TT_FATAL(inputs.size() > 1 && inputs[1] != nullptr, "Reshape node '{}': missing shape input", node_name(node));
    std::vector<std::int64_t> target = constant_i64s(inputs[1]);
    bool allowzero = node_attr_i64(node, "allowzero", 0) != 0;

    std::int64_t total = 1;
    for (std::int64_t dim : in_shape) {
        total *= dim;
    }

    // Resolve ONNX's 0 (copy the input dim, unless allowzero) and -1 (infer to preserve element count);
    // build_reshape needs fully resolved dims.
    std::int64_t known = 1;
    std::ptrdiff_t infer = -1;
    for (std::size_t i = 0; i < target.size(); ++i) {
        if (target[i] == -1) {
            infer = as<std::ptrdiff_t>(i);
        } else {
            if (target[i] == 0 && !allowzero) {
                TT_FATAL(i < in_shape.size(), "Reshape node '{}': dim 0 at index {} has no matching input dim",
                         node_name(node), i);
                target[i] = in_shape[i];
            }
            known *= target[i];
        }
    }
    if (infer >= 0) {
        target[as<std::size_t>(infer)] = total / known;
    }

    llvm::SmallVector<std::int64_t> resolved(target.begin(), target.end());
    set_output(node, ttc::build_reshape(m_mb, input, resolved));
}

void OnnxModuleBuilder::build_transpose(const OrtNode *node) {
    mlir::Value input = required_operand(node, 0);
    auto rank = shape_of(input).size();
    auto perm = node_attr_i64s(node, "perm", {});
    if (perm.empty()) {
        for (std::size_t i = rank; i > 0; --i) {
            perm.push_back(as<std::int64_t>(i - 1));
        }
    }
    for (auto &p : perm) {
        if (p < 0) {
            p += as<std::int64_t>(rank);
        }
    }
    set_output(node, ttc::build_permute(m_mb, input, perm));
}

void OnnxModuleBuilder::build_concat(const OrtNode *node) {
    TT_FATAL(node_has_attr(node, "axis"), "Concat node '{}': missing axis", node_name(node));
    llvm::SmallVector<mlir::Value> inputs;
    for (std::size_t i = 0; i < node_inputs(node).size(); ++i) {
        inputs.push_back(required_operand(node, i));
    }
    set_output(node, ttc::build_cat(m_mb, inputs, node_attr_i64(node, "axis", 0)));
}

void OnnxModuleBuilder::build_identity(const OrtNode *node) {
    set_output(node, required_operand(node, 0));
}

void OnnxModuleBuilder::build_cast(const OrtNode *node) {
    // Onnx cast's `to` is a ONNXTensorElementDataType enum, so must take it as an int value.
    auto to_dtype = to_runtime_dtype(as<ONNXTensorElementDataType>(node_attr_i64(node, "to", 0)));
    auto target = ttc::to_mlir_element_type(*m_mb.attrs().getContext(), to_dtype);
    set_output(node, m_mb.insert_typecast(required_operand(node, 0), target));
}

void OnnxModuleBuilder::build_clip(const OrtNode *node) {
    mlir::Value input = required_operand(node, 0);
    auto inputs = node_inputs(node);
    // opset>=11: min/max are scalar constant inputs 1/2; pre-11: min/max attrs.
    auto bound = [&](std::size_t i, const char *attr) -> std::optional<double> {
        if (inputs.size() > i && inputs[i] != nullptr) {
            return as<double>(constant_f32s(inputs[i]).at(0));
        }
        if (node_has_attr(node, attr)) {
            return as<double>(node_attr_f32(node, attr, 0.0F));
        }
        return std::nullopt;
    };
    set_output(node, ttc::build_clamp(m_mb, input, bound(1, "min"), bound(2, "max")));
}

void OnnxModuleBuilder::build_slice(const OrtNode *node) {
    mlir::Value input = required_operand(node, 0);
    auto in_shape = shape_of(input);
    std::int64_t rank = as<std::int64_t>(in_shape.size());
    auto inputs = node_inputs(node);

    TT_FATAL(inputs.size() >= 3 && inputs[1] != nullptr && inputs[2] != nullptr,
             "Slice node '{}': missing starts/ends input", node_name(node));
    std::vector<std::int64_t> starts = constant_i64s(inputs[1]);
    std::vector<std::int64_t> ends = constant_i64s(inputs[2]);
    std::vector<std::int64_t> axes;
    if (inputs.size() > 3 && inputs[3] != nullptr) {
        axes = constant_i64s(inputs[3]);
    } else {
        for (std::size_t i = 0; i < starts.size(); ++i) {
            axes.push_back(as<std::int64_t>(i));
        }
    }
    std::vector<std::int64_t> steps;
    if (inputs.size() > 4 && inputs[4] != nullptr) {
        steps = constant_i64s(inputs[4]);
    } else {
        steps.assign(starts.size(), 1);
    }

    // Full-rank begins/ends/step (build_slice needs pre-slice, resolved values).
    llvm::SmallVector<std::int64_t> begins(as<std::size_t>(rank), 0);
    llvm::SmallVector<std::int64_t> stops(in_shape.begin(), in_shape.end());
    llvm::SmallVector<std::int64_t> stride(as<std::size_t>(rank), 1);
    for (std::size_t k = 0; k < axes.size(); ++k) {
        TT_FATAL(steps[k] >= 1, "Slice node '{}': reverse/zero step ({}) not supported", node_name(node), steps[k]);
        std::int64_t ax = axes[k] < 0 ? axes[k] + rank : axes[k];
        std::int64_t dim = in_shape[as<std::size_t>(ax)];
        std::int64_t b = starts[k] < 0 ? starts[k] + dim : starts[k];
        std::int64_t e = ends[k] < 0 ? ends[k] + dim : ends[k];
        begins[as<std::size_t>(ax)] = std::clamp<std::int64_t>(b, 0, dim);
        stops[as<std::size_t>(ax)] = std::clamp<std::int64_t>(e, 0, dim);
        stride[as<std::size_t>(ax)] = steps[k];
    }
    set_output(node, ttc::build_slice(m_mb, input, begins, stops, stride));
}

void OnnxModuleBuilder::build_gather(const OrtNode *node) {
    // Only a constant scalar index: slice [idx:idx+1] on `axis` then squeeze (a rank>0 index is unsupported).
    mlir::Value input = required_operand(node, 0);
    auto in_shape = shape_of(input);
    std::int64_t rank = as<std::int64_t>(in_shape.size());
    std::int64_t axis = node_attr_i64(node, "axis", 0);
    if (axis < 0) {
        axis += rank;
    }
    auto inputs = node_inputs(node);
    TT_FATAL(inputs.size() >= 2 && inputs[1] != nullptr, "Gather node '{}': missing index input", node_name(node));
    const OrtValue *idx_init = value_initializer(inputs[1]);
    TT_FATAL(idx_init != nullptr && tensor_shape(idx_init).empty(),
             "Gather node '{}': only a constant scalar index is supported", node_name(node));
    std::int64_t dim = in_shape[as<std::size_t>(axis)];
    std::int64_t idx = constant_i64s(inputs[1]).at(0);
    if (idx < 0) {
        idx += dim;
    }
    llvm::SmallVector<std::int64_t> begins(as<std::size_t>(rank), 0);
    llvm::SmallVector<std::int64_t> ends(in_shape.begin(), in_shape.end());
    llvm::SmallVector<std::int64_t> stride(as<std::size_t>(rank), 1);
    begins[as<std::size_t>(axis)] = idx;
    ends[as<std::size_t>(axis)] = idx + 1;
    mlir::Value sliced = ttc::build_slice(m_mb, input, begins, ends, stride);
    set_output(node, ttc::build_squeeze(m_mb, sliced, axis));
}

void OnnxModuleBuilder::build_node(const OrtNode *node) {
    switch (op_kind(node)) { // clang-format off
        case OpKind::Add: return build_add(node);
        case OpKind::Sub: return build_sub(node);
        case OpKind::Mul: return build_binary<ttc::build_mul>(node);
        case OpKind::Div: return build_div(node);
        case OpKind::Relu: return build_unary<ttc::build_relu>(node);
        case OpKind::Sigmoid: return build_unary<ttc::build_sigmoid>(node);
        case OpKind::Exp: return build_unary<ttc::build_exp>(node);
        case OpKind::Log: return build_unary<ttc::build_log>(node);
        case OpKind::Neg: return build_unary<ttc::build_neg>(node);
        case OpKind::Softmax: return build_softmax(node);
        case OpKind::MatMul: return build_matmul(node);
        case OpKind::Gemm: return build_gemm(node);
        case OpKind::Conv: return build_conv(node);
        case OpKind::MaxPool: return build_max_pool(node);
        case OpKind::GlobalAveragePool: return build_global_average_pool(node);
        case OpKind::ReduceMean: return build_reduce_mean(node);
        case OpKind::BatchNormalization: return build_batch_norm(node);
        case OpKind::Flatten: return build_flatten(node);
        case OpKind::Reshape: return build_reshape(node);
        case OpKind::Transpose: return build_transpose(node);
        case OpKind::Concat: return build_concat(node);
        case OpKind::Identity: return build_identity(node);
        case OpKind::Cast: return build_cast(node);
        case OpKind::Clip: return build_clip(node);
        case OpKind::Slice: return build_slice(node);
        case OpKind::Gather: return build_gather(node);
        case OpKind::Unsupported: break;
    } // clang-format on

    TT_THROW("build_node: no lowering for op '{}' (node '{}')", node_op_type(node), node_name(node));
}

mlir::OwningOpRef<mlir::ModuleOp> OnnxModuleBuilder::finalize(const OrtNode *fused_node) && {
    llvm::SmallVector<mlir::Value> outputs;
    for (const OrtValueInfo *output : node_outputs(fused_node)) {
        auto name = value_name(output);
        auto it = m_values.find(name);
        TT_FATAL(it != m_values.end(), "fused output '{}' was never produced", name);
        outputs.push_back(it->second);
    }
    return std::move(m_mb).finalize(outputs);
}

} // namespace tt::crank::onnx
