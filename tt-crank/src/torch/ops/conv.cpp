#include "torch/ops/builders.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LLVM.h>
#include <torch/library.h>
#include <ttmlir/Dialect/TTIR/IR/TTIROps.h>

#include "cast.hpp"
#include "torch/backend.hpp"
#include "torch/tensor.hpp"
#include "torch/ttir_module_builder.hpp"

namespace tt::kurbla::torch_backend {

namespace {

at::Tensor tt_convolution(const at::Tensor &input_in, const at::Tensor &weight_in,
                          const std::optional<at::Tensor> &bias_in, at::IntArrayRef stride, at::IntArrayRef padding,
                          at::IntArrayRef dilation, bool transposed, at::IntArrayRef /*output_padding*/,
                          int64_t groups) {
    TORCH_CHECK(is_tt(input_in), "tt-kurbla aten::convolution: input must be on tt backend");
    TORCH_CHECK(!transposed, "tt-kurbla aten::convolution: transposed convolution not supported");

    if (bias_in.has_value() && bias_in->defined()) {
        const auto [input, weight, bias] = align_on_tt(input_in, weight_in, *bias_in);
        auto mb = ModuleBuilder::init({spec_for(input), spec_for(weight), spec_for(bias)});
        auto [promoted, inp_v, w_v, b_v] = promote_inputs(mb, input, weight, bias);
        auto result = build_conv2d(mb, inp_v, w_v, b_v, stride, padding, dilation, groups);
        auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
        std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
        auto module_op = std::move(mb).finalize({result});
        auto outputs = compile_and_run(std::move(module_op), {input, weight, bias});
        return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
    }

    const auto [input, weight] = align_on_tt(input_in, weight_in);
    auto mb = ModuleBuilder::init({spec_for(input), spec_for(weight)});
    auto [promoted, inp_v, w_v] = promote_inputs(mb, input, weight);
    auto result = build_conv2d(mb, inp_v, w_v, mlir::Value{}, stride, padding, dilation, groups);
    auto out_shape_ref = mlir::cast<mlir::RankedTensorType>(result.getType()).getShape();
    std::vector<int64_t> out_shape(out_shape_ref.begin(), out_shape_ref.end());
    auto module_op = std::move(mb).finalize({result});
    auto outputs = compile_and_run(std::move(module_op), {input, weight});
    return wrap_tt_tensor(std::move(outputs[0]), out_shape, promoted);
}

} // namespace

mlir::Value build_conv2d(ModuleBuilder &mb, mlir::Value input, mlir::Value weight, mlir::Value bias,
                         llvm::ArrayRef<int64_t> stride, llvm::ArrayRef<int64_t> padding,
                         llvm::ArrayRef<int64_t> dilation, int64_t groups) {
    auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto weight_type = mlir::cast<mlir::RankedTensorType>(weight.getType());
    auto shape = input_type.getShape();   // NCHW: [N, C_in, H, W]
    auto wshape = weight_type.getShape(); // OIHW: [C_out, C_in/groups, kH, kW]

    // NOLINTBEGIN
    int64_t kH = wshape[2], kW = wshape[3];
    int64_t pH = padding[0], pW = padding[1];
    int64_t dH = dilation[0], dW = dilation[1];
    int64_t sH = stride[0], sW = stride[1];
    int64_t H_out = (shape[2] + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    int64_t W_out = (shape[3] + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    // NOLINTEND

    auto elem_type = input_type.getElementType();

    // Reshape 1D bias (C_out,) so the channel sits at channel_dim=1: (1, C_out, 1, 1).
    // The TTIR Conv2dOp verifier reads the bias output-channel count from channel_dim.
    mlir::Value bias_4d;
    if (bias) {
        bias_4d = build_reshape(mb, bias, {1, wshape[0], 1, 1});
    }

    auto stride_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(sH), as<int32_t>(sW)});
    // Symmetric padding: [top, left, bottom, right] = [pH, pW, pH, pW].
    auto padding_attr =
        mb.attrs().getDenseI32ArrayAttr({as<int32_t>(pH), as<int32_t>(pW), as<int32_t>(pH), as<int32_t>(pW)});
    auto dilation_attr = mb.attrs().getDenseI32ArrayAttr({as<int32_t>(dH), as<int32_t>(dW)});

    // NCHW output shape: [N, C_out, H_out, W_out]; dims: batch=0, channel=1, height=2, width=3.
    auto result_type = mlir::RankedTensorType::get({shape[0], wshape[0], H_out, W_out}, elem_type);
    return mb
        .create<mlir::tt::ttir::Conv2dOp>(result_type, input, weight, bias_4d, stride_attr, padding_attr, dilation_attr,
                                          as<uint32_t>(groups), /*batch_dim=*/uint64_t{0},
                                          /*height_dim=*/uint64_t{2}, /*width_dim=*/uint64_t{3},
                                          /*channel_dim=*/uint64_t{1})
        .getResult();
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("convolution_overrideable", TORCH_FN(tt_convolution));
}

} // namespace tt::kurbla::torch_backend
