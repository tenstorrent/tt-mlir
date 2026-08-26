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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("convolution_overrideable", TORCH_FN(tt_convolution));
}

} // namespace tt::kurbla::torch_backend
