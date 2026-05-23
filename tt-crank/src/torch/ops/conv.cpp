#include <cstdint>
#include <optional>

#include <ATen/ATen.h>
#include <torch/library.h>

#include "torch/tensor.hpp"

namespace tt::kurbla::torch_backend {

namespace {

// Torch doesn't seem to support automatic fallback to the cpu kernel for convs - it always
// calls `conv_overrideable` which we need to implement & register.
// We implement it by falling back to the cpu aten conv - for now.
at::Tensor convolution_overrideable(const at::Tensor &input, const at::Tensor &weight,
                                    const std::optional<at::Tensor> &bias, at::IntArrayRef stride,
                                    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
                                    at::IntArrayRef output_padding, int64_t groups) {
    TORCH_CHECK(is_tt(input), "tt-kurbla aten::convolution_overrideable: input must be on the tt backend");
    auto cpu_bias = bias.has_value() && bias->defined() ? std::make_optional(bias->cpu()) : std::nullopt;
    auto cpu_result = at::convolution(input.cpu(), weight.cpu(), cpu_bias, stride, padding, dilation, transposed,
                                      output_padding, groups);
    return cpu_result.to(input.device());
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("convolution_overrideable", TORCH_FN(convolution_overrideable));
}

} // namespace tt::kurbla::torch_backend
