// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// conv2d weight-preparation cache for the EmitC execution path.
//
// With TTNNPrepareConv2dWeightsAndBias disabled, conv2d weights arrive as
// SystemMemory tensors. ttnn::conv2d calls prepare_conv_weights_biases_and_
// move_to_device on every invocation, which issues an EnqueueWriteBuffer.
// Device writes are illegal inside trace capture, causing:
//   TT_FATAL: Writes are not supported during trace capture
//
// The generated run_and_capture_trace_0_forward calls trace_0_forward twice:
// once before BeginTraceCapture (writes allowed) and once inside the captured
// region (writes forbidden). conv2d_cached intercepts both calls. On the first
// (cache miss) it uses return_weights_and_bias=true, caches the prepared DRAM
// tensors by weight.tensor_id, and returns normally. On the second (cache hit)
// it passes the cached DRAM tensors with return_weights_and_bias=false — no
// host-to-device transfer occurs inside the trace body.
//
// Mirrors conv2dPrepareCache in runtime/lib/ttnn/operations/conv/conv2d.cpp
// but at generated-code level so it covers the EmitC path.

#include <optional>
#include <unordered_map>
#include <variant>

#include "ttnn/operations/conv/conv2d/conv2d.hpp"

namespace tt::runtime::ttnn::emitc {

struct PreparedConv2dWeights {
    ::ttnn::Tensor weight;
    std::optional<::ttnn::Tensor> bias;
};

inline std::unordered_map<uint64_t, PreparedConv2dWeights> &
conv2d_weight_cache() {
    static thread_local auto *cache =
        new std::unordered_map<uint64_t, PreparedConv2dWeights>();
    return *cache;
}

inline ::ttnn::Conv2dResultWithOptions conv2d_cached(
    const ::ttnn::Tensor &input_tensor,
    const ::ttnn::Tensor &weight_tensor,
    ::ttnn::distributed::MeshDevice *device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride = std::array<uint32_t, 2>{1, 1},
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding =
        std::array<uint32_t, 2>{0, 0},
    std::array<uint32_t, 2> dilation = std::array<uint32_t, 2>{1, 1},
    uint32_t groups = 1,
    const std::optional<const ::ttnn::DataType> &dtype = std::nullopt,
    const std::optional<const ::ttnn::Tensor> &bias_tensor = std::nullopt,
    const std::optional<const ::ttnn::Conv2dConfig> &conv_config_ =
        std::nullopt,
    const std::optional<const ::ttnn::DeviceComputeKernelConfig>
        &compute_config_ = std::nullopt,
    const std::optional<const ::ttnn::MemoryConfig> &memory_config_ =
        std::nullopt,
    const std::optional<const ::ttnn::Conv2dSliceConfig> &dram_slice_config_ =
        std::nullopt) {

    uint64_t key = weight_tensor.tensor_id;
    auto &cache = conv2d_weight_cache();
    auto it = cache.find(key);

    if (it != cache.end()) {
        // Cache hit: pass cached DRAM tensors — no EnqueueWriteBuffer issued.
        return ::ttnn::conv2d(
            input_tensor, it->second.weight, device, in_channels, out_channels,
            batch_size, input_height, input_width, kernel_size, stride, padding,
            dilation, groups, dtype, it->second.bias, conv_config_,
            compute_config_, memory_config_, dram_slice_config_,
            /*return_output_dim=*/false,
            /*return_weights_and_bias=*/false);
    }

    // Cache miss: pre-trace warmup call — writes allowed here.
    auto result = ::ttnn::conv2d(
        input_tensor, weight_tensor, device, in_channels, out_channels,
        batch_size, input_height, input_width, kernel_size, stride, padding,
        dilation, groups, dtype, bias_tensor, conv_config_, compute_config_,
        memory_config_, dram_slice_config_,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/true);

    // Variant alt 2: tuple<Tensor, tuple<Tensor /*weight*/, optional<Tensor> /*bias*/>>
    auto &[out, wb] = std::get<2>(result);
    cache[key] = PreparedConv2dWeights{std::get<0>(wb), std::get<1>(wb)};

    // Alt 0 so ::std::get<0>(conv2d_cached(...)) in generated code extracts out.
    return ::ttnn::Conv2dResultWithOptions{out};
}

} // namespace tt::runtime::ttnn::emitc
