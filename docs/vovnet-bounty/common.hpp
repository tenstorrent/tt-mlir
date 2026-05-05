#pragma once

#include <iostream>
#include <chrono>
#include <map>
#include <string>
#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include "ttnn-precompiled.hpp"

#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/trace.hpp"

// Tracy zone marker. tt-metal is built with TRACY_ENABLE and emits device-side
// per-op zones from inside the library — that's where the real per-kernel time
// shows up in the Tracy GUI. The shipped libtracy.so doesn't export the client
// API (Tracy is linked privately into tt-metal), so we keep our wrapper as a
// no-op rather than adding misleading host-side zones from this binary.
// Capture with: `tracy-capture -o trace.tracy` while this binary runs, then
// open trace.tracy in the Tracy GUI.
#define TRACY_ZONE(name)

struct CachedConvConfig {
    bool valid = false;
    bool use_fallback = false;
    ttnn::operations::conv::Conv2dParallelizationConfig parallel_config;
    ttnn::operations::conv::Conv2dBlockConfig block_config;
    tt::tt_metal::MemoryConfig conv_out_memory_config;
    ttnn::operations::sliding_window::SlidingWindowConfig sliding_window_config;
    ttnn::Conv2dConfig resolved_conv_config;
    std::array<uint32_t, 4> input_tensor_shape;
    uint32_t output_height;
    uint32_t output_width;
};

// One-shot gate for conv weight prep (the only prep that needs runtime input
// shapes). VoVNet flips this true after the first forward() pass populates
// g_prepared_convs / g_conv_cache. Subsequent passes take the fast prim path.
inline bool g_convs_prepared = false;

// Per-conv cache (sharding/halo/block configs) keyed by conv index in forward().
inline CachedConvConfig g_conv_cache[41];

// Cached results for the 58 const-eval dispatch sites in forward(). Each slot
// is filled lazily on the first call; subsequent calls reuse the stored tensor.
inline std::array<std::vector<ttnn::Tensor>, 58> g_const_eval_cache;

// Prepared (preprocessed-for-device) conv weight + optional bias for each of
// the 41 convs in the model. Filled on the first forward() pass; reused after.
struct PreparedConv {
    ttnn::Tensor weight;
    std::optional<ttnn::Tensor> bias;
};
inline std::array<PreparedConv, 41> g_prepared_convs;

// Forward declarations.
std::vector<ttnn::Tensor> const_eval_permute(std::vector<ttnn::Tensor> v1);
std::vector<ttnn::Tensor> const_eval_reshape_scalar(std::vector<ttnn::Tensor> v1);
std::vector<ttnn::Tensor> const_eval_passthrough(std::vector<ttnn::Tensor> v1);

// Cache-aware dispatcher: returns cached tensor for `slot`, or runs `fn(input)`
// and stores it on first call. Defined in consteval.cpp.
ttnn::Tensor cached_const_eval(
    int slot,
    std::vector<ttnn::Tensor> (*fn)(std::vector<ttnn::Tensor>),
    const ttnn::Tensor& input);

void fold_bn_scale_into_weight(
    ttnn::Tensor& weight,
    const ttnn::Tensor& bn_scale,
    uint32_t c_out,
    ttnn::distributed::MeshDevice* device);

ttnn::Tensor conv2d_prim_cached(
    const ttnn::Tensor& input, const ttnn::Tensor& weight,
    const std::optional<ttnn::Tensor>& bias,
    ttnn::distributed::MeshDevice* device,
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    std::array<uint32_t, 2> kernel_size, std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding, std::array<uint32_t, 2> dilation,
    uint32_t groups, const ttnn::DeviceComputeKernelConfig& compute_config,
    int conv_idx,
    std::optional<ttnn::operations::unary::UnaryWithParam> activation = std::nullopt);

// Single entry point for a conv site: handles prep on first call, dispatches
// to cached path on subsequent calls. "Legacy" goes through ttnn::conv2d (used
// for the first 5 convs); the default goes through the prim-cached fast path.
// `activation`, when set, is fused into the conv kernel (avoids a separate
// post-conv unary op). Pass the BN bias as raw_bias to fuse the post-conv
// add too — conv kernel does `activation(W*x + bias)` in one pass.
ttnn::Tensor cached_conv2d_legacy(
    int conv_idx,
    const ttnn::Tensor& input, const ttnn::Tensor& raw_weight,
    const std::optional<ttnn::Tensor>& raw_bias,
    ttnn::distributed::MeshDevice* device,
    uint32_t in_ch, uint32_t out_ch, uint32_t batch,
    uint32_t in_h, uint32_t in_w,
    std::array<uint32_t, 2> kernel, std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding, std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<ttnn::operations::unary::UnaryWithParam> activation = std::nullopt);

ttnn::Tensor cached_conv2d(
    int conv_idx,
    const ttnn::Tensor& input, const ttnn::Tensor& raw_weight,
    const std::optional<ttnn::Tensor>& raw_bias,
    ttnn::distributed::MeshDevice* device,
    uint32_t in_ch, uint32_t out_ch, uint32_t batch,
    uint32_t in_h, uint32_t in_w,
    std::array<uint32_t, 2> kernel, std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding, std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<ttnn::operations::unary::UnaryWithParam> activation = std::nullopt);

std::vector<ttnn::Tensor> create_inputs_for_forward();
std::vector<ttnn::Tensor> forward(std::vector<ttnn::Tensor> v1);

// One-time weight prep, factored out of forward(). Both must be invoked before
// forward() is called: prepare_constevals fills g_const_eval_cache; prepare_bn_folds
// folds BN scale into the conv weight tensors in `inputs` in place.
void prepare_constevals(const std::vector<ttnn::Tensor>& inputs);
void prepare_bn_folds(std::vector<ttnn::Tensor>& inputs,
                      ttnn::distributed::MeshDevice* device);

// Thin model facade. Constructor runs all input-independent weight prep
// (consteval permutes + BN folding) once. operator() runs forward(); the first
// call also primes the per-conv shard/halo cache (input-shape-dependent), then
// flips g_convs_prepared so subsequent calls take the fast prim path.
//
// Once the program cache is warm, call capture_trace() to record the dispatch
// sequence into a metal trace. Subsequent replay() calls re-issue the recorded
// commands without per-op host dispatch overhead. The output Tensor returned
// by replay() shares its device buffer with the captured output, so each
// replay updates that buffer in place.
class VoVNet {
 public:
  // raw_weights is the full input vector accepted by forward(): index 0 is the
  // image placeholder (overwritten by operator()'s argument), 1..N are weights.
  VoVNet(ttnn::distributed::MeshDevice* device,
         std::vector<ttnn::Tensor> raw_weights);
  ~VoVNet();
  ttnn::Tensor operator()(const ttnn::Tensor& image);

  // Captures the dispatch sequence of one forward() pass into a metal trace.
  // Caller must have already run operator() at least twice (prep + warmup) so
  // the program cache and conv shard/halo cache are populated.
  void capture_trace();

  // Replays the previously-captured trace. Returns the same Tensor handle each
  // call; its device buffer is updated in place by the trace replay.
  ttnn::Tensor replay();

 private:
  ttnn::distributed::MeshDevice* device_;
  std::vector<ttnn::Tensor> inputs_;  // [image, w0, w1, ...]; image overwritten per call
  bool convs_prepared_ = false;       // false until first forward() pass completes
  std::optional<ttnn::MeshTraceId> trace_id_;
  std::optional<ttnn::Tensor> traced_output_;
};
