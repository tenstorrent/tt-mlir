#include "common.hpp"

// ---------------------------------------------------------------------------
// Conv2d shape arguments — packed for the cached_conv2d wrappers below.
// ---------------------------------------------------------------------------
namespace {

inline ::ttnn::Conv2dConfig kConvConfig() {
  return ::ttnn::Conv2dConfig{
      .config_tensors_in_dram = true,
      .enable_kernel_stride_folding = false,
      .enable_act_double_buffer = true,
      .enable_weights_double_buffer = true};
}

inline ::ttnn::WormholeComputeKernelConfig kComputeConfig() {
  return ::ttnn::WormholeComputeKernelConfig{
      .math_fidelity = ::MathFidelity::LoFi,
      .math_approx_mode = true,
      .fp32_dest_acc_en = false,
      .packer_l1_acc = false};
}

inline ::ttnn::MemoryConfig kInterleavedDram() {
  return ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                              ::ttnn::BufferType::DRAM, std::nullopt};
}

// Runs the prep+capture form of ttnn::conv2d, stores the prepared weight + bias
// in g_prepared_convs[conv_idx], and returns the activation output. Used by
// both cached_conv2d_legacy and cached_conv2d on the first forward() pass.
ttnn::Tensor prep_and_capture_conv2d(
    int conv_idx,
    const ttnn::Tensor& input, const ttnn::Tensor& raw_weight,
    const std::optional<ttnn::Tensor>& raw_bias,
    ttnn::distributed::MeshDevice* device,
    uint32_t in_ch, uint32_t out_ch, uint32_t batch,
    uint32_t in_h, uint32_t in_w,
    std::array<uint32_t, 2> kernel, std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding, std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<ttnn::operations::unary::UnaryWithParam> activation) {
  auto cfg = kConvConfig();
  cfg.activation = activation;
  auto result = ttnn::conv2d(input, raw_weight, device,
      in_ch, out_ch, batch, in_h, in_w,
      kernel, stride, padding, dilation, groups,
      ::ttnn::DataType::BFLOAT16, raw_bias,
      cfg, kComputeConfig(), kInterleavedDram(),
      std::nullopt, false, true);
  auto& tup = std::get<2>(result);
  g_prepared_convs[conv_idx].weight = std::get<0>(std::get<1>(tup));
  g_prepared_convs[conv_idx].bias   = std::get<1>(std::get<1>(tup));
  return std::get<0>(tup);
}

}  // namespace

// ---------------------------------------------------------------------------
// cached_conv2d_legacy / cached_conv2d
//
// Single entry point per conv site: on the first call (g_convs_prepared=false)
// runs the prep path and stores the prepared weight/bias; on subsequent calls
// dispatches to the cached path. The "legacy" variant goes through the regular
// ttnn::conv2d (used for the first 5 convs); the default variant goes through
// the prim-cached fast path.
// ---------------------------------------------------------------------------
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
    std::optional<ttnn::operations::unary::UnaryWithParam> activation) {
  TRACY_ZONE("conv2d_legacy");
  if (!g_convs_prepared) {
    return prep_and_capture_conv2d(conv_idx, input, raw_weight, raw_bias, device,
        in_ch, out_ch, batch, in_h, in_w, kernel, stride, padding, dilation, groups,
        activation);
  }
  auto cfg = kConvConfig();
  cfg.activation = activation;
  return ::std::get<0>(ttnn::conv2d(input, g_prepared_convs[conv_idx].weight, device,
      in_ch, out_ch, batch, in_h, in_w,
      kernel, stride, padding, dilation, groups,
      ::ttnn::DataType::BFLOAT16, g_prepared_convs[conv_idx].bias,
      cfg, kComputeConfig(), kInterleavedDram(),
      std::nullopt));
}

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
    std::optional<ttnn::operations::unary::UnaryWithParam> activation) {
  TRACY_ZONE("conv2d");
  if (!g_convs_prepared) {
    return prep_and_capture_conv2d(conv_idx, input, raw_weight, raw_bias, device,
        in_ch, out_ch, batch, in_h, in_w, kernel, stride, padding, dilation, groups,
        activation);
  }
  return conv2d_prim_cached(input,
      g_prepared_convs[conv_idx].weight, g_prepared_convs[conv_idx].bias,
      device, in_ch, out_ch, batch, in_h, in_w,
      kernel, stride, padding, dilation, groups,
      kComputeConfig(), conv_idx, activation);
}

// ---------------------------------------------------------------------------
// fold_bn_scale_into_weight
//
// Folds a per-output-channel BatchNorm scale vector into a conv weight tensor
// in-place, so the conv at runtime can skip the multiply.
// ---------------------------------------------------------------------------
void fold_bn_scale_into_weight(
    ttnn::Tensor& weight,
    const ttnn::Tensor& bn_scale,
    uint32_t c_out,
    ttnn::distributed::MeshDevice* device) {
  auto dram = ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ttnn::BufferType::DRAM, std::nullopt};
  auto w_dev = ttnn::to_device(weight, device, dram);
  auto w_tile = ttnn::to_layout(w_dev, ttnn::Layout::TILE, std::nullopt, dram);
  ttnn::deallocate(w_dev, false);
  auto s_reshaped = ttnn::reshape(bn_scale,
      std::vector<int32_t>{(int32_t)c_out, 1, 1, 1}, dram);
  auto w_folded = ttnn::multiply(w_tile, s_reshaped,
      ttnn::DataType::BFLOAT16, dram);
  ttnn::deallocate(w_tile, false);
  ttnn::deallocate(s_reshaped, false);
  auto w_rm = ttnn::to_layout(w_folded, ttnn::Layout::ROW_MAJOR,
      std::nullopt, dram);
  ttnn::deallocate(w_folded, false);
  weight = ttnn::from_device(w_rm);
  ttnn::deallocate(w_rm, false);
}

// ---------------------------------------------------------------------------
// conv2d_prim_cached
//
// Wraps ttnn::prim::conv2d with a per-conv-site cache of sharding / halo /
// block configs. First call resolves and stores them; subsequent calls reuse.
// Falls back to ttnn::conv2d if the prim path throws.
// ---------------------------------------------------------------------------
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
    std::optional<ttnn::operations::unary::UnaryWithParam> activation)
{
    using namespace ttnn::operations::conv;
    using namespace ttnn::operations::sliding_window;
    auto& cache = g_conv_cache[conv_idx];

    if (cache.use_fallback) {
        ttnn::Conv2dConfig fb_cfg{.config_tensors_in_dram = true,
                                  .enable_kernel_stride_folding = false,
                                  .enable_act_double_buffer = true,
                                  .enable_weights_double_buffer = true};
        fb_cfg.activation = activation;
        return ::std::get<0>(ttnn::conv2d(input, weight, device,
            in_channels, out_channels, batch_size, input_height, input_width,
            kernel_size, stride, padding, dilation, groups,
            ::ttnn::DataType::BFLOAT16, bias,
            fb_cfg,
            compute_config,
            ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt},
            ::std::nullopt));
    }

    ttnn::Conv2dConfig conv_config;
    conv_config.config_tensors_in_dram = true;
    conv_config.enable_kernel_stride_folding = false;
    conv_config.weights_dtype = weight.dtype();
    conv_config.enable_act_double_buffer = true;
    conv_config.enable_weights_double_buffer = true;
    conv_config.activation = activation;
    auto padding_n4 = get_pair_n4_padding(padding);
    bool is_mm_conv = (groups == 1);

    if (!cache.valid) {
      try {
        auto [oh, ow] = calculate_output_image_size(
            {input_height, input_width}, kernel_size, stride, padding_n4, dilation);
        cache.output_height = oh; cache.output_width = ow;
        cache.input_tensor_shape = {batch_size, input_height, input_width, in_channels};
        auto compute_grid = device->compute_with_storage_grid_size();
        auto folded = fold_input_tensor_if_required(
            input, device, batch_size, input_height, input_width,
            in_channels, kernel_size, stride, dilation, padding_n4, is_mm_conv, conv_config);
        if (!folded.is_sharded()) {
            conv_config = determine_conv_config_for_auto_shard(
                conv_config, is_mm_conv, batch_size, in_channels, out_channels, oh, ow,
                kernel_size[0]*kernel_size[1]*in_channels/groups,
                input_height, input_width, compute_grid, folded.layout(), folded.dtype(),
                ::ttnn::DataType::BFLOAT16, folded.memory_config(),
                kernel_size, stride, dilation, padding_n4, groups, bias.has_value(), compute_config);
        }
        bool auto_shard = !conv_config.shard_layout.has_value();
        auto [sharded, in_par, out_par] = shard_or_reshard_tensor_if_required(
            device, folded, conv_config, batch_size, input_height, input_width,
            in_channels, out_channels, is_mm_conv, auto_shard);
        uint32_t in_ch_align = get_input_channels_alignment(
            sharded.memory_config().memory_layout(), sharded.layout(), false, is_mm_conv, sharded.memory_config());
        uint32_t in_ch_padded = tt::round_up(in_channels, in_ch_align);
        bool is_1d_dw = (groups==out_channels && groups==in_channels && kernel_size[1]==1);
        auto [par_cfg, blk_cfg, out_mem] = get_conv_configs(
            conv_config, compute_config, in_par, out_par, in_ch_padded, out_channels,
            batch_size, oh, ow, kernel_size, compute_grid, is_1d_dw);
        SlidingWindowConfig sw; sw.batch_size=batch_size; sw.input_hw={input_height,input_width};
        sw.window_hw={kernel_size[0],kernel_size[1]}; sw.stride_hw={stride[0],stride[1]};
        sw.padding={padding_n4[0],padding_n4[1],padding_n4[2],padding_n4[3]};
        sw.dilation_hw={dilation[0],dilation[1]}; sw.num_cores_nhw=par_cfg.num_cores_nhw;
        sw.core_range_set=sharded.memory_config().shard_spec().value().grid; sw.snap_to_tile=true;
        cache.parallel_config=par_cfg; cache.block_config=blk_cfg;
        cache.conv_out_memory_config=out_mem; cache.sliding_window_config=sw;
        cache.resolved_conv_config=conv_config; cache.valid=true;
        bool needs_halo = (padding_n4[0]>0||padding_n4[1]>0||padding_n4[2]>0||padding_n4[3]>0||
                           !sharded.is_sharded()||sharded.layout()!=ttnn::Layout::ROW_MAJOR);
        ttnn::Tensor hi = sharded;
        if (needs_halo) hi = ttnn::halo(sharded, sw, 0, false,
            in_par.shard_orientation==ttnn::ShardOrientation::COL_MAJOR, true, conv_config.config_tensors_in_dram);
        auto co = ttnn::prim::conv2d(hi, weight, bias, sw, out_channels, groups, false,
            conv_config.activation, par_cfg, blk_cfg, out_mem, ::ttnn::DataType::BFLOAT16,
            cache.input_tensor_shape, compute_config, conv_config.enable_act_double_buffer,
            conv_config.enable_weights_double_buffer, conv_config.full_inner_dim,
            conv_config.enable_activation_reuse, conv_config.config_tensors_in_dram, conv_config.force_split_reader);
        return ttnn::to_memory_config(co, tt::tt_metal::MemoryConfig{
            tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
      } catch (const std::exception&) {
        // Expected for conv shapes whose resolved prim-path L1 config doesn't
        // fit (TT_THROW from validate_circular_buffer_region). The fallback
        // ttnn::conv2d re-resolves with auto-sharding; cache.use_fallback then
        // pins this index to the fallback path on subsequent calls.
        cache.use_fallback=true; cache.valid=true;
        ttnn::Conv2dConfig fb_cfg{.config_tensors_in_dram = true, .enable_kernel_stride_folding = false,
                                  .enable_act_double_buffer = true, .enable_weights_double_buffer = true};
        fb_cfg.activation = activation;
        return ::std::get<0>(ttnn::conv2d(input, weight, device,
            in_channels, out_channels, batch_size, input_height, input_width,
            kernel_size, stride, padding, dilation, groups,
            ::ttnn::DataType::BFLOAT16, bias,
            fb_cfg,
            compute_config, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt}, ::std::nullopt));
      }
    }
    auto& cc = cache.resolved_conv_config;
    uint32_t bh=batch_size,ih=input_height,iw=input_width,ic=in_channels;
    auto ks=kernel_size;auto st=stride;auto dl=dilation;auto pd=padding_n4;bool mm=is_mm_conv;
    auto folded = fold_input_tensor_if_required(input,device,bh,ih,iw,ic,ks,st,dl,pd,mm,cc);
    bool as = !cc.shard_layout.has_value();
    auto [sharded,in_par,out_par] = shard_or_reshard_tensor_if_required(
        device,folded,cc,batch_size,input_height,input_width,in_channels,out_channels,is_mm_conv,as);
    bool needs_halo = (padding[0]>0||padding[1]>0||padding[2]>0||padding[3]>0||
                       !sharded.is_sharded()||sharded.layout()!=ttnn::Layout::ROW_MAJOR);
    ttnn::Tensor hi = sharded;
    if (needs_halo) {
        auto sw=cache.sliding_window_config;
        sw.core_range_set=sharded.memory_config().shard_spec().value().grid;
        hi = ttnn::halo(sharded, sw, 0, false,
            in_par.shard_orientation==ttnn::ShardOrientation::COL_MAJOR, true, cc.config_tensors_in_dram);
    }
    auto sw=cache.sliding_window_config;
    sw.core_range_set=sharded.memory_config().shard_spec().value().grid;
    auto co = ttnn::prim::conv2d(hi, weight, bias, sw, out_channels, groups, false, cc.activation,
        cache.parallel_config, cache.block_config, cache.conv_out_memory_config, ::ttnn::DataType::BFLOAT16,
        cache.input_tensor_shape, compute_config, cc.enable_act_double_buffer, cc.enable_weights_double_buffer,
        cc.full_inner_dim, cc.enable_activation_reuse, cc.config_tensors_in_dram, cc.force_split_reader);
    return ttnn::to_memory_config(co, tt::tt_metal::MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM});
}
