#include "common.hpp"

VoVNet::VoVNet(ttnn::distributed::MeshDevice* device,
               std::vector<ttnn::Tensor> raw_weights)
    : device_(device), inputs_(std::move(raw_weights)) {
  // Input-independent weight prep runs eagerly here, so forward() carries only
  // the inference path. Conv weight prep is input-shape-dependent and so is
  // primed lazily on the first operator() call (gated by g_convs_prepared).
  prepare_constevals(inputs_);
  prepare_bn_folds(inputs_, device_);
  g_convs_prepared = false;

  // Pin known-bad prim-path conv indices to the fallback path. These indices
  // throw `validate_circular_buffer_region` (L1 clash) when the prim auto-
  // resolved config is tried; conv2d_prim_cached's try/catch then flips
  // use_fallback=true and re-resolves with auto-sharding via ttnn::conv2d.
  // Pre-flipping avoids the throw-and-recover on warmup (saves init time and
  // suppresses the loud TT_THROW backtraces the device prints to stderr).
  // List sourced from the `PRIM FALLBACK idx=N` log lines observed in earlier
  // runs at this batch size; revisit if conv shapes / batch change.
  for (int idx : {6, 8, 10, 12, 21, 30, 39}) {
    g_conv_cache[idx].use_fallback = true;
    g_conv_cache[idx].valid = true;
  }
}

ttnn::Tensor VoVNet::operator()(const ttnn::Tensor& image) {
  inputs_[0] = image;
  auto outputs = forward(inputs_);
  if (!convs_prepared_) {
    g_convs_prepared = true;
    convs_prepared_ = true;
  }
  return outputs[0];
}

VoVNet::~VoVNet() {
  if (trace_id_) {
    ttnn::operations::trace::release_trace(device_, *trace_id_);
  }
}

void VoVNet::capture_trace() {
  // Capture must happen after warmup: program cache + conv shard/halo cache
  // populated, BN folds done, no further first-time-only branches in forward().
  auto tid = ttnn::operations::trace::begin_trace_capture(
      device_, ttnn::QueueId(0));
  try {
    auto outputs = forward(inputs_);
    ttnn::operations::trace::end_trace_capture(
        device_, tid, ttnn::QueueId(0));
    trace_id_ = tid;
    traced_output_ = std::move(outputs[0]);
  } catch (...) {
    ttnn::operations::trace::end_trace_capture(
        device_, tid, ttnn::QueueId(0));
    ttnn::operations::trace::release_trace(device_, tid);
    throw;
  }
}

ttnn::Tensor VoVNet::replay() {
  ttnn::operations::trace::execute_trace(
      device_, *trace_id_, ttnn::QueueId(0), /*blocking=*/false);
  return *traced_output_;
}

// Runs all 58 const-eval permutes/reshapes once into g_const_eval_cache. Each
// (slot, fn, weight_index) triple comes from the codegen call sites in forward().
void prepare_constevals(const std::vector<ttnn::Tensor>& inputs) {
  cached_const_eval(0,  &const_eval_permute,         inputs[1]);
  cached_const_eval(1,  &const_eval_permute,         inputs[23]);
  cached_const_eval(2,  &const_eval_permute,         inputs[27]);
  cached_const_eval(3,  &const_eval_permute,         inputs[44]);
  cached_const_eval(4,  &const_eval_permute,         inputs[19]);
  cached_const_eval(5,  &const_eval_reshape_scalar,  inputs[30]);
  cached_const_eval(6,  &const_eval_permute,         inputs[5]);
  cached_const_eval(7,  &const_eval_permute,         inputs[52]);
  cached_const_eval(8,  &const_eval_permute,         inputs[38]);
  cached_const_eval(9,  &const_eval_permute,         inputs[37]);
  cached_const_eval(10, &const_eval_permute,         inputs[12]);
  cached_const_eval(11, &const_eval_permute,         inputs[13]);
  cached_const_eval(12, &const_eval_passthrough,     inputs[89]);
  cached_const_eval(13, &const_eval_permute,         inputs[9]);
  cached_const_eval(14, &const_eval_permute,         inputs[16]);
  cached_const_eval(15, &const_eval_permute,         inputs[34]);
  cached_const_eval(16, &const_eval_permute,         inputs[49]);
  cached_const_eval(17, &const_eval_permute,         inputs[4]);
  cached_const_eval(18, &const_eval_permute,         inputs[22]);
  cached_const_eval(19, &const_eval_reshape_scalar,  inputs[54]);
  cached_const_eval(20, &const_eval_permute,         inputs[47]);
  cached_const_eval(21, &const_eval_permute,         inputs[10]);
  cached_const_eval(22, &const_eval_permute,         inputs[24]);
  cached_const_eval(23, &const_eval_passthrough,     inputs[69]);
  cached_const_eval(24, &const_eval_reshape_scalar,  inputs[17]);
  cached_const_eval(25, &const_eval_permute,         inputs[35]);
  cached_const_eval(26, &const_eval_permute,         inputs[36]);
  cached_const_eval(27, &const_eval_reshape_scalar,  inputs[41]);
  cached_const_eval(28, &const_eval_permute,         inputs[11]);
  cached_const_eval(29, &const_eval_reshape_scalar,  inputs[29]);
  cached_const_eval(30, &const_eval_permute,         inputs[48]);
  cached_const_eval(31, &const_eval_reshape_scalar,  inputs[53]);
  cached_const_eval(32, &const_eval_permute,         inputs[3]);
  cached_const_eval(33, &const_eval_permute,         inputs[50]);
  cached_const_eval(34, &const_eval_permute,         inputs[7]);
  cached_const_eval(35, &const_eval_permute,         inputs[21]);
  cached_const_eval(36, &const_eval_passthrough,     inputs[79]);
  cached_const_eval(37, &const_eval_reshape_scalar,  inputs[42]);
  cached_const_eval(38, &const_eval_permute,         inputs[40]);
  cached_const_eval(39, &const_eval_permute,         inputs[43]);
  cached_const_eval(40, &const_eval_reshape_scalar,  inputs[18]);
  cached_const_eval(41, &const_eval_permute,         inputs[46]);
  cached_const_eval(42, &const_eval_passthrough,     inputs[99]);
  cached_const_eval(43, &const_eval_permute,         inputs[6]);
  cached_const_eval(44, &const_eval_permute,         inputs[25]);
  cached_const_eval(45, &const_eval_permute,         inputs[51]);
  cached_const_eval(46, &const_eval_permute,         inputs[28]);
  cached_const_eval(47, &const_eval_permute,         inputs[2]);
  cached_const_eval(48, &const_eval_permute,         inputs[26]);
  cached_const_eval(49, &const_eval_permute,         inputs[31]);
  cached_const_eval(50, &const_eval_permute,         inputs[45]);
  cached_const_eval(51, &const_eval_permute,         inputs[33]);
  cached_const_eval(52, &const_eval_permute,         inputs[39]);
  cached_const_eval(53, &const_eval_permute,         inputs[14]);
  cached_const_eval(54, &const_eval_permute,         inputs[15]);
  cached_const_eval(55, &const_eval_permute,         inputs[8]);
  cached_const_eval(56, &const_eval_permute,         inputs[20]);
  cached_const_eval(57, &const_eval_permute,         inputs[32]);
}

// Folds each BN scale tensor into its conv weight in place. Must run after
// prepare_constevals (the scale tensors live in g_const_eval_cache).
void prepare_bn_folds(std::vector<ttnn::Tensor>& inputs,
                      ttnn::distributed::MeshDevice* device) {
  auto scale = [](int slot) -> const ttnn::Tensor& {
    return g_const_eval_cache[slot][0];
  };
  fold_bn_scale_into_weight(inputs[55], scale(0),  64,   device);
  fold_bn_scale_into_weight(inputs[57], scale(32), 64,   device);
  fold_bn_scale_into_weight(inputs[59], scale(6),  64,   device);
  fold_bn_scale_into_weight(inputs[60], scale(34), 128,  device);
  fold_bn_scale_into_weight(inputs[62], scale(13), 128,  device);
  fold_bn_scale_into_weight(inputs[64], scale(28), 128,  device);
  fold_bn_scale_into_weight(inputs[66], scale(11), 128,  device);
  fold_bn_scale_into_weight(inputs[67], scale(54), 256,  device);
  fold_bn_scale_into_weight(inputs[70], scale(4),  160,  device);
  fold_bn_scale_into_weight(inputs[72], scale(35), 160,  device);
  fold_bn_scale_into_weight(inputs[74], scale(1),  160,  device);
  fold_bn_scale_into_weight(inputs[76], scale(44), 160,  device);
  fold_bn_scale_into_weight(inputs[77], scale(2),  512,  device);
  fold_bn_scale_into_weight(inputs[80], scale(49), 192,  device);
  fold_bn_scale_into_weight(inputs[82], scale(51), 192,  device);
  fold_bn_scale_into_weight(inputs[84], scale(25), 192,  device);
  fold_bn_scale_into_weight(inputs[86], scale(9),  192,  device);
  fold_bn_scale_into_weight(inputs[87], scale(52), 768,  device);
  fold_bn_scale_into_weight(inputs[90], scale(39), 224,  device);
  fold_bn_scale_into_weight(inputs[92], scale(50), 224,  device);
  fold_bn_scale_into_weight(inputs[94], scale(20), 224,  device);
  fold_bn_scale_into_weight(inputs[96], scale(16), 224,  device);
  fold_bn_scale_into_weight(inputs[97], scale(45), 1024, device);
}

::std::vector<::ttnn::Tensor> forward(::std::vector<::ttnn::Tensor> v1) {
  TRACY_ZONE("forward");
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ::ttnn::Tensor v4 = v1[2];
  ::ttnn::Tensor v5 = v1[3];
  ::ttnn::Tensor v6 = v1[4];
  ::ttnn::Tensor v7 = v1[5];
  ::ttnn::Tensor v8 = v1[6];
  ::ttnn::Tensor v9 = v1[7];
  ::ttnn::Tensor v10 = v1[8];
  ::ttnn::Tensor v11 = v1[9];
  ::ttnn::Tensor v12 = v1[10];
  ::ttnn::Tensor v13 = v1[11];
  ::ttnn::Tensor v14 = v1[12];
  ::ttnn::Tensor v15 = v1[13];
  ::ttnn::Tensor v16 = v1[14];
  ::ttnn::Tensor v17 = v1[15];
  ::ttnn::Tensor v18 = v1[16];
  ::ttnn::Tensor v19 = v1[17];
  ::ttnn::Tensor v20 = v1[18];
  ::ttnn::Tensor v21 = v1[19];
  ::ttnn::Tensor v22 = v1[20];
  ::ttnn::Tensor v23 = v1[21];
  ::ttnn::Tensor v24 = v1[22];
  ::ttnn::Tensor v25 = v1[23];
  ::ttnn::Tensor v26 = v1[24];
  ::ttnn::Tensor v27 = v1[25];
  ::ttnn::Tensor v28 = v1[26];
  ::ttnn::Tensor v29 = v1[27];
  ::ttnn::Tensor v30 = v1[28];
  ::ttnn::Tensor v31 = v1[29];
  ::ttnn::Tensor v32 = v1[30];
  ::ttnn::Tensor v33 = v1[31];
  ::ttnn::Tensor v34 = v1[32];
  ::ttnn::Tensor v35 = v1[33];
  ::ttnn::Tensor v36 = v1[34];
  ::ttnn::Tensor v37 = v1[35];
  ::ttnn::Tensor v38 = v1[36];
  ::ttnn::Tensor v39 = v1[37];
  ::ttnn::Tensor v40 = v1[38];
  ::ttnn::Tensor v41 = v1[39];
  ::ttnn::Tensor v42 = v1[40];
  ::ttnn::Tensor v43 = v1[41];
  ::ttnn::Tensor v44 = v1[42];
  ::ttnn::Tensor v45 = v1[43];
  ::ttnn::Tensor v46 = v1[44];
  ::ttnn::Tensor v47 = v1[45];
  ::ttnn::Tensor v48 = v1[46];
  ::ttnn::Tensor v49 = v1[47];
  ::ttnn::Tensor v50 = v1[48];
  ::ttnn::Tensor v51 = v1[49];
  ::ttnn::Tensor v52 = v1[50];
  ::ttnn::Tensor v53 = v1[51];
  ::ttnn::Tensor v54 = v1[52];
  ::ttnn::Tensor v55 = v1[53];
  ::ttnn::Tensor v56 = v1[54];
  ::ttnn::Tensor v57 = v1[55];
  ::ttnn::Tensor v58 = v1[56];
  ::ttnn::Tensor v59 = v1[57];
  ::ttnn::Tensor v60 = v1[58];
  ::ttnn::Tensor v61 = v1[59];
  ::ttnn::Tensor v62 = v1[60];
  ::ttnn::Tensor v63 = v1[61];
  ::ttnn::Tensor v64 = v1[62];
  ::ttnn::Tensor v65 = v1[63];
  ::ttnn::Tensor v66 = v1[64];
  ::ttnn::Tensor v67 = v1[65];
  ::ttnn::Tensor v68 = v1[66];
  ::ttnn::Tensor v69 = v1[67];
  ::ttnn::Tensor v70 = v1[68];
  ::ttnn::Tensor v71 = v1[69];
  ::ttnn::Tensor v72 = v1[70];
  ::ttnn::Tensor v73 = v1[71];
  ::ttnn::Tensor v74 = v1[72];
  ::ttnn::Tensor v75 = v1[73];
  ::ttnn::Tensor v76 = v1[74];
  ::ttnn::Tensor v77 = v1[75];
  ::ttnn::Tensor v78 = v1[76];
  ::ttnn::Tensor v79 = v1[77];
  ::ttnn::Tensor v80 = v1[78];
  ::ttnn::Tensor v81 = v1[79];
  ::ttnn::Tensor v82 = v1[80];
  ::ttnn::Tensor v83 = v1[81];
  ::ttnn::Tensor v84 = v1[82];
  ::ttnn::Tensor v85 = v1[83];
  ::ttnn::Tensor v86 = v1[84];
  ::ttnn::Tensor v87 = v1[85];
  ::ttnn::Tensor v88 = v1[86];
  ::ttnn::Tensor v89 = v1[87];
  ::ttnn::Tensor v90 = v1[88];
  ::ttnn::Tensor v91 = v1[89];
  ::ttnn::Tensor v92 = v1[90];
  ::ttnn::Tensor v93 = v1[91];
  ::ttnn::Tensor v94 = v1[92];
  ::ttnn::Tensor v95 = v1[93];
  ::ttnn::Tensor v96 = v1[94];
  ::ttnn::Tensor v97 = v1[95];
  ::ttnn::Tensor v98 = v1[96];
  ::ttnn::Tensor v99 = v1[97];
  ::ttnn::Tensor v100 = v1[98];
  ::ttnn::Tensor v101 = v1[99];
  ::ttnn::Tensor v102 = v1[100];
  ::ttnn::Tensor v103 = v1[101];
  // Const-eval results were computed in prepare_constevals (one-time, at VoVNet
  // construction). Just bind local handles to the cache slots so the rest of
  // forward() reads exactly as the codegen emitted.
  const ttnn::Tensor& v108 = g_const_eval_cache[0][0];
  const ttnn::Tensor& v113 = g_const_eval_cache[1][0];
  const ttnn::Tensor& v118 = g_const_eval_cache[2][0];
  const ttnn::Tensor& v123 = g_const_eval_cache[3][0];
  const ttnn::Tensor& v128 = g_const_eval_cache[4][0];
  const ttnn::Tensor& v133 = g_const_eval_cache[5][0];
  const ttnn::Tensor& v138 = g_const_eval_cache[6][0];
  const ttnn::Tensor& v143 = g_const_eval_cache[7][0];
  const ttnn::Tensor& v148 = g_const_eval_cache[8][0];
  const ttnn::Tensor& v153 = g_const_eval_cache[9][0];
  const ttnn::Tensor& v158 = g_const_eval_cache[10][0];
  const ttnn::Tensor& v163 = g_const_eval_cache[11][0];
  const ttnn::Tensor& v168 = g_const_eval_cache[12][0];
  const ttnn::Tensor& v173 = g_const_eval_cache[13][0];
  const ttnn::Tensor& v178 = g_const_eval_cache[14][0];
  const ttnn::Tensor& v183 = g_const_eval_cache[15][0];
  const ttnn::Tensor& v188 = g_const_eval_cache[16][0];
  const ttnn::Tensor& v193 = g_const_eval_cache[17][0];
  const ttnn::Tensor& v198 = g_const_eval_cache[18][0];
  const ttnn::Tensor& v203 = g_const_eval_cache[19][0];
  const ttnn::Tensor& v208 = g_const_eval_cache[20][0];
  const ttnn::Tensor& v213 = g_const_eval_cache[21][0];
  const ttnn::Tensor& v218 = g_const_eval_cache[22][0];
  const ttnn::Tensor& v223 = g_const_eval_cache[23][0];
  const ttnn::Tensor& v228 = g_const_eval_cache[24][0];
  const ttnn::Tensor& v233 = g_const_eval_cache[25][0];
  const ttnn::Tensor& v238 = g_const_eval_cache[26][0];
  const ttnn::Tensor& v243 = g_const_eval_cache[27][0];
  const ttnn::Tensor& v248 = g_const_eval_cache[28][0];
  const ttnn::Tensor& v253 = g_const_eval_cache[29][0];
  const ttnn::Tensor& v258 = g_const_eval_cache[30][0];
  const ttnn::Tensor& v263 = g_const_eval_cache[31][0];
  const ttnn::Tensor& v268 = g_const_eval_cache[32][0];
  const ttnn::Tensor& v273 = g_const_eval_cache[33][0];
  const ttnn::Tensor& v278 = g_const_eval_cache[34][0];
  const ttnn::Tensor& v283 = g_const_eval_cache[35][0];
  const ttnn::Tensor& v288 = g_const_eval_cache[36][0];
  const ttnn::Tensor& v293 = g_const_eval_cache[37][0];
  const ttnn::Tensor& v298 = g_const_eval_cache[38][0];
  const ttnn::Tensor& v303 = g_const_eval_cache[39][0];
  const ttnn::Tensor& v308 = g_const_eval_cache[40][0];
  const ttnn::Tensor& v313 = g_const_eval_cache[41][0];
  const ttnn::Tensor& v318 = g_const_eval_cache[42][0];
  const ttnn::Tensor& v323 = g_const_eval_cache[43][0];
  const ttnn::Tensor& v328 = g_const_eval_cache[44][0];
  const ttnn::Tensor& v333 = g_const_eval_cache[45][0];
  const ttnn::Tensor& v338 = g_const_eval_cache[46][0];
  const ttnn::Tensor& v343 = g_const_eval_cache[47][0];
  const ttnn::Tensor& v348 = g_const_eval_cache[48][0];
  const ttnn::Tensor& v353 = g_const_eval_cache[49][0];
  const ttnn::Tensor& v358 = g_const_eval_cache[50][0];
  const ttnn::Tensor& v363 = g_const_eval_cache[51][0];
  const ttnn::Tensor& v368 = g_const_eval_cache[52][0];
  const ttnn::Tensor& v373 = g_const_eval_cache[53][0];
  const ttnn::Tensor& v378 = g_const_eval_cache[54][0];
  const ttnn::Tensor& v383 = g_const_eval_cache[55][0];
  const ttnn::Tensor& v388 = g_const_eval_cache[56][0];
  const ttnn::Tensor& v393 = g_const_eval_cache[57][0];
  ttnn::distributed::MeshDevice *v394 = ttnn::DeviceGetter::getInstance();

  ::ttnn::Tensor v395 = ttnn::tilize_with_zero_padding(
      v2,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v396 = ttnn::permute(
      v395, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v395, false);
  ::ttnn::Tensor v397 = ttnn::reshape(
      v396, ::std::vector<int32_t>{1, 1, 802816, 3},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v396, false);
  ::ttnn::Tensor v398 = ttnn::untilize_with_unpadding(
      v397, ::ttnn::Shape({0, 0, 802815, 2}),
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v397, false);
  ::ttnn::Tensor v399 = cached_conv2d_legacy(0, v398, v57, v343, v394,
      3, 64, 16, 224, 224,
      {3, 3}, {2, 2}, {1, 1, 1, 1}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v398, false);
  ::ttnn::Tensor v402 = v399; // conv fused with BN bias + relu
  ::ttnn::Tensor v403 = ttnn::untilize(
      v402,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v402, false);
  ::ttnn::Tensor v404 = cached_conv2d_legacy(1, v403, v58, ::std::nullopt, v394,
      64, 64, 16, 112, 112,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 64);

  ttnn::deallocate(v403, false);
  ::ttnn::Tensor v405 = ttnn::untilize(
      v404,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v404, false);
  ::ttnn::Tensor v406 = cached_conv2d_legacy(2, v405, v59, v193, v394,
      64, 64, 16, 112, 112,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v405, false);
  ::ttnn::Tensor v409 = v406; // conv fused with BN bias + relu
  ::ttnn::Tensor v410 = ttnn::untilize(
      v409,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v409, false);
  ::ttnn::Tensor v411 = cached_conv2d_legacy(3, v410, v60, ::std::nullopt, v394,
      64, 64, 16, 112, 112,
      {3, 3}, {2, 2}, {1, 1, 1, 1}, {1, 1}, 64);

  ttnn::deallocate(v410, false);
  ::ttnn::Tensor v412 = ttnn::untilize(
      v411,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v411, false);
  ::ttnn::Tensor v413 = cached_conv2d_legacy(4, v412, v61, v323, v394,
      64, 64, 16, 56, 56,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v412, false);
  ::ttnn::Tensor v416 = v413; // conv fused with BN bias + relu
  ::ttnn::Tensor v417 = ttnn::untilize(
      v416,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v418 = cached_conv2d(5, v417, v62, v383, v394,
      64, 128, 16, 56, 56,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v417, false);
  ::ttnn::Tensor v421 = v418; // conv fused with BN bias + relu
  ::ttnn::Tensor v422 = ttnn::untilize(
      v421,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v421, false);
  ::ttnn::Tensor v423 = cached_conv2d(6, v422, v63, std::nullopt, v394,
      128, 128, 16, 56, 56,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 128);

  ttnn::deallocate(v422, false);
  ::ttnn::Tensor v424 = ttnn::untilize(
      v423,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v423, false);
  ::ttnn::Tensor v425 = cached_conv2d(7, v424, v64, v213, v394,
      128, 128, 16, 56, 56,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v424, false);
  ::ttnn::Tensor v428 = v425; // conv fused with BN bias + relu
  ::ttnn::Tensor v429 = ttnn::untilize(
      v428,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v430 = cached_conv2d(8, v429, v65, std::nullopt, v394,
      128, 128, 16, 56, 56,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 128);

  ttnn::deallocate(v429, false);
  ::ttnn::Tensor v431 = ttnn::untilize(
      v430,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v430, false);
  ::ttnn::Tensor v432 = cached_conv2d(9, v431, v66, v158, v394,
      128, 128, 16, 56, 56,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v431, false);
  ::ttnn::Tensor v435 = v432; // conv fused with BN bias + relu
  ::ttnn::Tensor v436 = ttnn::untilize(
      v435,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v437 = cached_conv2d(10, v436, v67, std::nullopt, v394,
      128, 128, 16, 56, 56,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 128);

  ttnn::deallocate(v436, false);
  ::ttnn::Tensor v438 = ttnn::untilize(
      v437,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v437, false);
  ::ttnn::Tensor v439 = cached_conv2d(11, v438, v68, v373, v394,
      128, 128, 16, 56, 56,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v438, false);
  ::ttnn::Tensor v442 = v439; // conv fused with BN bias + relu
  ::std::vector<::ttnn::Tensor> v443 = util_create_vec(v416, v428, v435, v442);
  ::ttnn::Tensor v444 = ttnn::concat(
      v443, 3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v442, false);
  ttnn::deallocate(v435, false);
  ttnn::deallocate(v428, false);
  ttnn::deallocate(v416, false);
  ::ttnn::Tensor v445 = ttnn::untilize(
      v444,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v444, false);
  ::ttnn::Tensor v446 = cached_conv2d(12, v445, v69, v178, v394,
      448, 256, 16, 56, 56,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v445, false);
  ::ttnn::Tensor v449 = v446; // conv fused with BN bias + relu
  ::ttnn::Tensor v450 = ttnn::reshape(
      v449, ::std::vector<int32_t>{16, 56, 56, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v451 = ttnn::mean(
      v450, ::ttsl::SmallVector<int32_t>{1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::LoFi, .math_approx_mode = true, .fp32_dest_acc_en = false, .packer_l1_acc = false});
  ttnn::deallocate(v450, false);
  ::ttnn::Tensor v452 = ttnn::mean(
      v451, ::ttsl::SmallVector<int32_t>{2}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::LoFi, .math_approx_mode = true, .fp32_dest_acc_en = false, .packer_l1_acc = false});
  ttnn::deallocate(v451, false);
  ::ttnn::Tensor v453 = ttnn::reshape(
      v452, ::std::vector<int32_t>{1, 1, 16, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v452, false);
  ::ttnn::Tensor v454 = ttnn::untilize(
      v453,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v453, false);
  ::ttnn::Tensor v455 = cached_conv2d(13, v454, v70, v223, v394,
      256, 256, 16, 1, 1,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1);

  ttnn::deallocate(v454, false);
  ::ttnn::Tensor v456 = ttnn::add(v455, v228, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v455, false);
  ::ttnn::Tensor v457 = ttnn::relu6(
      v456, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v456, false);
  ::ttnn::Tensor v458 = ttnn::divide(
      v457, v308, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v457, false);
  ::ttnn::Tensor v459 = ttnn::reshape(
      v458, ::std::vector<int32_t>{16, 1, 1, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v458, false);
  ::ttnn::Tensor v460 = ttnn::repeat(v459, ::ttnn::Shape({1, 56, 56, 1}));
  ttnn::deallocate(v459, false);
  ::ttnn::Tensor v461 = ttnn::reshape(
      v460, ::std::vector<int32_t>{1, 1, 50176, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v460, false);
  ::ttnn::Tensor v462 = ttnn::multiply(
      v449, v461, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v461, false);
  ttnn::deallocate(v449, false);
  ::ttnn::Tensor v463 = ttnn::untilize(
      v462,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v462, false);
  ::std::vector<::ttnn::Tensor> v464 = ttnn::max_pool2d(
      v463, 16, 56, 56, 256, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{0, 0},
      ::std::array<uint32_t, 2>{1, 1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt, ::std::nullopt, false, false, false,
      ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, true);
  ::ttnn::Tensor v465 = v464[0];
  ttnn::deallocate(v463, false);
  ::ttnn::Tensor v466 = ttnn::tilize_with_zero_padding(
      v465,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v467 = cached_conv2d(14, v465, v72, v388, v394,
      256, 160, 16, 28, 28,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v465, false);
  ::ttnn::Tensor v470 = v467; // conv fused with BN bias + relu
  ::ttnn::Tensor v471 = ttnn::untilize(
      v470,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v470, false);
  ::ttnn::Tensor v472 = cached_conv2d(15, v471, v73, std::nullopt, v394,
      160, 160, 16, 28, 28,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 160);

  ttnn::deallocate(v471, false);
  ::ttnn::Tensor v473 = ttnn::untilize(
      v472,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v472, false);
  ::ttnn::Tensor v474 = cached_conv2d(16, v473, v74, v198, v394,
      160, 160, 16, 28, 28,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v473, false);
  ::ttnn::Tensor v477 = v474; // conv fused with BN bias + relu
  ::ttnn::Tensor v478 = ttnn::untilize(
      v477,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v479 = cached_conv2d(17, v478, v75, std::nullopt, v394,
      160, 160, 16, 28, 28,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 160);

  ttnn::deallocate(v478, false);
  ::ttnn::Tensor v480 = ttnn::untilize(
      v479,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v479, false);
  ::ttnn::Tensor v481 = cached_conv2d(18, v480, v76, v218, v394,
      160, 160, 16, 28, 28,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v480, false);
  ::ttnn::Tensor v484 = v481; // conv fused with BN bias + relu
  ::ttnn::Tensor v485 = ttnn::untilize(
      v484,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v486 = cached_conv2d(19, v485, v77, std::nullopt, v394,
      160, 160, 16, 28, 28,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 160);

  ttnn::deallocate(v485, false);
  ::ttnn::Tensor v487 = ttnn::untilize(
      v486,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v486, false);
  ::ttnn::Tensor v488 = cached_conv2d(20, v487, v78, v348, v394,
      160, 160, 16, 28, 28,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v487, false);
  ::ttnn::Tensor v491 = v488; // conv fused with BN bias + relu
  // Flatten the 4D skip-connection tensor (max_pool→tilize output) to the same
  // [1, 1, N*H*W, C] layout as the conv outputs so concat's shape check passes.
  v466 = ttnn::reshape(v466,
      ::std::vector<int32_t>{1, 1, 16 * 28 * 28, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v492 = util_create_vec(v466, v477, v484, v491);
  ::ttnn::Tensor v493 = ttnn::concat(
      v492, 3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v491, false);
  ttnn::deallocate(v484, false);
  ttnn::deallocate(v477, false);
  ttnn::deallocate(v466, false);
  ::ttnn::Tensor v494 = ttnn::untilize(
      v493,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v493, false);
  ::ttnn::Tensor v495 = cached_conv2d(21, v494, v79, v338, v394,
      736, 512, 16, 28, 28,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v494, false);
  ::ttnn::Tensor v498 = v495; // conv fused with BN bias + relu
  ::ttnn::Tensor v499 = ttnn::reshape(
      v498, ::std::vector<int32_t>{16, 28, 28, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v500 = ttnn::mean(
      v499, ::ttsl::SmallVector<int32_t>{1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::LoFi, .math_approx_mode = true, .fp32_dest_acc_en = false, .packer_l1_acc = false});
  ttnn::deallocate(v499, false);
  ::ttnn::Tensor v501 = ttnn::mean(
      v500, ::ttsl::SmallVector<int32_t>{2}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::LoFi, .math_approx_mode = true, .fp32_dest_acc_en = false, .packer_l1_acc = false});
  ttnn::deallocate(v500, false);
  ::ttnn::Tensor v502 = ttnn::reshape(
      v501, ::std::vector<int32_t>{1, 1, 16, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v501, false);
  ::ttnn::Tensor v503 = ttnn::untilize(
      v502,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v502, false);
  ::ttnn::Tensor v504 = cached_conv2d(22, v503, v80, v288, v394,
      512, 512, 16, 1, 1,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1);

  ttnn::deallocate(v503, false);
  ::ttnn::Tensor v505 = ttnn::add(v504, v253, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v504, false);
  ::ttnn::Tensor v506 = ttnn::relu6(
      v505, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v505, false);
  ::ttnn::Tensor v507 = ttnn::divide(
      v506, v133, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v506, false);
  ::ttnn::Tensor v508 = ttnn::reshape(
      v507, ::std::vector<int32_t>{16, 1, 1, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v507, false);
  ::ttnn::Tensor v509 = ttnn::repeat(v508, ::ttnn::Shape({1, 28, 28, 1}));
  ttnn::deallocate(v508, false);
  ::ttnn::Tensor v510 = ttnn::reshape(
      v509, ::std::vector<int32_t>{1, 1, 12544, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v509, false);
  ::ttnn::Tensor v511 = ttnn::multiply(
      v498, v510, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v510, false);
  ttnn::deallocate(v498, false);
  ::ttnn::Tensor v512 = ttnn::untilize(
      v511,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v511, false);
  ::std::vector<::ttnn::Tensor> v513 = ttnn::max_pool2d(
      v512, 16, 28, 28, 512, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{0, 0},
      ::std::array<uint32_t, 2>{1, 1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt, ::std::nullopt, false, false, false,
      ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, true);
  ::ttnn::Tensor v514 = v513[0];
  ttnn::deallocate(v512, false);
  ::ttnn::Tensor v515 = ttnn::tilize_with_zero_padding(
      v514,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v516 = cached_conv2d(23, v514, v82, v393, v394,
      512, 192, 16, 14, 14,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v514, false);
  ::ttnn::Tensor v519 = v516; // conv fused with BN bias + relu
  ::ttnn::Tensor v520 = ttnn::untilize(
      v519,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v519, false);
  ::ttnn::Tensor v521 = cached_conv2d(24, v520, v83, std::nullopt, v394,
      192, 192, 16, 14, 14,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 192);

  ttnn::deallocate(v520, false);
  ::ttnn::Tensor v522 = ttnn::untilize(
      v521,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v521, false);
  ::ttnn::Tensor v523 = cached_conv2d(25, v522, v84, v183, v394,
      192, 192, 16, 14, 14,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v522, false);
  ::ttnn::Tensor v526 = v523; // conv fused with BN bias + relu
  ::ttnn::Tensor v527 = ttnn::untilize(
      v526,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v528 = cached_conv2d(26, v527, v85, std::nullopt, v394,
      192, 192, 16, 14, 14,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 192);

  ttnn::deallocate(v527, false);
  ::ttnn::Tensor v529 = ttnn::untilize(
      v528,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v528, false);
  ::ttnn::Tensor v530 = cached_conv2d(27, v529, v86, v238, v394,
      192, 192, 16, 14, 14,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v529, false);
  ::ttnn::Tensor v533 = v530; // conv fused with BN bias + relu
  ::ttnn::Tensor v534 = ttnn::untilize(
      v533,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v535 = cached_conv2d(28, v534, v87, std::nullopt, v394,
      192, 192, 16, 14, 14,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 192);

  ttnn::deallocate(v534, false);
  ::ttnn::Tensor v536 = ttnn::untilize(
      v535,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v535, false);
  ::ttnn::Tensor v537 = cached_conv2d(29, v536, v88, v148, v394,
      192, 192, 16, 14, 14,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v536, false);
  ::ttnn::Tensor v540 = v537; // conv fused with BN bias + relu
  v515 = ttnn::reshape(v515,
      ::std::vector<int32_t>{1, 1, 16 * 14 * 14, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v541 = util_create_vec(v515, v526, v533, v540);
  ::ttnn::Tensor v542 = ttnn::concat(
      v541, 3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v540, false);
  ttnn::deallocate(v533, false);
  ttnn::deallocate(v526, false);
  ttnn::deallocate(v515, false);
  ::ttnn::Tensor v543 = ttnn::untilize(
      v542,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v542, false);
  ::ttnn::Tensor v544 = cached_conv2d(30, v543, v89, v298, v394,
      1088, 768, 16, 14, 14,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v543, false);
  ::ttnn::Tensor v547 = v544; // conv fused with BN bias + relu
  ::ttnn::Tensor v548 = ttnn::reshape(
      v547, ::std::vector<int32_t>{16, 14, 14, 768},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v549 = ttnn::mean(
      v548, ::ttsl::SmallVector<int32_t>{1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::LoFi, .math_approx_mode = true, .fp32_dest_acc_en = false, .packer_l1_acc = false});
  ttnn::deallocate(v548, false);
  ::ttnn::Tensor v550 = ttnn::mean(
      v549, ::ttsl::SmallVector<int32_t>{2}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::LoFi, .math_approx_mode = true, .fp32_dest_acc_en = false, .packer_l1_acc = false});
  ttnn::deallocate(v549, false);
  ::ttnn::Tensor v551 = ttnn::reshape(
      v550, ::std::vector<int32_t>{1, 1, 16, 768},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v550, false);
  ::ttnn::Tensor v552 = ttnn::untilize(
      v551,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v551, false);
  ::ttnn::Tensor v553 = cached_conv2d(31, v552, v90, v168, v394,
      768, 768, 16, 1, 1,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1);

  ttnn::deallocate(v552, false);
  ::ttnn::Tensor v554 = ttnn::add(v553, v243, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v553, false);
  ::ttnn::Tensor v555 = ttnn::relu6(
      v554, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v554, false);
  ::ttnn::Tensor v556 = ttnn::divide(
      v555, v293, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v555, false);
  ::ttnn::Tensor v557 = ttnn::reshape(
      v556, ::std::vector<int32_t>{16, 1, 1, 768},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v556, false);
  ::ttnn::Tensor v558 = ttnn::repeat(v557, ::ttnn::Shape({1, 14, 14, 1}));
  ttnn::deallocate(v557, false);
  ::ttnn::Tensor v559 = ttnn::reshape(
      v558, ::std::vector<int32_t>{1, 1, 3136, 768},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v558, false);
  ::ttnn::Tensor v560 = ttnn::multiply(
      v547, v559, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v559, false);
  ttnn::deallocate(v547, false);
  ::ttnn::Tensor v561 = ttnn::untilize(
      v560,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v560, false);
  ::std::vector<::ttnn::Tensor> v562 = ttnn::max_pool2d(
      v561, 16, 14, 14, 768, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{0, 0},
      ::std::array<uint32_t, 2>{1, 1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt, ::std::nullopt, false, false, false,
      ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, true);
  ::ttnn::Tensor v563 = v562[0];
  ttnn::deallocate(v561, false);
  ::ttnn::Tensor v564 = ttnn::tilize_with_zero_padding(
      v563,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v565 = cached_conv2d(32, v563, v92, v123, v394,
      768, 224, 16, 7, 7,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v563, false);
  ::ttnn::Tensor v568 = v565; // conv fused with BN bias + relu
  ::ttnn::Tensor v569 = ttnn::untilize(
      v568,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v568, false);
  ::ttnn::Tensor v570 = cached_conv2d(33, v569, v93, std::nullopt, v394,
      224, 224, 16, 7, 7,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 224);

  ttnn::deallocate(v569, false);
  ::ttnn::Tensor v571 = ttnn::untilize(
      v570,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v570, false);
  ::ttnn::Tensor v572 = cached_conv2d(34, v571, v94, v313, v394,
      224, 224, 16, 7, 7,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v571, false);
  ::ttnn::Tensor v575 = v572; // conv fused with BN bias + relu
  ::ttnn::Tensor v576 = ttnn::untilize(
      v575,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v577 = cached_conv2d(35, v576, v95, std::nullopt, v394,
      224, 224, 16, 7, 7,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 224);

  ttnn::deallocate(v576, false);
  ::ttnn::Tensor v578 = ttnn::untilize(
      v577,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v577, false);
  ::ttnn::Tensor v579 = cached_conv2d(36, v578, v96, v258, v394,
      224, 224, 16, 7, 7,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v578, false);
  ::ttnn::Tensor v582 = v579; // conv fused with BN bias + relu
  ::ttnn::Tensor v583 = ttnn::untilize(
      v582,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v584 = cached_conv2d(37, v583, v97, std::nullopt, v394,
      224, 224, 16, 7, 7,
      {3, 3}, {1, 1}, {1, 1, 1, 1}, {1, 1}, 224);

  ttnn::deallocate(v583, false);
  ::ttnn::Tensor v585 = ttnn::untilize(
      v584,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v584, false);
  ::ttnn::Tensor v586 = cached_conv2d(38, v585, v98, v273, v394,
      224, 224, 16, 7, 7,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v585, false);
  ::ttnn::Tensor v589 = v586; // conv fused with BN bias + relu
  v564 = ttnn::reshape(v564,
      ::std::vector<int32_t>{1, 1, 16 * 7 * 7, 768},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v590 = util_create_vec(v564, v575, v582, v589);
  ::ttnn::Tensor v591 = ttnn::concat(
      v590, 3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v589, false);
  ttnn::deallocate(v582, false);
  ttnn::deallocate(v575, false);
  ttnn::deallocate(v564, false);
  ::ttnn::Tensor v592 = ttnn::untilize(
      v591,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v591, false);
  ::ttnn::Tensor v593 = cached_conv2d(39, v592, v99, v143, v394,
      1440, 1024, 16, 7, 7,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, ttnn::operations::unary::UnaryWithParam{ttnn::operations::unary::UnaryOpType::RELU});

  ttnn::deallocate(v592, false);
  ::ttnn::Tensor v596 = v593; // conv fused with BN bias + relu
  ::ttnn::Tensor v597 = ttnn::reshape(
      v596, ::std::vector<int32_t>{16, 7, 7, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v596, false);
  ::ttnn::Tensor v598 = ttnn::permute(
      v597, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v599 = ttnn::mean(
      v597, ::ttsl::SmallVector<int32_t>{1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::LoFi, .math_approx_mode = true, .fp32_dest_acc_en = false, .packer_l1_acc = false});
  ttnn::deallocate(v597, false);
  ::ttnn::Tensor v600 = ttnn::mean(
      v599, ::ttsl::SmallVector<int32_t>{2}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::LoFi, .math_approx_mode = true, .fp32_dest_acc_en = false, .packer_l1_acc = false});
  ttnn::deallocate(v599, false);
  ::ttnn::Tensor v601 = ttnn::reshape(
      v600, ::std::vector<int32_t>{1, 1, 16, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v600, false);
  ::ttnn::Tensor v602 = ttnn::untilize(
      v601,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v601, false);
  ::ttnn::Tensor v603 = cached_conv2d(40, v602, v100, v318, v394,
      1024, 1024, 16, 1, 1,
      {1, 1}, {1, 1}, {0, 0, 0, 0}, {1, 1}, 1);

  ttnn::deallocate(v602, false);
  ::ttnn::Tensor v604 = ttnn::add(v603, v263, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v603, false);
  ::ttnn::Tensor v605 = ttnn::relu6(
      v604, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v604, false);
  ::ttnn::Tensor v606 = ttnn::divide(
      v605, v203, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v605, false);
  ::ttnn::Tensor v607 = ttnn::reshape(
      v606, ::std::vector<int32_t>{16, 1, 1, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v606, false);
  ::ttnn::Tensor v608 = ttnn::permute(
      v607, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v607, false);
  ::ttnn::Tensor v609 = ttnn::reshape(
      v608, ::std::vector<int32_t>{16, 1, 1024, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v608, false);
  ::ttnn::Tensor v610 = ttnn::reshape(
      v598, ::std::vector<int32_t>{16, 1, 1024, 49},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v598, false);
  ::ttnn::Tensor v611 = ttnn::multiply(v610, v609, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v610, false);
  ttnn::deallocate(v609, false);
  ::ttnn::Tensor v612 = ttnn::mean(
      v611, ::ttsl::SmallVector<int32_t>{3}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::LoFi, .math_approx_mode = true, .fp32_dest_acc_en = false, .packer_l1_acc = false});
  ttnn::deallocate(v611, false);
  ::ttnn::Tensor v613 = ttnn::reshape(
      v612, ::std::vector<int32_t>{16, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v612, false);
  ::ttnn::Tensor v614 = ttnn::linear(
      v613, v102, v103, false, false,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::DataType::BFLOAT16, ::std::nullopt, ::std::nullopt,
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::LoFi, .math_approx_mode = true, .fp32_dest_acc_en = false, .packer_l1_acc = false});
  ttnn::deallocate(v613, false);
  ::std::vector<::ttnn::Tensor> v615 = util_create_vec(v614);
  return v615;
}
