// ttmlir-opt --ttcore-register-device --ttir-to-ttmetal-me-pipeline --ttir-to-ttmetal-be-pipeline test/ttmlir/Dialect/D2M/Transforms/im2col_conv_l1_copy_pre_cb.mlir
// ttmlir-opt --ttcore-register-device="system-desc-path=${SYSTEM_DESC_PATH}" --ttir-to-ttmetal-pipeline="system-desc-path=${SYSTEM_DESC_PATH} ttnn-mode=true use-tile-matmul=true" test/ttmlir/Dialect/D2M/Transforms/im2col_conv_l1_copy_pre_cb.mlir

// Post-bufferize, pre-ConvertLocalLoadStoreOpsToAliasedCBs im2col conv.
//
// Func args are TTNN tensors (height-sharded, L1).
// ttir.ttnn_metal_layout_cast bridges to metal memrefs for the D2M body.
// Output is untilized to row-major before writing back to the output tensor.
//
// Setup:
//   Image:   H=9, W=9, C=8 → activation shard: memref<81x8xbf16>
//   Kernel:  K_h=2, K_w=2, stride=1, no padding
//   OH=8, OW=8 → 64 output pixels
//   K = K_h*K_w*C = 2*2*8 = 32  (exactly one tile width)
//   C_out = 64
//
//   Blocking: d0 in 2 blocks of 32 output pixels
//   Per-block im2col: [32, 32] bf16 → exactly 1 tile per push
//
// Tile space:
//   Col per block:  [32, 32] bf16 → [1, 1] tile
//   Weights:        [32, 64] bf16 → [1, 2] tiles
//   Output/block:   [32, 64] bf16 (untilized from [1, 2] tiles)
//
// Per-block matmul:  [1,1] × [1,2] → [1,2] tiles  (2 tile matmuls, no reduction)

#l1_ = #ttcore.memory_space<l1>
#l1_ttnn = #ttnn.buffer_type<l1>

#shard_act = #ttcore.shard<16x2, 1>
#shard_wt  = #ttcore.shard<128x2, 1>
#shard_out = #ttcore.shard<128x2, 1>
#map_id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#ttnn_act = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<81x8xbf16, #l1_ttnn>, <height_sharded>, exactGrid = true>
#ttnn_wt  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64xbf16, #l1_ttnn>, <height_sharded>, exactGrid = true>
#ttnn_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64xbf16, #l1_ttnn>, <height_sharded>, exactGrid = true>

module {
  // CHECK-LABEL: func.func @im2col_conv
  func.func @im2col_conv(
      %arg0 : tensor<81x8xbf16, #ttnn_act>,
      %arg1 : tensor<32x64xbf16, #ttnn_wt>,
      %arg2 : tensor<32x64xbf16, #ttnn_out>) -> tensor<32x64xbf16, #ttnn_out> {

    %act_phys = ttir.ttnn_metal_layout_cast %arg0
        : tensor<81x8xbf16, #ttnn_act>
        -> memref<1x1x81x8xbf16, #shard_act, #l1_>
    %wt_phys = ttir.ttnn_metal_layout_cast %arg1
        : tensor<32x64xbf16, #ttnn_wt>
        -> memref<1x1x32x64xbf16, #shard_wt, #l1_>
    %out_phys = ttir.ttnn_metal_layout_cast %arg2
        : tensor<32x64xbf16, #ttnn_out>
        -> memref<1x1x32x64xbf16, #shard_out, #l1_>

    %act_local = memref.alloc() {alignment = 16 : i64}
        : memref<1x1x81x8xbf16, #shard_act, #l1_>
    %wt_local = memref.alloc() {alignment = 16 : i64}
        : memref<1x1x32x64xbf16, #shard_wt, #l1_>
    %out_local = memref.alloc() {alignment = 16 : i64}
        : memref<1x1x32x64xbf16, #shard_out, #l1_>

    %act_stream = "d2m.stream_layout"(%act_phys, %act_local) <{remapping = #map_id}>
        : (memref<1x1x81x8xbf16, #shard_act, #l1_>,
           memref<1x1x81x8xbf16, #shard_act, #l1_>)
        -> memref<1x1x81x8xbf16, #shard_act, #ttcore.view<4>, #l1_>
    %wt_stream = "d2m.stream_layout"(%wt_phys, %wt_local) <{remapping = #map_id}>
        : (memref<1x1x32x64xbf16, #shard_wt, #l1_>,
           memref<1x1x32x64xbf16, #shard_wt, #l1_>)
        -> memref<1x1x32x64xbf16, #shard_wt, #ttcore.view<4>, #l1_>
    %out_stream = "d2m.stream_layout"(%out_phys, %out_local) <{remapping = #map_id}>
        : (memref<1x1x32x64xbf16, #shard_out, #l1_>,
           memref<1x1x32x64xbf16, #shard_out, #l1_>)
        -> memref<1x1x32x64xbf16, #shard_out, #ttcore.view<4>, #l1_>

    d2m.generic {
        block_factors = [], grid = #ttcore.grid<1x1>,
        indexing_maps = [], iterator_types = [],
        threads = [#d2m.thread<unified>]
    }
    ins(%act_stream, %wt_stream :
        memref<1x1x81x8xbf16, #shard_act, #ttcore.view<4>, #l1_>,
        memref<1x1x32x64xbf16, #shard_wt, #ttcore.view<4>, #l1_>)
    outs(%out_stream :
        memref<1x1x32x64xbf16, #shard_out, #ttcore.view<4>, #l1_>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %gy = d2m.core_index(0) : index
      %gx = d2m.core_index(1) : index

      // --- DM: load activation shard into local buffer ---
      %act_alloc = memref.alloc() {alignment = 64 : i64} : memref<81x8xbf16>
      %act = d2m.remote_load %act_alloc %act_stream[%gy, %gx]
          : memref<81x8xbf16>,
            memref<1x1x81x8xbf16, #shard_act, #ttcore.view<4>, #l1_>
          -> memref<81x8xbf16, #l1_>

      // --- DM: load weight shard into local buffer ---
      %wt_alloc = memref.alloc() {alignment = 64 : i64} : memref<32x64xbf16>
      %wt = d2m.remote_load %wt_alloc %wt_stream[%gy, %gx]
          : memref<32x64xbf16>,
            memref<1x1x32x64xbf16, #shard_wt, #ttcore.view<4>, #l1_>
          -> memref<32x64xbf16, #l1_>

      // --- Compute: tilize weights (done once, reused across blocks) ---
      %wt_tiled = memref.alloc() {alignment = 64 : i64}
          : memref<1x2x!ttcore.tile<32x32, bf16>>
      "d2m.tile_tilize_block"(%wt, %wt_tiled)
          : (memref<32x64xbf16, #l1_>,
             memref<1x2x!ttcore.tile<32x32, bf16>>)
          -> memref<1x2x!ttcore.tile<32x32, bf16>>

      // --- Alloc per-block intermediates (reused across iterations) ---
      %col_rm = memref.alloc() {alignment = 64 : i64} : memref<32x32xbf16>
      %col_tiled = memref.alloc() {alignment = 64 : i64}
          : memref<1x1x!ttcore.tile<32x32, bf16>>
      %out_tiled = memref.alloc() {alignment = 64 : i64}
          : memref<1x2x!ttcore.tile<32x32, bf16>>
      %out_rm = memref.alloc() {alignment = 64 : i64} : memref<32x64xbf16>

      // --- Blocking loop: 2 blocks of 32 output pixels ---
      scf.for %block = %c0 to %c2 step %c1 {
        // --- DM: im2col gather ONE tile (32×32 scalars) ---
        d2m.l1_copy %act, %col_rm
            indexing_maps = [
              affine_map<(d0, d1) -> (
                (d0 floordiv 8 + d1 floordiv 16) * 9
                + d0 mod 8 + (d1 mod 16) floordiv 8,
                d1 mod 8)>,
              affine_map<(d0, d1) -> (d0, d1)>
            ]
            : memref<81x8xbf16, #l1_>, memref<32x32xbf16>

        // --- Compute: tilize one tile (32×32 → 1×1 tile) ---
        "d2m.tile_tilize_block"(%col_rm, %col_tiled)
            : (memref<32x32xbf16>,
               memref<1x1x!ttcore.tile<32x32, bf16>>)
            -> memref<1x1x!ttcore.tile<32x32, bf16>>

        // --- Compute: matmul [1,1] × [1,2] → [1,2] tiles ---
        linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d2, d1)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]
        }
        ins(%col_tiled, %wt_tiled :
            memref<1x1x!ttcore.tile<32x32, bf16>>,
            memref<1x2x!ttcore.tile<32x32, bf16>>)
        outs(%out_tiled :
            memref<1x2x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%in0: !ttcore.tile<32x32, bf16>,
             %in1: !ttcore.tile<32x32, bf16>,
             %acc: !ttcore.tile<32x32, bf16>):
          %mm = "d2m.tile_matmul"(%in0, %in1, %acc)
            : (!ttcore.tile<32x32, bf16>,
               !ttcore.tile<32x32, bf16>,
               !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
          linalg.yield %mm : !ttcore.tile<32x32, bf16>
        }

      } {d2m.blocking_loop = 0 : i64}

      // --- Compute: untilize [1,2] tiles → 32×64 row-major ---
      // (after the reduction loop so we untilize the fully accumulated result)
      "d2m.tile_untilize_block"(%out_tiled, %out_rm)
          : (memref<1x2x!ttcore.tile<32x32, bf16>>,
             memref<32x64xbf16>)
          -> memref<32x64xbf16>

      // --- DM: store output ---
      d2m.remote_store %out_stream[%gy, %gx] %out_rm
          : memref<1x1x32x64xbf16, #shard_out, #ttcore.view<4>, #l1_>,
            memref<32x64xbf16>
          -> memref<1x1x32x64xbf16, #shard_out, #ttcore.view<4>, #l1_>
    }
    %result = ttir.ttnn_metal_layout_cast %out_phys
        : memref<1x1x32x64xbf16, #shard_out, #l1_>
        -> tensor<32x64xbf16, #ttnn_out>
    return %result : tensor<32x64xbf16, #ttnn_out>
  }
}
