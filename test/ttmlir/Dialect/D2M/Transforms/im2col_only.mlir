// Post-bufferize im2col gather only (no matmul, no tilize/untilize).
//
// Extracts the d2m.l1_copy im2col gather from the full conv pipeline.
// Activation shard is loaded, im2col gathers the full 64×32 column matrix,
// and the result is stored directly.
//
// Setup:
//   Image:   H=9, W=9, C=8 → activation shard: memref<81x8xbf16>
//   Kernel:  K_h=2, K_w=2, stride=1, no padding
//   OH=8, OW=8 → 64 output pixels
//   K = K_h*K_w*C = 2*2*8 = 32  (exactly one tile width)
//
//   Full im2col output: [64, 32] bf16

#l1_ = #ttcore.memory_space<l1>
#l1_ttnn = #ttnn.buffer_type<l1>

#shard_act = #ttcore.shard<16x2, 1>
#shard_out = #ttcore.shard<128x1, 1>
#map_id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#ttnn_act = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<81x8xbf16, #l1_ttnn>, <height_sharded>, exactGrid = true>
#ttnn_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32xbf16, #l1_ttnn>, <height_sharded>, exactGrid = true>

module {
  func.func @im2col_only(
      %arg0 : tensor<81x8xbf16, #ttnn_act>,
      %arg1 : tensor<64x32xbf16, #ttnn_out>) -> tensor<64x32xbf16, #ttnn_out> {

    %act_phys = ttir.ttnn_metal_layout_cast %arg0
        : tensor<81x8xbf16, #ttnn_act>
        -> memref<1x1x81x8xbf16, #shard_act, #l1_>
    %out_phys = ttir.ttnn_metal_layout_cast %arg1
        : tensor<64x32xbf16, #ttnn_out>
        -> memref<1x1x64x32xbf16, #shard_out, #l1_>

    %act_local = memref.alloc() {alignment = 16 : i64}
        : memref<1x1x81x8xbf16, #shard_act, #l1_>
    %out_local = memref.alloc() {alignment = 16 : i64}
        : memref<1x1x64x32xbf16, #shard_out, #l1_>

    %act_stream = "d2m.stream_layout"(%act_phys, %act_local) <{remapping = #map_id}>
        : (memref<1x1x81x8xbf16, #shard_act, #l1_>,
           memref<1x1x81x8xbf16, #shard_act, #l1_>)
        -> memref<1x1x81x8xbf16, #shard_act, #ttcore.view<4>, #l1_>
    %out_stream = "d2m.stream_layout"(%out_phys, %out_local) <{remapping = #map_id}>
        : (memref<1x1x64x32xbf16, #shard_out, #l1_>,
           memref<1x1x64x32xbf16, #shard_out, #l1_>)
        -> memref<1x1x64x32xbf16, #shard_out, #ttcore.view<4>, #l1_>

    d2m.generic {
        block_factors = [], grid = #ttcore.grid<1x1>,
        indexing_maps = [], iterator_types = [],
        threads = [#d2m.thread<unified>]
    }
    ins(%act_stream :
        memref<1x1x81x8xbf16, #shard_act, #ttcore.view<4>, #l1_>)
    outs(%out_stream :
        memref<1x1x64x32xbf16, #shard_out, #ttcore.view<4>, #l1_>) {
    ^unified0:
      %gy = d2m.core_index(0) : index
      %gx = d2m.core_index(1) : index

      // Load activation shard into local buffer.
      %act_alloc = memref.alloc() {alignment = 64 : i64} : memref<81x8xbf16>
      %act = d2m.remote_load %act_alloc %act_stream[%gy, %gx]
          : memref<81x8xbf16>,
            memref<1x1x81x8xbf16, #shard_act, #ttcore.view<4>, #l1_>
          -> memref<81x8xbf16, #l1_>

      // Full im2col gather: 64×32 column matrix.
      %col_rm = memref.alloc() {alignment = 64 : i64} : memref<64x32xbf16>
      d2m.l1_copy %act, %col_rm
          indexing_maps = [
            affine_map<(d0, d1) -> (
              (d0 floordiv 8 + d1 floordiv 16) * 9
              + d0 mod 8 + (d1 mod 16) floordiv 8,
              d1 mod 8)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ]
          : memref<81x8xbf16, #l1_>, memref<64x32xbf16>

      // Store im2col result.
      d2m.remote_store %out_stream[%gy, %gx] %col_rm
          : memref<1x1x64x32xbf16, #shard_out, #ttcore.view<4>, #l1_>,
            memref<64x32xbf16>
          -> memref<1x1x64x32xbf16, #shard_out, #ttcore.view<4>, #l1_>
    }
    %result = ttir.ttnn_metal_layout_cast %out_phys
        : memref<1x1x64x32xbf16, #shard_out, #l1_>
        -> tensor<64x32xbf16, #ttnn_out>
    return %result : tensor<64x32xbf16, #ttnn_out>
  }
}
