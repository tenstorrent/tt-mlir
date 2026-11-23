// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-decompose-layouts --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// A TILE -> ROW_MAJOR to_layout that keeps the input's L1 shards (same
// sharding, grid, cores and tile-padded shard shape) is a single tt-metal
// sharded untilize and is left as is. Any other sharded untilize is still
// de-sharded first.

#l1 = #ttnn.buffer_type<l1>

// [1,192,16,832] height-sharded on 64 cores: 16 rows per image row padded to
// 32, 3 image rows per core -> shard [96, 832].
#hs_tile = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 6144 + d1 * 32 + d2, d3), <64x1>, memref<3x26x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#hs_rm_same_shard = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 6144 + d1 * 32 + d2, d3), <64x1>, memref<96x832xbf16, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
// Same cores, unpadded shard [48, 832]: not the shard the untilize produces.
#hs_rm_other_shard = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3072 + d1 * 16 + d2, d3), <64x1>, memref<48x832xbf16, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>

module attributes {} {
  // CHECK-LABEL: func.func @untilize_in_place
  func.func @untilize_in_place(%arg0: tensor<1x192x16x832xbf16, #hs_tile>) -> tensor<1x192x16x832xbf16, #hs_rm_same_shard> {
    // CHECK: %[[R:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: memref<96x832xbf16
    // CHECK-NOT: "ttnn.to_memory_config"
    // CHECK: return %[[R]]
    %0 = "ttnn.to_layout"(%arg0) : (tensor<1x192x16x832xbf16, #hs_tile>) -> tensor<1x192x16x832xbf16, #hs_rm_same_shard>
    return %0 : tensor<1x192x16x832xbf16, #hs_rm_same_shard>
  }

  // CHECK-LABEL: func.func @untilize_other_shard
  func.func @untilize_other_shard(%arg0: tensor<1x192x16x832xbf16, #hs_tile>) -> tensor<1x192x16x832xbf16, #hs_rm_other_shard> {
    // CHECK: "ttnn.to_memory_config"(%arg0)
    // CHECK: "ttnn.to_layout"
    %0 = "ttnn.to_layout"(%arg0) : (tensor<1x192x16x832xbf16, #hs_tile>) -> tensor<1x192x16x832xbf16, #hs_rm_other_shard>
    return %0 : tensor<1x192x16x832xbf16, #hs_rm_other_shard>
  }
}
