// Verifies that a DMA-only generic (remote_load + remote_store, no compute)
// correctly gets stream_layout ops inserted for both its inputs and outputs
// during the allocator pass.
// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate %s | FileCheck %s

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
#parallel = #ttcore.iterator_type<parallel>

module {
  // CHECK-LABEL: func.func @dma_only_load_store
  func.func @dma_only_load_store(
      %arg0: memref<1x1x32x32xf32, #ttcore.interleaved<128x4>, #dram>,
      %arg1: memref<1x1x32x32xf32, #ttcore.interleaved<128x4>, #dram>) -> memref<1x1x32x32xf32, #ttcore.interleaved<128x4>, #dram> {
    // CHECK-DAG: %[[IN_STREAM:.*]] = "d2m.stream_layout"(%arg0, %{{.*}})
    // CHECK-DAG: %[[OUT_STREAM:.*]] = "d2m.stream_layout"(%arg1, %{{.*}})
    // CHECK: d2m.generic
    // CHECK: ins(%[[IN_STREAM]]
    // CHECK: outs(%[[OUT_STREAM]]
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<1x1x32x32xf32, #ttcore.interleaved<128x4>, #dram>)
        outs(%arg1 : memref<1x1x32x32xf32, #ttcore.interleaved<128x4>, #dram>) {
    ^unified0(%cb0: !d2m.cb<memref<32x32xf32, #l1>>, %cb1: !d2m.cb<memref<32x32xf32, #l1>>):
      %block_factor0 = d2m.get_block_factor(0) : index
      %block_factor1 = d2m.get_block_factor(1) : index
      affine.for %arg2 = 0 to %block_factor0 {
        affine.for %arg3 = 0 to %block_factor1 {
          %block_offset0 = d2m.block_offset(0) : index
          %0 = affine.apply #map1(%arg2)[%block_offset0]
          %block_offset1 = d2m.block_offset(1) : index
          %1 = affine.apply #map1(%arg3)[%block_offset1]
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %2 = d2m.remote_load %alloc %arg0[%0, %1] : memref<32x32xf32>, memref<1x1x32x32xf32, #ttcore.interleaved<128x4>, #dram> -> memref<32x32xf32, #l1>
          %3 = d2m.remote_store %arg1[%0, %1] %alloc : memref<1x1x32x32xf32, #ttcore.interleaved<128x4>, #dram>, memref<32x32xf32> -> memref<1x1x32x32xf32, #ttcore.interleaved<128x4>, #dram>
        } {d2m.blocking_loop = 1 : i64}
      } {d2m.blocking_loop = 0 : i64}
    }
    return %arg1 : memref<1x1x32x32xf32, #ttcore.interleaved<128x4>, #dram>
  }
}
