// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-ttmetal-me-pipeline %s | FileCheck %s --check-prefix=CHECK-ME
// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-ttmetal-me-pipeline --ttir-to-ttmetal-be-pipeline %s -o /dev/null

// Minimal DMA-only generic
// One remote_load + one remote_store, no compute in region body.

#parallel = #ttcore.iterator_type<parallel>
#map = affine_map<(d0, d1) -> (d0, d1)>
#dram_layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, interleaved>

module {
  // CHECK-ME-LABEL: func.func @dma_only_load_store
  func.func @dma_only_load_store(
      %arg0: tensor<1x1x32x32xf32, #dram_layout>,
      %arg1: tensor<1x1x32x32xf32, #dram_layout>) -> tensor<1x1x32x32xf32, #dram_layout> {
    // CHECK-ME-DAG: %[[IN_STREAM:.*]] = "d2m.stream_layout"(%arg0, %{{.*}})
    // CHECK-ME-DAG: %[[OUT_STREAM:.*]] = "d2m.stream_layout"(%arg1, %{{.*}})
    // CHECK-ME: d2m.generic
    // CHECK-ME: ins(%[[IN_STREAM]]
    // CHECK-ME: outs(%[[OUT_STREAM]]
    %result = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0 : tensor<1x1x32x32xf32, #dram_layout>)
        outs(%arg1 : tensor<1x1x32x32xf32, #dram_layout>) {
    ^unified0(%cb0: !d2m.cb<tensor<32x32xf32>>, %cb1: !d2m.cb<tensor<32x32xf32>>):
      %block0 = d2m.block_index(0) : index
      %block1 = d2m.block_index(1) : index
      %tile = tensor.empty() : tensor<32x32xf32>
      %loaded = d2m.remote_load %tile %arg0[%block0, %block1] : tensor<32x32xf32>, tensor<1x1x32x32xf32, #dram_layout> -> tensor<32x32xf32>
      %stored = d2m.remote_store %arg1[%block0, %block1] %loaded : tensor<1x1x32x32xf32, #dram_layout>, tensor<32x32xf32> -> tensor<1x1x32x32xf32, #dram_layout>
      d2m.yield %stored : (tensor<1x1x32x32xf32, #dram_layout>)
    } : tensor<1x1x32x32xf32, #dram_layout>

    return %result : tensor<1x1x32x32xf32, #dram_layout>
  }
}
