// RUN: ttmlir-opt --ttcore-register-device="mock-system-desc-arch=blackhole" --ttir-to-d2m --d2m-materialize-view-returns --d2m-grid-selection --canonicalize %s | FileCheck %s

module {
  // DMA-only generics still reblock operands through withParallelization, so
  // loop-space grid selection must respect operand/result reblock legality.
  // CHECK-LABEL: func.func @embedding_dma_loop_grid_reblockable
  // CHECK: d2m.generic
  // CHECK-SAME: block_factors = [1, 1]
  // CHECK-SAME: grid = #ttcore.grid<2x1
  // CHECK-SAME: threads = [#d2m.thread<datamovement>]
  // CHECK: d2m.embedding
  func.func @embedding_dma_loop_grid_reblockable(
      %indices: tensor<2x16xui32>,
      %weight: tensor<1024x32xf32>) -> tensor<2x16x32xf32> {
    %0 = "ttir.embedding"(%indices, %weight) : (tensor<2x16xui32>, tensor<1024x32xf32>) -> tensor<2x16x32xf32>
    return %0 : tensor<2x16x32xf32>
  }
}
