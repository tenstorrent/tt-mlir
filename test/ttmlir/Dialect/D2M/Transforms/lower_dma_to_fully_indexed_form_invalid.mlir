// RUN: not ttmlir-opt --ttcore-register-device --d2m-lower-dma-to-fully-indexed-form="use-tensor-accessor-dma=true" %s 2>&1 | FileCheck %s

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#page_map = affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d1 * 2 + d2 * 2 + d3)>

module {
  func.func @preannotated_dma(
      %remote: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>,
      %local: memref<2x2x!ttcore.tile<32x32, f32>, #l1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: error: 'd2m.dma_read' op unexpected pre-existing TensorAccessor page map
    %tx = d2m.dma_read %remote[%c0, %c1], %local, <0> {tensorAccessorPageMap = #page_map} : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
    return
  }
}
