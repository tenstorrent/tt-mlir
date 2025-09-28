// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s -dump-input=always

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

// CHECK-LABEL: func.func @no_dma_wait
func.func @no_dma_wait(%arg0: memref<2x2x!ttcore.tile<32x32, f32>, #l1>, %arg1: memref<2x2x!ttcore.tile<32x32, f32>, #l1>) attributes {d2m.thread = #d2m.thread<datamovement>} {
  %c0 = arith.constant 0 : index
  // CHECK: ttkernel.noc_async_write
  %tx = d2m.dma_write %arg0[%c0], %arg1[%c0], <1> : (memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
  // CHECK-NOT: ttkernel.noc_async_write_barrier
  return
}
