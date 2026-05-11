// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @lower_my_thread_id
  func.func @lower_my_thread_id(
      %A: memref<8x8x!ttcore.tile<32x32, bf16>, #l1>,
      %B: memref<8x8x!ttcore.tile<32x32, bf16>, #l1>) {
    d2m.generic {
        block_factors = [],
        grid = #ttcore.grid<1x1>,
        indexing_maps = [],
        iterator_types = [],
        threads = [#d2m.thread<unified>]
    } ins(%A : memref<8x8x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%B : memref<8x8x!ttcore.tile<32x32, bf16>, #l1>) {
      // CHECK: ttkernel.my_thread_id_
      // CHECK: arith.muli
      %tid = d2m.my_thread_id : index
      %c2 = arith.constant 2 : index
      %off = arith.muli %tid, %c2 : index
      %sub = memref.subview %B[%off, 0][2, 8][1, 1]
        : memref<8x8x!ttcore.tile<32x32, bf16>, #l1>
        to memref<2x8x!ttcore.tile<32x32, bf16>, strided<[8, 1], offset: ?>, #l1>
    }
    return
  }
}
