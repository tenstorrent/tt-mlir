// RUN: ttmlir-opt --allow-unregistered-dialect %s | FileCheck %s

// Round-trip test for d2m.my_thread_id.

#l1 = #ttcore.memory_space<l1>

func.func @my_thread_id_in_generic(
    %arg0: memref<8x8x!ttcore.tile<32x32, bf16>, #l1>,
    %arg1: memref<8x8x!ttcore.tile<32x32, bf16>, #l1>) {
  d2m.generic {
      block_factors = [],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [],
      iterator_types = [],
      threads = [#d2m.thread<unified>]
  } ins(%arg0 : memref<8x8x!ttcore.tile<32x32, bf16>, #l1>)
    outs(%arg1 : memref<8x8x!ttcore.tile<32x32, bf16>, #l1>) {
    // CHECK: %[[TID:.*]] = d2m.my_thread_id : index
    %tid = d2m.my_thread_id : index
    // CHECK: "use"(%[[TID]]) : (index) -> ()
    "use"(%tid) : (index) -> ()
  }
  return
}
