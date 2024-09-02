// RUN: ttmlir-opt --convert-ttkernel-to-emitc %s | FileCheck %s
#l1_ = #tt.memory_space<l1>
module attributes {} {
  func.func @ttkernel_noc() -> () {
    // CHECK: [[C:.*]] = "emitc.constant"[[C:.*]]
    %c262432_i32 = arith.constant 262432 : i32
    // CHECK: [[C:.*]] = "emitc.constant"[[C:.*]]
    %c262208_i32 = arith.constant 262208 : i32
    // CHECK: [[C:.*]] = "emitc.constant"[[C:.*]]
    %c32_i32 = arith.constant 32 : i32
    // CHECK: [[C:.*]] = "emitc.constant"[[C:.*]]
    %c262400_i32 = arith.constant 262400 : i32
    // CHECK: [[C:.*]] = "emitc.constant"[[C:.*]]
    %c0_i32 = arith.constant 0 : i32
    // CHECK: [[C:.*]] = "emitc.constant"[[C:.*]]
    %c262144_i32 = arith.constant 262144 : i32
    // CHECK: [[C:.*]] = emitc.call_opaque "get_noc_addr"[[C:.*]]
    %3 = "ttkernel.get_noc_addr"(%c0_i32, %c0_i32, %c262144_i32) : (i32, i32, i32) -> !ttkernel.noc_addr
    // CHECK: emitc.call_opaque "noc_async_read"[[C:.*]]
    "ttkernel.noc_async_read"(%3, %c262400_i32, %c32_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    // CHECK: [[C:.*]] = emitc.call_opaque "get_noc_addr"[[C:.*]]
    %4 = "ttkernel.get_noc_addr"(%c0_i32, %c0_i32, %c262208_i32) : (i32, i32, i32) -> !ttkernel.noc_addr
    // CHECK: emitc.call_opaque "noc_async_read"[[C:.*]]
    "ttkernel.noc_async_read"(%4, %c262432_i32, %c32_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    // CHECK: emitc.call_opaque "noc_async_read_barrier"[[C:.*]]
    "ttkernel.noc_async_read_barrier"() : () -> ()
    "ttkernel.return"() : () -> ()
  }

  func.func @ttkernel_tensix(%arg1: !ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>,
                             %arg2: !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> () {
      %c4_i32 = arith.constant 4 : i32
      // CHECK: emitc.call_opaque "untilize_init"[[C:.*]]
      "ttkernel.untilize_init"(%arg1, %arg2) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
      // CHECK: emitc.call_opaque "untilize_block"[[C:.*]]
      "ttkernel.untilize_block"(%arg1, %c4_i32, %arg2) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
      // CHECK: emitc.call_opaque "cb_pop_front"[[C:.*]]
      "ttkernel.cb_pop_front"(%arg1, %c4_i32) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      // CHECK: emitc.call_opaque "cb_push_back"[[C:.*]]
      "ttkernel.cb_push_back"(%arg2, %c4_i32) : (!ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      // CHECK: emitc.call_opaque "untilize_block"[[C:.*]]
      "ttkernel.untilize_block"(%arg1, %c4_i32, %arg2) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
      // CHECK: emitc.call_opaque "cb_pop_front"[[C:.*]]
      "ttkernel.cb_pop_front"(%arg1, %c4_i32) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      // CHECK: emitc.call_opaque "cb_push_back"[[C:.*]]
      "ttkernel.cb_push_back"(%arg2, %c4_i32) : (!ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      // CHECK: return
      "ttkernel.return"() : () -> ()
  }
}
