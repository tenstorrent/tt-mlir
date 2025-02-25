// RUN: ttmlir-opt --convert-ttkernel-to-emitc %s | FileCheck %s
#l1_ = #tt.memory_space<l1>
module attributes {} {
  func.func @ttkernel_noc() -> () {
    // CHECK: = "emitc.constant"
    %c262432_i32 = arith.constant 262432 : i32
    // CHECK: = "emitc.constant"
    %c262208_i32 = arith.constant 262208 : i32
    // CHECK: = "emitc.constant"
    %c32_i32 = arith.constant 32 : i32
    // CHECK: = "emitc.constant"
    %c262400_i32 = arith.constant 262400 : i32
    // CHECK: = "emitc.constant"
    %c0_i32 = arith.constant 0 : i32
    // CHECK: = "emitc.constant"
    %c262144_i32 = arith.constant 262144 : i32
    // CHECK: = emitc.call_opaque "get_noc_addr"
    %3 = "ttkernel.get_noc_addr_xy"(%c0_i32, %c0_i32, %c262144_i32) : (i32, i32, i32) -> !ttkernel.noc_addr
    // CHECK: emitc.call_opaque "noc_async_read"
    "ttkernel.noc_async_read"(%3, %c262400_i32, %c32_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    // CHECK: = emitc.call_opaque "get_noc_addr"
    %4 = "ttkernel.get_noc_addr_xy"(%c0_i32, %c0_i32, %c262208_i32) : (i32, i32, i32) -> !ttkernel.noc_addr
    // CHECK: emitc.call_opaque "noc_async_read"
    "ttkernel.noc_async_read"(%4, %c262432_i32, %c32_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    %bank_id = arith.constant 1 : i32
    %addr_offset = arith.constant 262400 : i32
    %noc_addr = "ttkernel.get_noc_addr_from_bank_id"(%bank_id, %addr_offset) : (i32, i32) -> !ttkernel.noc_addr
    // CHECK: = emitc.call_opaque "get_noc_addr_from_bank_id"({{.*}}) {template_args = [#emitc.opaque<"true">]}
    // CHECK: emitc.call_opaque "noc_async_read_barrier"
    "ttkernel.noc_async_read_barrier"() : () -> ()
    "ttkernel.return"() : () -> ()
  }

  func.func @ttkernel_tensix(%arg1: !ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>,
                             %arg2: !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> () {
      %c4_i32 = arith.constant 4 : i32
      // CHECK: emitc.call_opaque "untilize_init"
      "ttkernel.untilize_init"(%arg1, %arg2) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
      // CHECK: emitc.call_opaque "untilize_block"
      "ttkernel.untilize_block"(%arg1, %c4_i32, %arg2) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
      // CHECK: emitc.call_opaque "cb_pop_front"
      "ttkernel.cb_pop_front"(%arg1, %c4_i32) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      // CHECK: emitc.call_opaque "cb_push_back"
      "ttkernel.cb_push_back"(%arg2, %c4_i32) : (!ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      // CHECK: emitc.call_opaque "untilize_block"
      "ttkernel.untilize_block"(%arg1, %c4_i32, %arg2) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32, !ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>) -> ()
      // CHECK: emitc.call_opaque "cb_pop_front"
      "ttkernel.cb_pop_front"(%arg1, %c4_i32) : (!ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>, i32) -> ()
      // CHECK: emitc.call_opaque "cb_push_back"
      "ttkernel.cb_push_back"(%arg2, %c4_i32) : (!ttkernel.cb<cb_out0, 327680, memref<64x128xf32, #l1_>, 4096, 1>, i32) -> ()
      // CHECK: return
      "ttkernel.return"() : () -> ()
  }
}
