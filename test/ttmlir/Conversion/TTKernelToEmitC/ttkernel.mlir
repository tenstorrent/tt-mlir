// RUN: ttmlir-opt --convert-ttkernel-to-emitc %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
!cb0_type = !ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>

module attributes {} {

  // CHECK-LABEL: ttkernel_register_operations
  module @ttkernel_register_operations attributes {} {

    func.func @tile_regs_acquire() -> () {
      // CHECK: emitc.call_opaque "tile_regs_acquire"
      "ttkernel.tile_regs_acquire"() : () -> ()
      return
    }

    func.func @tile_regs_commit() -> () {
      // CHECK: emitc.call_opaque "tile_regs_commit"
      "ttkernel.tile_regs_commit"() : () -> ()
      return
    }

    func.func @tile_regs_wait() -> () {
      // CHECK: emitc.call_opaque "tile_regs_wait"
      "ttkernel.tile_regs_wait"() : () -> ()
      return
    }

    func.func @tile_regs_release() -> () {
      // CHECK: emitc.call_opaque "tile_regs_release"
      "ttkernel.tile_regs_release"() : () -> ()
      return
    }

    // CHECK-LABEL: func @pack_tile
    func.func @pack_tile(%out_cb: !cb0_type) -> () {
      // CHECK: %[[OUT_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: %[[OUT_CB_INDEX:.*]] = "emitc.constant"
      %out_cb_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "pack_tile"(%[[DST_INDEX]], %[[OUT_CB]], %[[OUT_CB_INDEX]])
      "ttkernel.pack_tile"(%dst_index, %out_cb, %out_cb_index) : (i32, !cb0_type, i32) -> ()
      return
    }

    // CHECK-LABEL: func @copy_tile_init
    func.func @copy_tile_init(%cb: !cb0_type) -> () {
      // CHECK: %[[CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB]])
      "ttkernel.copy_tile_init"(%cb) : (!cb0_type) -> ()
      return
    }

    // CHECK-LABEL: func @copy_tile
    func.func @copy_tile(%cb: !cb0_type) -> () {
      // CHECK: %[[CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB_INDEX:.*]] = "emitc.constant"
      %cb_index = arith.constant 2 : i32
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "copy_tile"(%[[CB]], %[[CB_INDEX]], %[[DST_INDEX]])
      "ttkernel.copy_tile"(%cb, %cb_index, %dst_index) : (!cb0_type, i32, i32) -> ()
      return
    }    
  }

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
