// RUN: ttmlir-opt --convert-ttkernel-to-emitc %s | FileCheck %s

#l1_ = #tt.memory_space<l1>

!cb0_type = !ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>
!cb1_type = !ttkernel.cb<cb_in1, 327680, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>
!cb2_type = !ttkernel.cb<cb_in2, 327680, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>

module attributes {} {

  //===----------------------------------------------------------------------===//
  // TTKernel Register operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_register_operations
  module @ttkernel_register_operations attributes {} {

    // CHECK-LABEL: func @tile_regs_acquire
    func.func @tile_regs_acquire() -> () {
      // CHECK: emitc.call_opaque "tile_regs_acquire"()
      "ttkernel.tile_regs_acquire"() : () -> ()
      return
    }

    // CHECK-LABEL: func @tile_regs_commit
    func.func @tile_regs_commit() -> () {
      // CHECK: emitc.call_opaque "tile_regs_commit"()
      "ttkernel.tile_regs_commit"() : () -> ()
      return
    }

    // CHECK-LABEL: func @tile_regs_wait
    func.func @tile_regs_wait() -> () {
      // CHECK: emitc.call_opaque "tile_regs_wait"()
      "ttkernel.tile_regs_wait"() : () -> ()
      return
    }

    // CHECK-LABEL: func @tile_regs_release
    func.func @tile_regs_release() -> () {
      // CHECK: emitc.call_opaque "tile_regs_release"()
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
  } // module

  //===----------------------------------------------------------------------===//
  // TTKernel FPU operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_fpu_operations
  module @ttkernel_fpu_operations attributes {} {

    // CHECK-LABEL: func @binary_op_init_common
    func.func @binary_op_init_common(%cb0: !cb0_type, %cb1: !cb1_type, %out_cb: !cb2_type) -> () {
      // CHECK: %[[CB0:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB1:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[OUT_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "binary_op_init_common"(%[[CB0]], %[[CB1]], %[[OUT_CB]])
      "ttkernel.binary_op_init_common"(%cb0, %cb1, %out_cb) : (!cb0_type, !cb1_type, !cb2_type) -> ()
      return
    }

    // CHECK-LABEL: func @add_tiles_init
    func.func @add_tiles_init(%cb0: !cb0_type, %cb1: !cb1_type) -> () {
      // CHECK: %[[CB0:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB1:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "add_tiles_init"(%[[CB0]], %[[CB1]])
      "ttkernel.add_tiles_init"(%cb0, %cb1) : (!cb0_type, !cb1_type) -> ()
      return
    }

    // CHECK-LABEL: func @add_tiles
    func.func @add_tiles(%cb0: !cb0_type, %cb1: !cb1_type) -> () {
      // CHECK: %[[CB0:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB1:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB0_INDEX:.*]] = "emitc.constant"
      %cb0_index = arith.constant 1 : i32
      // CHECK: %[[CB1_INDEX:.*]] = "emitc.constant"
      %cb1_index = arith.constant 2 : i32
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "add_tiles"(%[[CB0]], %[[CB1]], %[[CB0_INDEX]], %[[CB1_INDEX]], %[[DST_INDEX]])
      "ttkernel.add_tiles"(%cb0, %cb1, %cb0_index, %cb1_index, %dst_index) : (!cb0_type, !cb1_type, i32, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @mul_tiles_init
    func.func @mul_tiles_init(%cb0: !cb0_type, %cb1: !cb1_type) -> () {
      // CHECK: %[[CB0:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB1:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "mul_tiles_init"(%[[CB0]], %[[CB1]])
      "ttkernel.mul_tiles_init"(%cb0, %cb1) : (!cb0_type, !cb1_type) -> ()
      return
    }

    // CHECK-LABEL: func @mul_tiles_init_f
    func.func @mul_tiles_init_f() -> () {
      // CHECK: emitc.call_opaque "mul_tiles_init_f"()
      "ttkernel.mul_tiles_init_f"() : () -> ()
      return
    }

    // CHECK-LABEL: func @mul_tiles
    func.func @mul_tiles(%cb0: !cb0_type, %cb1: !cb1_type) -> () {
      // CHECK: %[[CB0:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB1:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB0_INDEX:.*]] = "emitc.constant"
      %cb0_index = arith.constant 1 : i32
      // CHECK: %[[CB1_INDEX:.*]] = "emitc.constant"
      %cb1_index = arith.constant 2 : i32
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "mul_tiles"(%[[CB0]], %[[CB1]], %[[CB0_INDEX]], %[[CB1_INDEX]], %[[DST_INDEX]])
      "ttkernel.mul_tiles"(%cb0, %cb1, %cb0_index, %cb1_index, %dst_index) : (!cb0_type, !cb1_type, i32, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @unary_op_init_common
    func.func @unary_op_init_common(%in_cb: !cb0_type, %out_cb: !cb1_type) -> () {
      // CHECK: %[[IN_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[OUT_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "unary_op_init_common"(%[[IN_CB]], %[[OUT_CB]])
      "ttkernel.unary_op_init_common"(%in_cb, %out_cb) : (!cb0_type, !cb1_type) -> ()
      return
    }

    // CHECK-LABEL: func @exp_tile_init
    func.func @exp_tile_init() -> () {
      // CHECK: emitc.call_opaque "exp_tile_init"()
      "ttkernel.exp_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @exp_tile
    func.func @exp_tile() -> () {
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "exp_tile"(%[[DST_INDEX]])
      "ttkernel.exp_tile"(%dst_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @recip_tile_init
    func.func @recip_tile_init() -> () {
      // CHECK: emitc.call_opaque "recip_tile_init"()
      "ttkernel.recip_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @recip_tile
    func.func @recip_tile() -> () {
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "recip_tile"(%[[DST_INDEX]])
      "ttkernel.recip_tile"(%dst_index) : (i32) -> ()
      return
    }

    // CHECK-LABEL: func @reduce_init
    func.func @reduce_init(%in_cb: !cb0_type, %scaling_cb: !cb1_type, %out_cb: !cb2_type) -> () {
      // CHECK: %[[IN_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[SCALING_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[OUT_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "reduce_init"(%[[IN_CB]], %[[SCALING_CB]], %[[OUT_CB]]) {{.+}}SUM{{.+}}REDUCE_SCALAR
      "ttkernel.reduce_init"(%in_cb, %scaling_cb, %out_cb) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_scalar>, reduce_type = #ttkernel.reduce_type<reduce_sum>}> : (!cb0_type, !cb1_type, !cb2_type) -> ()
      return
    }    

    // CHECK-LABEL: func @reduce_tile
    func.func @reduce_tile(%in_cb: !cb0_type, %scaling_cb: !cb1_type) -> () {
      // CHECK: %[[IN_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[SCALING_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[IN_TILE_INDEX:.*]] = "emitc.constant"
      %in_tile_index = arith.constant 1 : i32
      // CHECK: %[[SCALING_TILE_INDEX:.*]] = "emitc.constant"
      %scaling_tile_index = arith.constant 2 : i32
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "reduce_tile"(%[[IN_CB]], %[[SCALING_CB]],  %[[IN_TILE_INDEX]], %[[SCALING_TILE_INDEX]], %[[DST_INDEX]]) {{.+}}MAX{{.+}}REDUCE_ROW
      "ttkernel.reduce_tile"(%in_cb, %scaling_cb, %in_tile_index, %scaling_tile_index, %dst_index) <{
        reduce_dim = #ttkernel.reduce_dim<reduce_dim_row>, reduce_type = #ttkernel.reduce_type<reduce_max>
        }> : (!cb0_type, !cb1_type, i32, i32, i32) -> ()
      return
    }
  } // module

  //===----------------------------------------------------------------------===//
  // TTKernel SFPU operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_sfpu_operations
  module @ttkernel_sfpu_operations attributes {} {  

    // CHECK-LABEL: func @max_tile_init
    func.func @max_tile_init() -> () {
      // CHECK: emitc.call_opaque "max_tile_init"()
      "ttkernel.max_tile_init"() : () -> ()
      return
    }

    // CHECK-LABEL: func @max_tile
    func.func @max_tile() -> () {
      // CHECK: %[[DST0_INDEX:.*]] = "emitc.constant"
      %dst0_index = arith.constant 1 : i32
      // CHECK: %[[DST1_INDEX:.*]] = "emitc.constant"
      %dst1_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "max_tile"(%[[DST0_INDEX]], %[[DST1_INDEX]])
      "ttkernel.max_tile"(%dst0_index, %dst1_index) : (i32, i32) -> ()
      return
    }

  } // module

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
