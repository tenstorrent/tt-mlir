// RUN: ttmlir-opt --convert-ttkernel-to-emitc %s | FileCheck %s

#l1_ = #tt.memory_space<l1>

!cb0_scalar = !ttkernel.cb<cb_in0, 294912, memref<64x128xf32, #l1_>, 4096, 1>
!cb1_scalar = !ttkernel.cb<cb_in1, 299008, memref<64x128xf32, #l1_>, 4096, 1>
!cb2_scalar = !ttkernel.cb<cb_in2, 303104, memref<64x128xf32, #l1_>, 4096, 1>

!cb0_tiles = !ttkernel.cb<cb_in0, 294912, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>
!cb1_tiles = !ttkernel.cb<cb_in1, 299008, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>
!cb2_tiles = !ttkernel.cb<cb_in2, 303104, memref<2x4x!tt.tile<32x32, f32>, #l1_>, 4096, 1>

module {

  //===----------------------------------------------------------------------===//
  // TTKernel Register operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_register_operations
  module @ttkernel_register_operations {

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
    func.func @pack_tile(%out_cb: !cb0_tiles) -> () {
      // CHECK: %[[OUT_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: %[[OUT_CB_INDEX:.*]] = "emitc.constant"
      %out_cb_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "pack_tile"(%[[DST_INDEX]], %[[OUT_CB]], %[[OUT_CB_INDEX]])
      "ttkernel.pack_tile"(%dst_index, %out_cb, %out_cb_index) : (i32, !cb0_tiles, i32) -> ()
      return
    }

    // CHECK-LABEL: func @copy_tile_init
    func.func @copy_tile_init(%cb: !cb0_tiles) -> () {
      // CHECK: %[[CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB]])
      "ttkernel.copy_tile_init"(%cb) : (!cb0_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @copy_tile
    func.func @copy_tile(%cb: !cb0_tiles) -> () {
      // CHECK: %[[CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB_INDEX:.*]] = "emitc.constant"
      %cb_index = arith.constant 2 : i32
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "copy_tile"(%[[CB]], %[[CB_INDEX]], %[[DST_INDEX]])
      "ttkernel.copy_tile"(%cb, %cb_index, %dst_index) : (!cb0_tiles, i32, i32) -> ()
      return
    }

  } // module

  //===----------------------------------------------------------------------===//
  // TTKernel FPU operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_fpu_operations
  module @ttkernel_fpu_operations {

    // CHECK-LABEL: func @binary_op_init_common
    func.func @binary_op_init_common(%cb0: !cb0_tiles, %cb1: !cb1_tiles, %out_cb: !cb2_tiles) -> () {
      // CHECK: %[[CB0:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB1:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[OUT_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "binary_op_init_common"(%[[CB0]], %[[CB1]], %[[OUT_CB]])
      "ttkernel.binary_op_init_common"(%cb0, %cb1, %out_cb) : (!cb0_tiles, !cb1_tiles, !cb2_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @add_tiles_init
    func.func @add_tiles_init(%cb0: !cb0_tiles, %cb1: !cb1_tiles) -> () {
      // CHECK: %[[CB0:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB1:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "add_tiles_init"(%[[CB0]], %[[CB1]])
      "ttkernel.add_tiles_init"(%cb0, %cb1) : (!cb0_tiles, !cb1_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @add_tiles
    func.func @add_tiles(%cb0: !cb0_tiles, %cb1: !cb1_tiles) -> () {
      // CHECK: %[[CB0:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB1:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB0_INDEX:.*]] = "emitc.constant"
      %cb0_index = arith.constant 1 : i32
      // CHECK: %[[CB1_INDEX:.*]] = "emitc.constant"
      %cb1_index = arith.constant 2 : i32
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "add_tiles"(%[[CB0]], %[[CB1]], %[[CB0_INDEX]], %[[CB1_INDEX]], %[[DST_INDEX]])
      "ttkernel.add_tiles"(%cb0, %cb1, %cb0_index, %cb1_index, %dst_index) : (!cb0_tiles, !cb1_tiles, i32, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @mul_tiles_init
    func.func @mul_tiles_init(%cb0: !cb0_tiles, %cb1: !cb1_tiles) -> () {
      // CHECK: %[[CB0:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB1:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "mul_tiles_init"(%[[CB0]], %[[CB1]])
      "ttkernel.mul_tiles_init"(%cb0, %cb1) : (!cb0_tiles, !cb1_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @mul_tiles_init_f
    func.func @mul_tiles_init_f() -> () {
      // CHECK: emitc.call_opaque "mul_tiles_init_f"()
      "ttkernel.mul_tiles_init_f"() : () -> ()
      return
    }

    // CHECK-LABEL: func @mul_tiles
    func.func @mul_tiles(%cb0: !cb0_tiles, %cb1: !cb1_tiles) -> () {
      // CHECK: %[[CB0:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB1:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[CB0_INDEX:.*]] = "emitc.constant"
      %cb0_index = arith.constant 1 : i32
      // CHECK: %[[CB1_INDEX:.*]] = "emitc.constant"
      %cb1_index = arith.constant 2 : i32
      // CHECK: %[[DST_INDEX:.*]] = "emitc.constant"
      %dst_index = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "mul_tiles"(%[[CB0]], %[[CB1]], %[[CB0_INDEX]], %[[CB1_INDEX]], %[[DST_INDEX]])
      "ttkernel.mul_tiles"(%cb0, %cb1, %cb0_index, %cb1_index, %dst_index) : (!cb0_tiles, !cb1_tiles, i32, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @unary_op_init_common
    func.func @unary_op_init_common(%in_cb: !cb0_tiles, %out_cb: !cb1_tiles) -> () {
      // CHECK: %[[IN_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[OUT_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "unary_op_init_common"(%[[IN_CB]], %[[OUT_CB]])
      "ttkernel.unary_op_init_common"(%in_cb, %out_cb) : (!cb0_tiles, !cb1_tiles) -> ()
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
    func.func @reduce_init(%in_cb: !cb0_tiles, %scaling_cb: !cb1_tiles, %out_cb: !cb2_tiles) -> () {
      // CHECK: %[[IN_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[SCALING_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[OUT_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "reduce_init"(%[[IN_CB]], %[[SCALING_CB]], %[[OUT_CB]]) {{.+}}SUM{{.+}}REDUCE_SCALAR
      "ttkernel.reduce_init"(%in_cb, %scaling_cb, %out_cb) <{reduce_dim = #ttkernel.reduce_dim<reduce_dim_scalar>, reduce_type = #ttkernel.reduce_type<reduce_sum>}> : (!cb0_tiles, !cb1_tiles, !cb2_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @reduce_tile
    func.func @reduce_tile(%in_cb: !cb0_tiles, %scaling_cb: !cb1_tiles) -> () {
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
        }> : (!cb0_tiles, !cb1_tiles, i32, i32, i32) -> ()
      return
    }

  } // module

  //===----------------------------------------------------------------------===//
  // TTKernel SFPU operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_sfpu_operations
  module @ttkernel_sfpu_operations {

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

  //===----------------------------------------------------------------------===//
  // TTKernel CB operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_cb_operations
  module @ttkernel_cb_operations {

    // CHECK-LABEL: func @cb_push_back
    func.func @cb_push_back(%cb: !cb0_tiles) -> () {
      // CHECK: %[[CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[NUM_PAGES:.*]] = "emitc.constant"
      %num_pages = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "cb_push_back"(%[[CB]], %[[NUM_PAGES]])
      "ttkernel.cb_push_back"(%cb, %num_pages) : (!cb0_tiles, i32) -> ()
      return
    }

    // CHECK-LABEL: func @cb_pop_front
    func.func @cb_pop_front(%cb: !cb0_tiles) -> () {
      // CHECK: %[[CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[NUM_PAGES:.*]] = "emitc.constant"
      %num_pages = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "cb_pop_front"(%[[CB]], %[[NUM_PAGES]])
      "ttkernel.cb_pop_front"(%cb, %num_pages) : (!cb0_tiles, i32) -> ()
      return
    }

    // CHECK-LABEL: func @cb_reserve_back
    func.func @cb_reserve_back(%cb: !cb0_tiles) -> () {
      // CHECK: %[[CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[NUM_PAGES:.*]] = "emitc.constant"
      %num_pages = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "cb_reserve_back"(%[[CB]], %[[NUM_PAGES]])
      "ttkernel.cb_reserve_back"(%cb, %num_pages) : (!cb0_tiles, i32) -> ()
      return
    }

    // CHECK-LABEL: func @cb_wait_front
    func.func @cb_wait_front(%cb: !cb0_tiles) -> () {
      // CHECK: %[[CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[NUM_PAGES:.*]] = "emitc.constant"
      %num_pages = arith.constant 1 : i32
      // CHECK: emitc.call_opaque "cb_wait_front"(%[[CB]], %[[NUM_PAGES]])
      "ttkernel.cb_wait_front"(%cb, %num_pages) : (!cb0_tiles, i32) -> ()
      return
    }

  } // module

  //===----------------------------------------------------------------------===//
  // TTKernel Tile operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_tile_operations
  module @ttkernel_tile_operations {

    // CHECK-LABEL: func @tilize_init
    func.func @tilize_init(%in_cb: !cb0_scalar, %out_cb: !cb1_tiles) -> () {
      // CHECK: %[[IN_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[OUT_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[NUM_TILES:.*]] = "emitc.constant"
      %num_tiles = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "tilize_init"(%[[IN_CB]], %[[NUM_TILES]], %[[OUT_CB]])
      "ttkernel.tilize_init"(%in_cb, %num_tiles, %out_cb) : (!cb0_scalar, i32, !cb1_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @untilize_init
    func.func @untilize_init(%in_cb: !cb0_tiles, %out_cb: !cb1_scalar) -> () {
      // CHECK: %[[IN_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[OUT_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: emitc.call_opaque "untilize_init"(%[[IN_CB]], %[[OUT_CB]])
      "ttkernel.untilize_init"(%in_cb, %out_cb) : (!cb0_tiles, !cb1_scalar) -> ()
      return
    }

    // CHECK-LABEL: func @tilize_block
    func.func @tilize_block(%in_cb: !cb0_scalar, %out_cb: !cb1_tiles) -> () {
      // CHECK: %[[IN_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[OUT_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[NUM_TILES:.*]] = "emitc.constant"
      %num_tiles = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "tilize_block"(%[[IN_CB]], %[[NUM_TILES]], %[[OUT_CB]])
      "ttkernel.tilize_block"(%in_cb, %num_tiles, %out_cb) : (!cb0_scalar, i32, !cb1_tiles) -> ()
      return
    }

    // CHECK-LABEL: func @untilize_block
    func.func @untilize_block(%in_cb: !cb0_tiles, %out_cb: !cb1_scalar) -> () {
      // CHECK: %[[IN_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[OUT_CB:.*]] = emitc.load{{.+}}<"::tt::CB">
      // CHECK: %[[NUM_TILES:.*]] = "emitc.constant"
      %num_tiles = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "untilize_block"(%[[IN_CB]], %[[NUM_TILES]], %[[OUT_CB]])
      "ttkernel.untilize_block"(%in_cb, %num_tiles, %out_cb) : (!cb0_tiles, i32, !cb1_scalar) -> ()
      return
    }

  } // module

  //===----------------------------------------------------------------------===//
  // TTKernel NOC operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: ttkernel_noc_operations
  module @ttkernel_noc_operations {

    // CHECK-LABEL: func @get_noc_addr_xy
    func.func @get_noc_addr_xy() -> () {
      // CHECK: %[[X:.*]] = "emitc.constant"
      %x = arith.constant 1 : i32
      // CHECK: %[[Y:.*]] = "emitc.constant"
      %y = arith.constant 2 : i32
      // CHECK: %[[ADDR:.*]] = "emitc.constant"
      %addr = arith.constant 262400 : i32
      // note: ttkernel.get_noc_addr_xy() converts to one of metallium get_noc_addr() overloads
      // CHECK: emitc.call_opaque "get_noc_addr"(%[[X]], %[[Y]], %[[ADDR]])
      "ttkernel.get_noc_addr_xy"(%x, %y, %addr) : (i32, i32, i32) -> (!ttkernel.noc_addr)
      return
    }

    // CHECK-LABEL: func @get_noc_addr_from_bank_id
    func.func @get_noc_addr_from_bank_id() -> () {
      // CHECK: %[[BANK_ID:.*]] = "emitc.constant"
      %bank_id = arith.constant 1 : i32
      // CHECK: %[[ADDR_OFFSET:.*]] = "emitc.constant"
      %addr_offset = arith.constant 262400 : i32
      // CHECK: emitc.call_opaque "get_noc_addr_from_bank_id"(%[[BANK_ID]], %[[ADDR_OFFSET]])
      "ttkernel.get_noc_addr_from_bank_id"(%bank_id, %addr_offset) : (i32, i32) -> (!ttkernel.noc_addr)
      return
    }

    // CHECK-LABEL: func @get_noc_addr
    func.func @get_noc_addr() -> () {
      // CHECK: %[[ADDR:.*]] = "emitc.constant"
      %addr = arith.constant 262400 : i32
      // CHECK: emitc.call_opaque "get_noc_addr"(%[[ADDR]])
      "ttkernel.get_noc_addr"(%addr) : (i32) -> (!ttkernel.noc_addr)
      return
    }

    // CHECK-LABEL: func @noc_async_read
    func.func @noc_async_read() -> () {
      // CHECK: %[[SRC_ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %temp = arith.constant 262400 : i32
      %src_addr = "ttkernel.get_noc_addr"(%temp) : (i32) -> (!ttkernel.noc_addr)
      // CHECK: %[[DST_ADDR:.*]] = "emitc.constant"
      %dst_addr = arith.constant 303104 : i32
      // CHECK: %[[SIZE:.*]] = "emitc.constant"
      %size = arith.constant 2048 : i32
      // CHECK: emitc.call_opaque "noc_async_read"(%[[SRC_ADDR]], %[[DST_ADDR]], %[[SIZE]])
      "ttkernel.noc_async_read"(%src_addr, %dst_addr, %size) : (!ttkernel.noc_addr, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_async_read_one_packet_set_state
    func.func @noc_async_read_one_packet_set_state() -> () {
      // CHECK: %[[SRC_ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %temp = arith.constant 262400 : i32
      %src_addr = "ttkernel.get_noc_addr"(%temp) : (i32) -> (!ttkernel.noc_addr)
      // CHECK: %[[SIZE:.*]] = "emitc.constant"
      %size = arith.constant 2048 : i32
      // CHECK: emitc.call_opaque "noc_async_read_one_packet_set_state"(%[[SRC_ADDR]], %[[SIZE]])
      "ttkernel.noc_async_read_one_packet_set_state"(%src_addr, %size) : (!ttkernel.noc_addr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_async_read_one_packet_with_state
    func.func @noc_async_read_one_packet_with_state() -> () {
      // CHECK: %[[SRC_ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %temp1 = arith.constant 262400 : i32
      %src_addr = "ttkernel.get_noc_addr"(%temp1) : (i32) -> (!ttkernel.noc_addr)
      // CHECK: %[[DST_ADDR:.*]] = "emitc.constant"
      %dst_addr = arith.constant 327680 : i32
      // CHECK: emitc.call_opaque "noc_async_read_one_packet_with_state"(%[[SRC_ADDR]], %[[DST_ADDR]])
      "ttkernel.noc_async_read_one_packet_with_state"(%src_addr, %dst_addr) : (!ttkernel.noc_addr, i32) -> ()
      // TODO: test %dst_addr of type TTKernel_L1Addr?
      return
    }

    // CHECK-LABEL: func @noc_async_read_barrier
    func.func @noc_async_read_barrier() -> () {
      // CHECK: emitc.call_opaque "noc_async_read_barrier"()
      "ttkernel.noc_async_read_barrier"() : () -> ()
      return
    }

    // CHECK-LABEL: func @noc_async_write
    func.func @noc_async_write() -> () {
      // CHECK: %[[SRC_ADDR:.*]] = "emitc.constant"
      %src_addr = arith.constant 303104 : i32
      // CHECK: %[[DST_ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %temp = arith.constant 262400 : i32
      %dst_addr = "ttkernel.get_noc_addr"(%temp) : (i32) -> (!ttkernel.noc_addr) // a dummy noc addr
      // CHECK: %[[SIZE:.*]] = "emitc.constant"
      %size = arith.constant 2048 : i32
      // CHECK: emitc.call_opaque "noc_async_write"(%[[SRC_ADDR]], %[[DST_ADDR]], %[[SIZE]])
      "ttkernel.noc_async_write"(%src_addr, %dst_addr, %size) : (i32, !ttkernel.noc_addr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_async_write_barrier
    func.func @noc_async_write_barrier() -> () {
      // CHECK: emitc.call_opaque "noc_async_write_barrier"()
      "ttkernel.noc_async_write_barrier"() : () -> ()
      return
    }

    // CHECK-LABEL: func @get_semaphore
    func.func @get_semaphore() -> () {
      // CHECK: %[[SEMAPHORE_ID:.*]] = "emitc.constant"
      %semaphore_id = arith.constant 2 : i32
      // CHECK: emitc.call_opaque "get_semaphore"(%[[SEMAPHORE_ID]])
      "ttkernel.get_semaphore"(%semaphore_id) : (i32) -> (!ttkernel.l1_addr)
      return
    }

    // CHECK-LABEL: func @noc_semaphore_inc
    func.func @noc_semaphore_inc() -> () {
      // CHECK: %[[ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %temp = arith.constant 262400 : i32
      %addr = "ttkernel.get_noc_addr"(%temp) : (i32) -> (!ttkernel.noc_addr) // a dummy noc addr
      // CHECK: %[[INCR:.*]] = "emitc.constant"
      %incr = arith.constant 1 : i32
      // CHECK: %[[NOC_ID:.*]] = "emitc.constant"
      %noc_id = arith.constant 3 : i32
      // CHECK: emitc.call_opaque "noc_semaphore_inc"(%[[ADDR]], %[[INCR]], %[[NOC_ID]])
      "ttkernel.noc_semaphore_inc"(%addr, %incr, %noc_id) : (!ttkernel.noc_addr, i32, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_semaphore_set
    func.func @noc_semaphore_set() -> () {
      // CHECK: %[[ADDR:.*]] = emitc.call_opaque "reinterpret_cast
      %temp = arith.constant 262400 : i32
      %addr = "ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>"(%temp) : (i32) -> (!ttkernel.l1_addr_ptr) // a dummy l1 addr ptr
      // CHECK: %[[VAL:.*]] = "emitc.constant"
      %val = arith.constant 123 : i32
      // CHECK: emitc.call_opaque "noc_semaphore_set"(%[[ADDR]], %[[VAL]])
      "ttkernel.noc_semaphore_set"(%addr, %val) : (!ttkernel.l1_addr_ptr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_semaphore_wait
    func.func @noc_semaphore_wait() -> () {
      // CHECK: %[[ADDR:.*]] = emitc.call_opaque "reinterpret_cast
      %temp = arith.constant 262400 : i32
      %addr = "ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>"(%temp) : (i32) -> (!ttkernel.l1_addr_ptr) // a dummy l1 addr ptr
      // CHECK: %[[VAL:.*]] = "emitc.constant"
      %val = arith.constant 123 : i32
      // CHECK: emitc.call_opaque "noc_semaphore_wait"(%[[ADDR]], %[[VAL]])
      "ttkernel.noc_semaphore_wait"(%addr, %val) : (!ttkernel.l1_addr_ptr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_semaphore_wait_min
    func.func @noc_semaphore_wait_min() -> () {
      // CHECK: %[[ADDR:.*]] = emitc.call_opaque "reinterpret_cast
      %temp = arith.constant 262400 : i32
      %addr = "ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>"(%temp) : (i32) -> (!ttkernel.l1_addr_ptr) // a dummy l1 addr ptr
      // CHECK: %[[VAL:.*]] = "emitc.constant"
      %val = arith.constant 123 : i32
      // CHECK: emitc.call_opaque "noc_semaphore_wait_min"(%[[ADDR]], %[[VAL]])
      "ttkernel.noc_semaphore_wait_min"(%addr, %val) : (!ttkernel.l1_addr_ptr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_semaphore_set_multicast
    func.func @noc_semaphore_set_multicast() -> () {
      // CHECK: %[[SRC_ADDR:.*]] = emitc.call_opaque "get_semaphore"
      %temp1 = arith.constant 2 : i32
      %src_addr = "ttkernel.get_semaphore"(%temp1) : (i32) -> (!ttkernel.l1_addr) // a dummy l1 addr
      // CHECK: %[[DST_MCAST_ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %temp2 = arith.constant 303104 : i32
      %dst_mcast_addr = "ttkernel.get_noc_addr"(%temp2) : (i32) -> (!ttkernel.noc_addr) // a dummy noc addr TODO use mcast getter
      // CHECK: %[[NUM_DSTS:.*]] = "emitc.constant"
      %num_dsts = arith.constant 8 : i32
      // TODO(#2229): emitc lowering ignores 'linked' and 'multicast_path_reserve'
      // CHECK: emitc.call_opaque "noc_semaphore_set_multicast"(%[[SRC_ADDR]], %[[DST_MCAST_ADDR]], %[[NUM_DSTS]])
      "ttkernel.noc_semaphore_set_multicast"(%src_addr, %dst_mcast_addr, %num_dsts) <{
          linked = false, multicast_path_reserve = true
        }> : (!ttkernel.l1_addr, !ttkernel.noc_addr, i32) -> ()
      return
    }

    // CHECK-LABEL: func @noc_semaphore_set_multicast_loopback_src
    func.func @noc_semaphore_set_multicast_loopback_src() -> () {
      // CHECK: %[[SRC_ADDR:.*]] = emitc.call_opaque "get_semaphore"
      %temp1 = arith.constant 2 : i32
      %src_addr = "ttkernel.get_semaphore"(%temp1) : (i32) -> (!ttkernel.l1_addr) // a dummy l1 addr
      // CHECK: %[[DST_MCAST_ADDR:.*]] = emitc.call_opaque "get_noc_addr"
      %temp2 = arith.constant 303104 : i32
      %dst_mcast_addr = "ttkernel.get_noc_addr"(%temp2) : (i32) -> (!ttkernel.noc_addr) // a dummy noc addr TODO use mcast getter
      // CHECK: %[[NUM_DSTS:.*]] = "emitc.constant"
      %num_dsts = arith.constant 8 : i32
      // TODO(#2229): emitc lowering ignores 'linked' and 'multicast_path_reserve'
      // CHECK: emitc.call_opaque "noc_semaphore_set_multicast_loopback_src"(%[[SRC_ADDR]], %[[DST_MCAST_ADDR]], %[[NUM_DSTS]])
      "ttkernel.noc_semaphore_set_multicast_loopback_src"(%src_addr, %dst_mcast_addr, %num_dsts) <{
          linked = false, multicast_path_reserve = true
        }> : (!ttkernel.l1_addr, !ttkernel.noc_addr, i32) -> ()
      return
    }

    // TODO(#2230): without a use like the commented out part below the noc table def will be simply elided; however,
    // the current lowering doesn't seem to work with nested modules, so just testing the elision for now:

    // CHECK-LABEL: func @noc_transactions_table
    func.func @noc_transactions_table() -> () {
      // CHECK: %{{.+}} = "emitc.constant"
      %unused = arith.constant 0 : i32
      // CHECK-NOT: emitc
      %table = "ttkernel.noc_transactions_table"() <{
          entries = array<i32: 99360, 99104, 32, 1179666, 99392, 99168, 32, 1179666>
        }> : () -> memref<2x4xi32>
      // %i = arith.constant 1 : index
      // %j = arith.constant 0 : index
      // %entry = memref.load %table[%i, %j] : memref<2x4xi32>
      // CHECK-NEXT: return
      return
    }

  } // module

} // module
