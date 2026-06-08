// RUN: ttmlir-opt -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>

// CHECK-LABEL: func.func @test_tilize_uninit
func.func @test_tilize_uninit() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>} {
  %tilized_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<16, !ttcore.tile<32x32, f32>>
  %untilized_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<16384, f32>
  ttkernel.tilize_uninit(%untilized_cb, %tilized_cb) : (!ttkernel.cb<16384, f32>, !ttkernel.cb<16, !ttcore.tile<32x32, f32>>) -> ()
  // CHECK: ttkernel.tilize_uninit(%{{.*}}, %{{.*}}) : (!ttkernel.cb<16384, f32>, !ttkernel.cb<16, !ttcore.tile<32x32, f32>>) -> ()
  return
}

// CHECK-LABEL: func.func @test_untilize_uninit
func.func @test_untilize_uninit() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>} {
  %tilized_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<16, !ttcore.tile<32x32, f32>>
  ttkernel.untilize_uninit(%tilized_cb) : (!ttkernel.cb<16, !ttcore.tile<32x32, f32>>) -> ()
  // CHECK: ttkernel.untilize_uninit(%{{.*}}) : (!ttkernel.cb<16, !ttcore.tile<32x32, f32>>) -> ()
  return
}

// CHECK-LABEL: func.func @test_add_binary_tile_init
func.func @test_add_binary_tile_init() -> () {
  ttkernel.add_binary_tile_init() : () -> ()
  // CHECK: ttkernel.add_binary_tile_init() : () -> ()
  return
}

// CHECK-LABEL: func.func @test_add_binary_tile
func.func @test_add_binary_tile() -> () {
  %c0 = arith.constant 0 : index
  ttkernel.add_binary_tile(%c0, %c0, %c0) : (index, index, index) -> ()
  // CHECK: ttkernel.add_binary_tile(%{{.*}}, %{{.*}}, %{{.*}}) : (index, index, index) -> ()
  return
}

// CHECK-LABEL: func.func @test_mul_binary_tile_init
func.func @test_mul_binary_tile_init() -> () {
  ttkernel.mul_binary_tile_init() : () -> ()
  // CHECK: ttkernel.mul_binary_tile_init() : () -> ()
  return
}

// CHECK-LABEL: func.func @test_mul_binary_tile
func.func @test_mul_binary_tile() -> () {
  %c0 = arith.constant 0 : index
  ttkernel.mul_binary_tile(%c0, %c0, %c0) : (index, index, index) -> ()
  // CHECK: ttkernel.mul_binary_tile(%{{.*}}, %{{.*}}, %{{.*}}) : (index, index, index) -> ()
  return
}

// CHECK-LABEL: func.func @test_sub_binary_tile_init
func.func @test_sub_binary_tile_init() -> () {
  ttkernel.sub_binary_tile_init() : () -> ()
  // CHECK: ttkernel.sub_binary_tile_init() : () -> ()
  return
}

// CHECK-LABEL: func.func @test_sub_binary_tile
func.func @test_sub_binary_tile() -> () {
  %c0 = arith.constant 0 : index
  ttkernel.sub_binary_tile(%c0, %c0, %c0) : (index, index, index) -> ()
  // CHECK: ttkernel.sub_binary_tile(%{{.*}}, %{{.*}}, %{{.*}}) : (index, index, index) -> ()
  return
}

// CHECK-LABEL: func.func @test_pack_tile_block
func.func @test_pack_tile_block() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<16, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  ttkernel.pack_tile_block(%c0, %cb, %c4) : (index, !ttkernel.cb<16, !ttcore.tile<32x32, f32>>, index) -> ()
  // CHECK: ttkernel.pack_tile_block(%{{.*}}, %{{.*}}, %{{.*}}) : (index, !ttkernel.cb<16, !ttcore.tile<32x32, f32>>, index) -> ()
  return
}

// CHECK-LABEL: func.func @test_copy_block_matmul_partials
func.func @test_copy_block_matmul_partials() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>} {
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<16, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  ttkernel.copy_block_matmul_partials(%cb, %c0, %c0, %c4) : (!ttkernel.cb<16, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  // CHECK: ttkernel.copy_block_matmul_partials(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!ttkernel.cb<16, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  return
}

// CHECK-LABEL: func.func @test_copy_dest_values_init
func.func @test_copy_dest_values_init() -> () {
  // CHECK: ttkernel.copy_dest_values_init() : () -> ()
  ttkernel.copy_dest_values_init() : () -> ()
  return
}

// CHECK-LABEL: func.func @test_copy_dest_values
func.func @test_copy_dest_values() -> () {
  %c0 = arith.constant 0 : index
  ttkernel.copy_dest_values(%c0, %c0) : (index, index) -> ()
  // CHECK: ttkernel.copy_dest_values(%{{.*}}, %{{.*}}) : (index, index) -> ()
  return
}

// CHECK-LABEL: func.func @test_noc_trid_and_noc_nonconst
// CHECK-SAME: (%[[TRID:.*]]: i32, %[[NOC:.*]]: i8)
func.func @test_noc_trid_and_noc_nonconst(%trid: i32, %noc: i8) {
  // CHECK: ttkernel.noc_async_read_set_trid(%[[TRID]], %[[NOC]]) : (i32, i8) -> ()
  ttkernel.noc_async_read_set_trid(%trid, %noc) : (i32, i8) -> ()
  return
}

//===----------------------------------------------------------------------===//
// TensorAccessorArgsOp assembly format round-trip tests.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tensor_accessor_args_literal_offsets
func.func @test_tensor_accessor_args_literal_offsets() -> !ttkernel.TensorAccessorArgs {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  %c0 = arith.constant 0 : i32
  // CHECK: %[[C5:.*]] = arith.constant 5 : i32
  %c5 = arith.constant 5 : i32
  // Test: literal offsets without chaining.
  // CHECK: ttkernel.TensorAccessorArgs(%[[C0]], %[[C5]])
  %args = ttkernel.TensorAccessorArgs(%c0, %c5)
  return %args : !ttkernel.TensorAccessorArgs
}

// CHECK-LABEL: func.func @test_tensor_accessor_args_chaining
func.func @test_tensor_accessor_args_chaining() -> !ttkernel.TensorAccessorArgs {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  %c0 = arith.constant 0 : i32
  // Test: chaining via prev_args.
  // CHECK: %[[ARGS1:.*]] = ttkernel.TensorAccessorArgs(%[[C0]], %[[C0]])
  %args1 = ttkernel.TensorAccessorArgs(%c0, %c0)
  // CHECK: ttkernel.TensorAccessorArgs(prev = %[[ARGS1]])
  %args2 = ttkernel.TensorAccessorArgs(prev = %args1)
  return %args2 : !ttkernel.TensorAccessorArgs
}

// CHECK-LABEL: func.func @test_tensor_accessor_args_cta_expr
func.func @test_tensor_accessor_args_cta_expr() -> !ttkernel.TensorAccessorArgs {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  %c0 = arith.constant 0 : i32
  // Test: with cta_expr attribute.
  // CHECK: ttkernel.TensorAccessorArgs(%[[C0]], %[[C0]]) cta_expr = "get_offset()"
  %args = ttkernel.TensorAccessorArgs(%c0, %c0) cta_expr = "get_offset()"
  return %args : !ttkernel.TensorAccessorArgs
}

// CHECK-LABEL: func.func @test_tensor_accessor_args_crta_expr
func.func @test_tensor_accessor_args_crta_expr() -> !ttkernel.TensorAccessorArgs {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  %c0 = arith.constant 0 : i32
  // Test: with crta_expr attribute.
  // CHECK: ttkernel.TensorAccessorArgs(%[[C0]], %[[C0]]) crta_expr = "compute_crta()"
  %args = ttkernel.TensorAccessorArgs(%c0, %c0) crta_expr = "compute_crta()"
  return %args : !ttkernel.TensorAccessorArgs
}

// CHECK-LABEL: func.func @test_tensor_accessor_args_both_exprs
func.func @test_tensor_accessor_args_both_exprs() -> !ttkernel.TensorAccessorArgs {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  %c0 = arith.constant 0 : i32
  // Test: with both cta_expr and crta_expr.
  // CHECK: ttkernel.TensorAccessorArgs(%[[C0]], %[[C0]]) cta_expr = "cta_func()" crta_expr = "crta_func()"
  %args = ttkernel.TensorAccessorArgs(%c0, %c0) cta_expr = "cta_func()" crta_expr = "crta_func()"
  return %args : !ttkernel.TensorAccessorArgs
}

// CHECK-LABEL: func.func @test_tensor_accessor_args_chaining_with_crta_override
func.func @test_tensor_accessor_args_chaining_with_crta_override() -> !ttkernel.TensorAccessorArgs {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  %c0 = arith.constant 0 : i32
  // Test: chaining with crta_expr override.
  // CHECK: %[[ARGS1:.*]] = ttkernel.TensorAccessorArgs(%[[C0]], %[[C0]])
  %args1 = ttkernel.TensorAccessorArgs(%c0, %c0)
  // CHECK: ttkernel.TensorAccessorArgs(prev = %[[ARGS1]]) crta_expr = "0"
  %args2 = ttkernel.TensorAccessorArgs(prev = %args1) crta_expr = "0"
  return %args2 : !ttkernel.TensorAccessorArgs
}

// CHECK-LABEL: func.func @test_noc_trid_with_implicit_noc
// CHECK-SAME: (%[[TRID:.*]]: i32)
func.func @test_noc_trid_with_implicit_noc(%trid: i32) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: ttkernel.noc_async_read_one_packet_with_state_with_trid(%[[C0]], %[[C0]], %[[C0]], %[[TRID]]) : (i32, i32, i32, i32) -> ()
  %c0 = arith.constant 0 : i32
  "ttkernel.noc_async_read_one_packet_with_state_with_trid"(%c0, %c0, %c0, %trid)
      : (i32, i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: func.func @test_remote_sram_write_u32_sram_addr
// CHECK-SAME: (%[[SRC:.*]]: !ttkernel.l1_addr, %[[DST:.*]]: !ttkernel.noc_addr)
func.func @test_remote_sram_write_u32_sram_addr(%src: !ttkernel.l1_addr, %dst: !ttkernel.noc_addr) {
  // CHECK: ttkernel.remote_sram_write_u32(%[[SRC]], %[[DST]]) : (!ttkernel.l1_addr, !ttkernel.noc_addr) -> ()
  ttkernel.remote_sram_write_u32(%src, %dst) : (!ttkernel.l1_addr, !ttkernel.noc_addr) -> ()
  return
}

// CHECK-LABEL: func.func @test_remote_sram_write_u32_computed_sram_addr
// CHECK-SAME: (%[[BASE:.*]]: i32, %[[DST:.*]]: !ttkernel.noc_addr)
func.func @test_remote_sram_write_u32_computed_sram_addr(%base: i32, %dst: !ttkernel.noc_addr) {
  // CHECK: %[[OFFSET:.*]] = arith.constant 16 : i32
  %offset = arith.constant 16 : i32
  // CHECK: %[[SRC:.*]] = arith.addi %[[BASE]], %[[OFFSET]] : i32
  %src = arith.addi %base, %offset : i32
  // CHECK: ttkernel.remote_sram_write_u32(%[[SRC]], %[[DST]]) : (i32, !ttkernel.noc_addr) -> ()
  ttkernel.remote_sram_write_u32(%src, %dst) : (i32, !ttkernel.noc_addr) -> ()
  return
}

// CHECK-LABEL: func.func @test_noc_inline_dw_write
// CHECK-SAME: (%[[DST:.*]]: !ttkernel.noc_addr, %[[VAL:.*]]: i32, %[[BE:.*]]: i8, %[[NOC:.*]]: i8)
func.func @test_noc_inline_dw_write(%dst: !ttkernel.noc_addr, %val: i32, %be: i8, %noc: i8) {
  // CHECK: ttkernel.noc_inline_dw_write(%[[DST]], %[[VAL]], %[[BE]], %[[NOC]]) : (!ttkernel.noc_addr, i32, i8, i8) -> ()
  ttkernel.noc_inline_dw_write(%dst, %val, %be, %noc) : (!ttkernel.noc_addr, i32, i8, i8) -> ()
  return
}

// CHECK-LABEL: func.func @test_remote_sram_write_u32_local_semaphore
// CHECK-SAME: (%[[SRC:.*]]: !ttkernel.local_semaphore, %[[DST:.*]]: !ttkernel.noc_addr, %[[NOC:.*]]: i8)
func.func @test_remote_sram_write_u32_local_semaphore(%src: !ttkernel.local_semaphore, %dst: !ttkernel.noc_addr, %noc: i8) {
  // CHECK: ttkernel.remote_sram_write_u32(%[[SRC]], %[[DST]], %[[NOC]]) : (!ttkernel.local_semaphore, !ttkernel.noc_addr, i8) -> ()
  ttkernel.remote_sram_write_u32(%src, %dst, %noc) : (!ttkernel.local_semaphore, !ttkernel.noc_addr, i8) -> ()
  return
}

// CHECK-LABEL: func.func @test_remote_mailbox_protocol_ops
func.func @test_remote_mailbox_protocol_ops(%mailbox: !ttkernel.l1_addr) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : index
  %value = arith.constant 64 : i32
  // CHECK: %[[SEM:.*]] = ttkernel.get_semaphore
  %sem = ttkernel.get_semaphore(%zero) : (i32) -> !ttkernel.local_semaphore
  // CHECK: %[[PTR:.*]] = ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>
  %sem_ptr = "ttkernel.reinterpret_cast<tt_l1_ptr uint32_t*>"(%sem) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
  // CHECK: ttkernel.store_to_l1
  ttkernel.store_to_l1(%value, %sem_ptr, %zero) : (i32, !ttkernel.l1_addr_ptr, i32) -> ()
  // CHECK: ttkernel.load_from_l1
  %loaded = ttkernel.load_from_l1(%sem_ptr, %zero) : (!ttkernel.l1_addr_ptr, i32) -> i32
  // CHECK: %[[DST:.*]] = ttkernel.get_noc_addr
  %dst = ttkernel.get_noc_addr(%one, %one, %mailbox) : (index, index, !ttkernel.l1_addr) -> !ttkernel.noc_addr
  // CHECK: ttkernel.remote_sram_write_u32(%[[SEM]], %[[DST]])
  ttkernel.remote_sram_write_u32(%sem, %dst) : (!ttkernel.local_semaphore, !ttkernel.noc_addr) -> ()
  %sink = arith.addi %loaded, %zero : i32
  return
}

//===----------------------------------------------------------------------===//
// Binary broadcast operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_binary_bcast
func.func @test_binary_bcast() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>} {
  %in0_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
  %in1_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  ttkernel.binary_bcast_init(%in0_cb, %in1_cb, %in0_cb, <mul>, <col>) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, !ttkernel.cb<8, !ttcore.tile<32x32, f32>>, !ttkernel.cb<8, !ttcore.tile<32x32, f32>>) -> ()
  // CHECK: ttkernel.binary_bcast_init(%{{.*}}, %{{.*}}, %{{.*}}, <mul>, <col>)
  ttkernel.binary_bcast(%in0_cb, %in1_cb, %c0, %c0, %c0, <mul>, <col>) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, !ttkernel.cb<8, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
  // CHECK: ttkernel.binary_bcast(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, <mul>, <col>)
  return
}
