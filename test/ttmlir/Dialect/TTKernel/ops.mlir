// RUN: ttmlir-opt -o %t %s
// RUN: ttmlir-opt %s -o - | ttmlir-opt -o /dev/null
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>

// CHECK-LABEL: func.func @test_compute_kernel_hw_startup_unary
func.func @test_compute_kernel_hw_startup_unary() {
  %icb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1024, si32>
  %ocb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, si32>>
  ttkernel.compute_kernel_hw_startup(%icb, %ocb) : (!ttkernel.cb<1024, si32>, !ttkernel.cb<1, !ttcore.tile<32x32, si32>>) -> ()
  // CHECK: ttkernel.compute_kernel_hw_startup(%{{.*}}, %{{.*}}) : (!ttkernel.cb<1024, si32>, !ttkernel.cb<1, !ttcore.tile<32x32, si32>>) -> ()
  return
}

// CHECK-LABEL: func.func @test_compute_kernel_hw_startup_binary
func.func @test_compute_kernel_hw_startup_binary() {
  %icb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1024, si32>
  %icb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<1024, si32>
  %ocb = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, si32>>
  ttkernel.compute_kernel_hw_startup(%icb0, %icb1, %ocb) : (!ttkernel.cb<1024, si32>, !ttkernel.cb<1024, si32>, !ttkernel.cb<1, !ttcore.tile<32x32, si32>>) -> ()
  // CHECK: ttkernel.compute_kernel_hw_startup(%{{.*}}, %{{.*}}, %{{.*}}) : (!ttkernel.cb<1024, si32>, !ttkernel.cb<1024, si32>, !ttkernel.cb<1, !ttcore.tile<32x32, si32>>) -> ()
  return
}

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
  ttkernel.copy_dest_values(%c0, %c0, <f32>) : (index, index) -> ()
  // CHECK: ttkernel.copy_dest_values(%{{.*}}, %{{.*}}, <f32>) : (index, index) -> ()
  return
}

// CHECK-LABEL: func.func @test_noc_trid_and_noc_nonconst
// CHECK-SAME: (%[[TRID:.*]]: i32, %[[NOC:.*]]: i8)
func.func @test_noc_trid_and_noc_nonconst(%trid: i32, %noc: i8) {
  // CHECK: ttkernel.noc_async_read_barrier_with_trid(%[[TRID]], %[[NOC]]) : (i32, i8) -> ()
  ttkernel.noc_async_read_barrier_with_trid(%trid, %noc) : (i32, i8) -> ()
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

// CHECK-LABEL: func.func @test_noc_write_one_packet_with_trid_implicit_noc
// CHECK-SAME: (%[[TRID:.*]]: i32, %[[SRC:.*]]: i32, %[[X:.*]]: index, %[[Y:.*]]: index, %[[DST:.*]]: i32, %[[SIZE:.*]]: i32)
func.func @test_noc_write_one_packet_with_trid_implicit_noc(%trid: i32, %src: i32, %x: index, %y: index, %dst: i32, %size: i32) {
  // CHECK: ttkernel.noc_async_write_one_packet_with_trid(%[[SRC]], core[%[[X]], %[[Y]]], %[[DST]], %[[SIZE]], %[[TRID]]) : (i32, index, index, i32, i32, i32) -> ()
  ttkernel.noc_async_write_one_packet_with_trid(%src, core[%x, %y], %dst, %size, %trid)
      : (i32, index, index, i32, i32, i32) -> ()
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
// CHECK-SAME: (%[[X:.*]]: index, %[[Y:.*]]: index, %[[DST:.*]]: i32, %[[VAL:.*]]: i32, %[[BE:.*]]: i8, %[[NOC:.*]]: i8)
func.func @test_noc_inline_dw_write(%x: index, %y: index, %dst: i32, %val: i32, %be: i8, %noc: i8) {
  // CHECK: ttkernel.noc_inline_dw_write(core[%[[X]], %[[Y]]], %[[DST]], %[[VAL]], %[[BE]], noc %[[NOC]]) : (index, index, i32, i32, i8, i8) -> ()
  ttkernel.noc_inline_dw_write(core[%x, %y], %dst, %val, %be, noc %noc) : (index, index, i32, i32, i8, i8) -> ()
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
  // CHECK: %[[PTR:.*]] = ttkernel.reinterpret_cast
  %sem_ptr = ttkernel.reinterpret_cast(%sem) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
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
// Numeric operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_bfloat16_greater
// CHECK-SAME: (%[[A:.*]]: i16, %[[B:.*]]: i16)
func.func @test_bfloat16_greater(%arg0: i16, %arg1: i16) -> () {
  %0 = ttkernel.bfloat16_greater(%arg0, %arg1) : (i16, i16) -> i1
  // CHECK: ttkernel.bfloat16_greater(%[[A]], %[[B]]) : (i16, i16) -> i1
  return
}

// CHECK-LABEL: func.func @test_float32_greater
// CHECK-SAME: (%[[A:.*]]: i32, %[[B:.*]]: i32)
func.func @test_float32_greater(%arg0: i32, %arg1: i32) -> () {
  %0 = ttkernel.float32_greater(%arg0, %arg1) : (i32, i32) -> i1
  // CHECK: ttkernel.float32_greater(%[[A]], %[[B]]) : (i32, i32) -> i1
  return
}
