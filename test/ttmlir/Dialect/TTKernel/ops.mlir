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
