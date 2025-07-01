// RUN: ttmlir-opt %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>

// CHECK-LABEL: func.func @test_tilize_uninit
func.func @test_tilize_uninit() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>} {
  %tilized_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>
  %untilized_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !ttkernel.cb<memref<128x128xf32, #l1_>>
  "ttkernel.tilize_uninit"(%untilized_cb, %tilized_cb) : (!ttkernel.cb<memref<128x128xf32, #l1_>>, !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>) -> ()
  return
}

// CHECK-LABEL: func.func @test_untilize_uninit
func.func @test_untilize_uninit() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>} {
  %tilized_cb = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>
  "ttkernel.untilize_uninit"(%tilized_cb) : (!ttkernel.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>) -> ()
  return
}
