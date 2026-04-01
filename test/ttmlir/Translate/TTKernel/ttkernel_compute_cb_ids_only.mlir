// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: ttmlir-translate --ttkernel-to-cpp -o %t.cpp %t
// RUN: FileCheck %s --input-file=%t.cpp

// CHECK-NOT: #include "experimental/circular_buffer.h"
// CHECK: void kernel_main()
func.func @ttkernel_compute_cb_ids_only() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
  %icb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
  %ocb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
  // CHECK: compute_kernel_hw_startup(get_compile_time_arg_val(0), get_compile_time_arg_val(1))
  "ttkernel.compute_kernel_hw_startup"(%icb, %ocb) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, !ttkernel.cb<8, !ttcore.tile<32x32, f32>>) -> ()
  func.return
}
