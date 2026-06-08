// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: ttmlir-translate --ttkernel-to-cpp -o %t.cpp %t
// RUN: FileCheck %s --input-file=%t.cpp

// CHECK: #include "api/compute/bcast.h"
// CHECK: void kernel_main()
func.func @binary_bcast_test() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    %in0_cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
    %in1_cb = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    // CHECK: init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(
    ttkernel.binary_bcast_init(%in0_cb, %in1_cb, %in0_cb, <mul>, <col>) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, !ttkernel.cb<8, !ttcore.tile<32x32, f32>>, !ttkernel.cb<8, !ttcore.tile<32x32, f32>>) -> ()
    // CHECK: mul_tiles_bcast<BroadcastType::COL>(
    ttkernel.binary_bcast(%in0_cb, %in1_cb, %c0, %c0, %c0, <mul>, <col>) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, !ttkernel.cb<8, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
    func.return
}
