// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: ttmlir-translate --ttkernel-to-cpp -o %t.cpp %t
// RUN: FileCheck %s --input-file=%t.cpp

// CHECK: #include "api/numeric/bfloat16.h"
// CHECK: void kernel_main()
func.func @bfloat16_greater_test() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = scalar, operand_index = 0>, <arg_type = scalar, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    %raw0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> ui32
    %raw1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> ui32
    %a = ttkernel.bitcast %raw0 : ui32 to ui16
    %b = ttkernel.bitcast %raw1 : ui32 to ui16
    // CHECK: bool [[V1:v[0-9]+]] = bfloat16_greater(
    %0 = ttkernel.bfloat16_greater(%a, %b) : (ui16, ui16) -> i1
    func.return
}

// CHECK: #include "api/numeric/float32.h"
// CHECK: void kernel_main()
func.func @float32_greater_test() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = scalar, operand_index = 0>, <arg_type = scalar, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    %arg0 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> ui32
    %arg1 = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> ui32
    // CHECK: bool [[V1:v[0-9]+]] = float32_greater(
    %0 = ttkernel.float32_greater(%arg0, %arg1) : (ui32, ui32) -> i1
    func.return
}
