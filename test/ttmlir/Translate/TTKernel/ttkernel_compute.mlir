// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: ttmlir-translate --ttkernel-to-cpp -o %t.cpp %t
// RUN: FileCheck %s --input-file=%t.cpp

#l1_ = #ttcore.memory_space<l1>

// CHECK: #include "experimental/circular_buffer.h"
// CHECK: void kernel_main()
func.func @ttkernel_compute() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    // CHECK: int32_t v1 = 4
    %c4_i32 = arith.constant 4 : i32
    %arg1 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
    %arg2 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<8192, f32>
    // CHECK: experimental::CircularBuffer [[CB0:cb_[a-zA-Z0-9_]+]](get_compile_time_arg_val(0));
    // CHECK: experimental::CircularBuffer [[CB1:cb_[a-zA-Z0-9_]+]](get_compile_time_arg_val(1));
    // CHECK: untilize_init(get_compile_time_arg_val(0))
    ttkernel.untilize_init(%arg1) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>) -> ()
    // CHECK: untilize_block(get_compile_time_arg_val(0), v1, get_compile_time_arg_val(1))
    ttkernel.untilize_block(%arg1, %c4_i32, %arg2) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, i32, !ttkernel.cb<8192, f32>) -> ()
    // CHECK: [[CB0]].pop_front(v1)
    ttkernel.cb_pop_front(%arg1, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, i32) -> ()
    // CHECK: [[CB1]].push_back(v1)
    ttkernel.cb_push_back(%arg2, %c4_i32) : (!ttkernel.cb<8192, f32>, i32) -> ()
    // CHECK: untilize_block(get_compile_time_arg_val(0), v1, get_compile_time_arg_val(1))
    ttkernel.untilize_block(%arg1, %c4_i32, %arg2) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, i32, !ttkernel.cb<8192, f32>) -> ()
    // CHECK: [[CB0]].pop_front(v1)
    ttkernel.cb_pop_front(%arg1, %c4_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, i32) -> ()
    // CHECK: [[CB1]].push_back(v1)
    ttkernel.cb_push_back(%arg2, %c4_i32) : (!ttkernel.cb<8192, f32>, i32) -> ()
    // CHECK: return
    func.return
}
