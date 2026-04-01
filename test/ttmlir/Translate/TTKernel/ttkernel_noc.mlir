// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: ttmlir-translate --ttkernel-to-cpp -o %t.cpp %t
// RUN: FileCheck %s --input-file=%t.cpp

// CHECK: #include "api/dataflow/dataflow_api.h"
// CHECK: #include "experimental/circular_buffer.h"
// CHECK: void kernel_main
func.func @ttkernel_noc_cb() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    // CHECK: int32_t [[C0:.*]] = 32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: size_t [[A0:.*]] = 0
    %c0_idx = arith.constant 0 : index
    // CHECK: int32_t [[A1:.*]] = 262144;
    %c262144_i32 = arith.constant 262144 : i32
    %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
    // CHECK: experimental::CircularBuffer [[CB:cb_ctarg_0]](get_compile_time_arg_val(0));
    %c1_i32 = arith.constant 1 : i32
    // CHECK: [[CB]].reserve_back
    "ttkernel.cb_reserve_back"(%cb, %c1_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, i32) -> ()
    // CHECK: int64_t [[NOCADDR0:.*]] = get_noc_addr([[A0]], [[A0]], [[A1]])
    %3 = ttkernel.get_noc_addr(%c0_idx, %c0_idx, %c262144_i32) : (index, index, i32) -> !ttkernel.noc_addr
    // CHECK: noc_async_read([[NOCADDR0]], [[CB]].get_write_ptr(), [[C0]])
    %wptr = "ttkernel.get_write_ptr"(%cb) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>) -> i32
    ttkernel.noc_async_read(%3, %wptr, %c32_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    // CHECK: noc_async_read_barrier
    ttkernel.noc_async_read_barrier() : () -> ()
    // CHECK: [[CB]].push_back
    "ttkernel.cb_push_back"(%cb, %c1_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, i32) -> ()
    // CHECK: return
    func.return
}

// CHECK: void kernel_main
func.func @ttkernel_noc() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // CHECK: int32_t [[B0:.*]] = 262432
    %c262432_i32 = arith.constant 262432 : i32
    // CHECK: int32_t [[B1:.*]] = 262208
    %c262208_i32 = arith.constant 262208 : i32
    // CHECK: int32_t [[C0:.*]] = 32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: int32_t [[C1:.*]] = 262400
    %c262400_i32 = arith.constant 262400 : i32
    // CHECK: size_t [[A0:.*]] = 0
    %c0_idx = arith.constant 0 : index
    // CHECK: int32_t [[A1:.*]] = 262144;
    %c262144_i32 = arith.constant 262144 : i32
    // CHECK: int64_t [[NOCADDR0:.*]] = get_noc_addr([[A0]], [[A0]], [[A1]])
    %3 = ttkernel.get_noc_addr(%c0_idx, %c0_idx, %c262144_i32) : (index, index, i32) -> !ttkernel.noc_addr
    // CHECK: noc_async_read([[NOCADDR0]], [[C1]], [[C0]])
    ttkernel.noc_async_read(%3, %c262400_i32, %c32_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    // CHECK: int64_t [[NOCADDR1:.*]] = get_noc_addr([[A0]], [[A0]], [[B1]])
    %4 = ttkernel.get_noc_addr(%c0_idx, %c0_idx, %c262208_i32) : (index, index, i32) -> !ttkernel.noc_addr
    // CHECK: noc_async_read([[NOCADDR1]], [[B0]], [[C0]])
    ttkernel.noc_async_read(%4, %c262432_i32, %c32_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    // CHECK: noc_async_read_barrier
    ttkernel.noc_async_read_barrier() : () -> ()
    // CHECK: return
    func.return
}
