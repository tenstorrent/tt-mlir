// RUN: ttmlir-translate --ttkernel-to-cpp-noc %s | FileCheck %s

// CHECK: #include "dataflow_api.h"
// CHECK: void kernel_main
func.func @ttkernel_noc() -> () {
    // CHECK: int32_t [[B0:.*]] = 262432
    %c262432_i32 = arith.constant 262432 : i32
    // CHECK: int32_t [[B1:.*]] = 262208
    %c262208_i32 = arith.constant 262208 : i32
    // CHECK: int32_t [[C0:.*]] = 32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: int32_t [[C1:.*]] = 262400
    %c262400_i32 = arith.constant 262400 : i32
    // CHECK: int32_t [[A0:.*]] = 0
    %c0_i32 = arith.constant 0 : i32
    // CHECK: int32_t [[A1:.*]] = 262144;
    %c262144_i32 = arith.constant 262144 : i32
    // CHECK: int64_t [[NOCADDR0:.*]] = get_noc_addr([[A0]], [[A0]], [[A1]])
    %3 = "ttkernel.get_noc_addr_xy"(%c0_i32, %c0_i32, %c262144_i32) : (i32, i32, i32) -> !ttkernel.noc_addr
    // CHECK: noc_async_read([[NOCADDR0]], [[C1]], [[C0]])
    "ttkernel.noc_async_read"(%3, %c262400_i32, %c32_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    // CHECK: int64_t [[NOCADDR1:.*]] = get_noc_addr([[A0]], [[A0]], [[B1]])
    %4 = "ttkernel.get_noc_addr_xy"(%c0_i32, %c0_i32, %c262208_i32) : (i32, i32, i32) -> !ttkernel.noc_addr
    // CHECK: noc_async_read([[NOCADDR1]], [[B0]], [[C0]])
    "ttkernel.noc_async_read"(%4, %c262432_i32, %c32_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    // CHECK: noc_async_read_barrier
    "ttkernel.noc_async_read_barrier"() : () -> ()
    // CHECK: return
    "ttkernel.return"() : () -> ()
}
