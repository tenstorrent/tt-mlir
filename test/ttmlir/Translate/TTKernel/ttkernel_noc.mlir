// RUN: ttmlir-translate --nockernel-to-cpp %s

func.func @ttkernel_noc() -> () {
    %c262432_i32 = arith.constant 262432 : i32
    %c262208_i32 = arith.constant 262208 : i32
    %c32_i32 = arith.constant 32 : i32
    %c262400_i32 = arith.constant 262400 : i32
    %c0_i32 = arith.constant 0 : i32
    %c262144_i32 = arith.constant 262144 : i32
    %3 = "ttkernel.get_noc_addr_xy"(%c0_i32, %c0_i32, %c262144_i32) : (i32, i32, i32) -> !ttkernel.noc_addr
    "ttkernel.noc_async_read"(%3, %c262400_i32, %c32_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    %4 = "ttkernel.get_noc_addr_xy"(%c0_i32, %c0_i32, %c262208_i32) : (i32, i32, i32) -> !ttkernel.noc_addr
    "ttkernel.noc_async_read"(%4, %c262432_i32, %c32_i32) : (!ttkernel.noc_addr, i32, i32) -> ()
    "ttkernel.noc_async_read_barrier"() : () -> ()
    "ttkernel.return"() : () -> ()
}
