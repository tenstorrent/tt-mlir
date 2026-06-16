// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: ttmlir-translate --ttkernel-to-cpp -o %t.cpp %t
// RUN: FileCheck %s --input-file=%t.cpp

// CHECK: #include "api/core_local_mem.h"
// CHECK: #include "api/dataflow/circular_buffer.h"
// CHECK: #include "api/dataflow/dataflow_api.h"
// CHECK: #include "api/dataflow/endpoints.h"
// CHECK: #include "api/dataflow/noc.h"
// CHECK: #include "api/dataflow/noc_semaphore.h"
// SEMDOWN: #include "api/dataflow/noc_semaphore.h"
// CHECK: void kernel_main
func.func @ttkernel_noc_cb() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    // CHECK: UnicastEndpoint [[EP:unicast_ep]]
    // CHECK: Noc [[NOC:noc]]
    // CHECK: int32_t [[C0:.*]] = 32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: size_t [[A0:.*]] = 0
    %c0_idx = arith.constant 0 : index
    %c1_i8 = arith.constant 1 : i8
    // CHECK: int32_t [[A1:.*]] = 262144;
    %c262144_i32 = arith.constant 262144 : i32
    %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
    // CHECK: CircularBuffer [[CB:cb_ctarg_0]](get_compile_time_arg_val(0));
    %c1_i32 = arith.constant 1 : i32
    // CHECK: [[CB]].reserve_back
    "ttkernel.cb_reserve_back"(%cb, %c1_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, i32) -> ()
    %wptr = "ttkernel.get_write_ptr"(%cb) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>) -> i32
    // CHECK: [[NOC]].async_read([[EP]], CoreLocalMem<uint32_t>([[CB]].get_write_ptr()), [[C0]]
    ttkernel.noc_async_read core[%c0_idx, %c0_idx], %c262144_i32, %wptr, %c32_i32 : (index, index, i32, i32, i32) -> ()
    // CHECK: [[NOC]].async_read_barrier()
    ttkernel.noc_async_read_barrier() : () -> ()
    // CHECK: [[CB]].push_back
    "ttkernel.cb_push_back"(%cb, %c1_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, i32) -> ()
    // CHECK: return
    func.return
}

// CHECK: void kernel_main
func.func @ttkernel_noc() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // CHECK: UnicastEndpoint [[EP1:unicast_ep]]
    // CHECK: Noc [[NOC1:noc]]
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
    %c1_i8 = arith.constant 1 : i8
    // CHECK: int32_t [[A1:.*]] = 262144;
    %c262144_i32 = arith.constant 262144 : i32
    // CHECK: [[NOC1]].async_read([[EP1]], CoreLocalMem<uint32_t>([[C1]]), [[C0]]
    ttkernel.noc_async_read core[%c0_idx, %c0_idx], %c262144_i32, %c262400_i32, %c32_i32 : (index, index, i32, i32, i32) -> ()
    // CHECK: uint64_t [[NOCADDR1:.*]] = [[EP1]].get_noc_unicast_addr(static_cast<uint32_t>([[A0]]), static_cast<uint32_t>([[A0]]), static_cast<uint32_t>([[B1]]), [[NOC1]].get_noc_id())
    %4 = ttkernel.get_noc_addr(%c0_idx, %c0_idx, %c262208_i32) : (index, index, i32) -> !ttkernel.noc_addr
    // CHECK: [[NOC1]].async_read([[EP1]], CoreLocalMem<uint32_t>([[B0]]), [[C0]]
    ttkernel.noc_async_read core[%c0_idx, %c0_idx], %c262208_i32, %c262432_i32, %c32_i32 : (index, index, i32, i32, i32) -> ()
    // CHECK: int32_t [[SEM:.*]] = get_semaphore
    %sem = ttkernel.get_semaphore(%c0_idx) : (index) -> !ttkernel.local_semaphore
    // CHECK: Semaphore([[A0]]).down([[C0]]);
    // SEMDOWN: Semaphore({{.*}}).down({{.*}});
    ttkernel.semaphore_down(%c0_idx, %c32_i32) : (index, i32) -> ()
    // CHECK: Semaphore([[A0]]).set([[C0]]);
    // SEMDOWN: Semaphore({{.*}}).set({{.*}});
    ttkernel.semaphore_set(%c0_idx, %c32_i32) : (index, i32) -> ()
    // CHECK: Semaphore([[A0]]).up([[NOC1]], static_cast<uint32_t>([[A0]]), static_cast<uint32_t>([[A0]]), static_cast<uint32_t>([[C0]]));
    // SEMDOWN: Semaphore({{.*}}).up(noc1, static_cast<uint32_t>({{.*}}), static_cast<uint32_t>({{.*}}), static_cast<uint32_t>({{.*}}));
    ttkernel.semaphore_up_remote(%c0_idx, core[%c0_idx, %c0_idx], %c32_i32, noc %c1_i8) : (index, index, index, i32, i8) -> ()
    // CHECK: noc_semaphore_set_remote([[SEM]], [[NOCADDR1]])
    ttkernel.remote_sram_write_u32(%sem, %4) : (!ttkernel.local_semaphore, !ttkernel.noc_addr) -> ()
    // CHECK-DAG: int32_t [[INLINE_VALUE:.*]] = 7
    // CHECK-DAG: int8_t [[BE:.*]] = 15
    // CHECK-DAG: int8_t [[NOC:.*]] = 1
    %inline_value = arith.constant 7 : i32
    %be = arith.constant 15 : i8
    %noc = arith.constant 1 : i8
    // CHECK: noc_inline_dw_write<InlineWriteDst::L1>([[NOCADDR1]], [[INLINE_VALUE]], [[BE]], [[NOC]])
    ttkernel.noc_inline_dw_write(%4, %inline_value, %be, %noc) : (!ttkernel.noc_addr, i32, i8, i8) -> ()
    // CHECK: [[NOC1]].async_atomic_barrier()
    ttkernel.noc_async_atomic_barrier() : () -> ()
    // CHECK: noc1.async_atomic_barrier()
    ttkernel.noc_async_atomic_barrier(%noc) : (i8) -> ()
    // CHECK: [[NOC1]].async_read_barrier()
    ttkernel.noc_async_read_barrier() : () -> ()
    // CHECK: return
    func.return
}

// CHECK: void kernel_main
func.func @ttkernel_noc_with_noc_id() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // CHECK-DAG: UnicastEndpoint [[EXPLICIT_EP:unicast_ep]]
    // CHECK-DAG: Noc [[EXPLICIT_NOC:noc1]](1)
    // CHECK-DAG: size_t [[EXPLICIT_X:.*]] = 1
    %x = arith.constant 1 : index
    // CHECK-DAG: size_t [[EXPLICIT_Y:.*]] = 2
    %y = arith.constant 2 : index
    // CHECK-DAG: int32_t [[EXPLICIT_ADDR:.*]] = 262400
    %addr = arith.constant 262400 : i32
    // CHECK-DAG: int8_t [[EXPLICIT_NOC_ID:.*]] = 1
    %noc_id = arith.constant 1 : i8
    // CHECK: uint64_t [[EXPLICIT_NOC_ADDR:.*]] = [[EXPLICIT_EP]].get_noc_unicast_addr(static_cast<uint32_t>([[EXPLICIT_X]]), static_cast<uint32_t>([[EXPLICIT_Y]]), static_cast<uint32_t>([[EXPLICIT_ADDR]]), [[EXPLICIT_NOC]].get_noc_id())
    %noc_addr = ttkernel.get_noc_addr(%x, %y, %addr, %noc_id) : (index, index, i32, i8) -> !ttkernel.noc_addr
    // CHECK: [[EXPLICIT_NOC]].async_read_barrier()
    ttkernel.noc_async_read_barrier(%noc_id) : (i8) -> ()
    // CHECK: [[EXPLICIT_NOC]].async_write_barrier()
    ttkernel.noc_async_write_barrier(%noc_id) : (i8) -> ()
    // CHECK: return
    func.return
}
