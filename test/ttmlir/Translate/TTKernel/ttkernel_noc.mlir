// RUN: ttmlir-opt --convert-ttkernel-to-emitc -o %t %s
// RUN: ttmlir-translate --ttkernel-to-cpp -o %t.cpp %t
// RUN: FileCheck %s --input-file=%t.cpp

// CHECK: #include "api/core_local_mem.h"
// CHECK: #include "api/dataflow/circular_buffer.h"
// CHECK: #include "api/dataflow/dataflow_api.h"
// CHECK: #include "api/dataflow/endpoints.h"
// CHECK: #include "api/dataflow/noc.h"
// CHECK: void kernel_main
func.func @ttkernel_noc_cb() -> () attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    // CHECK: UnicastEndpoint [[EP:unicast_ep]]
    // CHECK: Noc [[NOC:noc0]]
    // CHECK: int32_t [[C0:.*]] = 32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: size_t [[A0:.*]] = 0
    %c0_idx = arith.constant 0 : index
    // CHECK: int32_t [[A1:.*]] = 262144;
    %c262144_i32 = arith.constant 262144 : i32
    %noc0 = arith.constant 0 : i8
    %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
    // CHECK: CircularBuffer [[CB:cb_ctarg_0]](get_compile_time_arg_val(0));
    %c1_i32 = arith.constant 1 : i32
    // CHECK: [[CB]].reserve_back
    "ttkernel.cb_reserve_back"(%cb, %c1_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, i32) -> ()
    %wptr = "ttkernel.get_write_ptr"(%cb) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>) -> i32
    // CHECK: [[NOC]].async_read([[EP]], CoreLocalMem<uint32_t>([[CB]].get_write_ptr()), [[C0]]
    ttkernel.noc_async_read core[%c0_idx, %c0_idx], %c262144_i32, %wptr, %c32_i32, noc %noc0 : (index, index, i32, i32, i32, i8) -> ()
    // CHECK: [[NOC]].async_read_barrier()
    ttkernel.noc_async_read_barrier(%noc0) : (i8) -> ()
    // CHECK: [[CB]].push_back
    "ttkernel.cb_push_back"(%cb, %c1_i32) : (!ttkernel.cb<8, !ttcore.tile<32x32, f32>>, i32) -> ()
    // CHECK: return
    func.return
}

// CHECK: void kernel_main
func.func @ttkernel_noc() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // CHECK: UnicastEndpoint [[EP1:unicast_ep]]
    // CHECK: Noc [[NOC1:noc0]]
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
    %noc0 = arith.constant 0 : i8
    // CHECK: [[NOC1]].async_read([[EP1]], CoreLocalMem<uint32_t>([[C1]]), [[C0]]
    ttkernel.noc_async_read core[%c0_idx, %c0_idx], %c262144_i32, %c262400_i32, %c32_i32, noc %noc0 : (index, index, i32, i32, i32, i8) -> ()
    // CHECK: uint64_t [[NOCADDR1:.*]] = [[EP1]].get_noc_unicast_addr(static_cast<uint32_t>([[A0]]), static_cast<uint32_t>([[A0]]), static_cast<uint32_t>([[B1]]), [[NOC1]].get_noc_id())
    %4 = ttkernel.get_noc_addr(%c0_idx, %c0_idx, %c262208_i32, %noc0) : (index, index, i32, i8) -> !ttkernel.noc_addr
    // CHECK: [[NOC1]].async_read([[EP1]], CoreLocalMem<uint32_t>([[B0]]), [[C0]]
    ttkernel.noc_async_read core[%c0_idx, %c0_idx], %c262208_i32, %c262432_i32, %c32_i32, noc %noc0 : (index, index, i32, i32, i32, i8) -> ()
    // CHECK: int32_t [[SEM:.*]] = get_semaphore
    %sem = ttkernel.get_semaphore(%c0_idx) : (index) -> !ttkernel.local_semaphore
    // CHECK: noc_semaphore_set_remote([[SEM]], [[NOCADDR1]])
    ttkernel.remote_sram_write_u32(%sem, %4) : (!ttkernel.local_semaphore, !ttkernel.noc_addr) -> ()
    // CHECK-DAG: int32_t [[INLINE_VALUE:.*]] = 7
    // CHECK-DAG: int8_t [[BE:.*]] = 15
    // CHECK-DAG: int8_t [[NOC:.*]] = 1
    %inline_value = arith.constant 7 : i32
    %be = arith.constant 15 : i8
    %noc = arith.constant 1 : i8
    // CHECK: noc1.inline_dw_write<NocOptions::INLINE_L1>([[EP1]], [[INLINE_VALUE]]
    ttkernel.noc_inline_dw_write(core[%c0_idx, %c0_idx], %c262208_i32, %inline_value, %be, noc %noc) : (index, index, i32, i32, i8, i8) -> ()
    // CHECK: [[NOC1]].async_atomic_barrier()
    ttkernel.noc_async_atomic_barrier(%noc0) : (i8) -> ()
    // CHECK: noc1.async_atomic_barrier()
    ttkernel.noc_async_atomic_barrier(%noc) : (i8) -> ()
    // CHECK: [[NOC1]].async_read_barrier()
    ttkernel.noc_async_read_barrier(%noc0) : (i8) -> ()
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

// Render the non-D2M NoC emissions (stateful one-packet, TRID write/barrier,
// multicast one-packet, tile read/write) all the way to C++ so that brace
// balance in the emitted OO calls is actually checked. Checking the verbatim
// format string alone cannot catch a malformed `NocOptVals{...}}` because the
// `}}` is only collapsed at C++ rendering time, not in the verbatim op.
// CHECK: void kernel_main
func.func @ttkernel_noc_oo_render() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // CHECK-DAG: UnicastEndpoint [[EP:unicast_ep]]
    // CHECK-DAG: MulticastEndpoint [[MEP:mcast_ep]]
    // CHECK-DAG: Noc [[NOC:noc0]]
    %trid = arith.constant 3 : i32
    %x = arith.constant 0 : index
    %y = arith.constant 1 : index
    %addr = arith.constant 512 : i32
    %size = arith.constant 128 : i32
    %dstl1 = arith.constant 262400 : i32
    %num = arith.constant 7 : i32
    %xe = arith.constant 3 : index
    %ye = arith.constant 3 : index
    %noc0 = arith.constant 0 : i8

    // CHECK: [[NOC]].set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>([[EP]], {{.*}}, {.noc_x = {{.*}}, .noc_y = {{.*}}, .addr = static_cast<uint32_t>({{.*}})});
    ttkernel.noc_async_read_one_packet_set_state(core[%x, %y], %addr, %size, noc %noc0) : (index, index, i32, i32, i8) -> ()

    // CHECK: [[NOC]].async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>([[EP]], CoreLocalMem<uint32_t>({{.*}}), {{.*}}, {.noc_x = {{.*}}, .noc_y = {{.*}}, .addr = static_cast<uint32_t>({{.*}})}, {});
    ttkernel.noc_async_read_one_packet_with_state(core[%x, %y], %addr, %dstl1, %size, noc %noc0) : (index, index, i32, i32, i32, i8) -> ()

    // The TRID write must close with a single balanced `NocOptVals{...}`.
    // CHECK: [[NOC]].async_write<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(CoreLocalMem<uint32_t>({{.*}}), [[EP]], {{.*}}, {} , {.noc_x = {{.*}}, .noc_y = {{.*}}, .addr = static_cast<uint32_t>({{.*}})}, NocOptVals{.trid = {{[^}]*}}});
    ttkernel.noc_async_write_one_packet_with_trid(%dstl1, core[%x, %y], %addr, %size, %trid, noc %noc0) : (i32, index, index, i32, i32, i32, i8) -> ()

    // CHECK: [[NOC]].async_write_multicast<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(CoreLocalMem<uint32_t>({{.*}}), [[MEP]], {{.*}}, {{.*}}, {} , noc_traits_t<MulticastEndpoint>::dst_args_mcast_type{.noc_x_start = {{.*}}, .noc_y_start = {{.*}}, .noc_x_end = {{.*}}, .noc_y_end = {{.*}}, .addr = static_cast<uint32_t>({{.*}})}, false);
    ttkernel.noc_async_write_multicast_one_packet(%dstl1, %size, %num, start_xy[%x, %y], end_xy[%xe, %ye], %addr, noc %noc0) : (i32, i32, i32, index, index, index, index, i32, i8) -> ()

    // The TRID barrier must close with a single balanced `NocOptVals{...}`.
    // CHECK: [[NOC]].async_read_barrier<NocOptions::TXN_ID>(NocOptVals{.trid = {{[^}]*}}});
    ttkernel.noc_async_read_barrier_with_trid(%trid, %noc0) : (i32, i8) -> ()
    // CHECK: return
    func.return
}

// CHECK: void kernel_main
func.func @ttkernel_noc_oo_tile_render() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
    // CHECK: Noc [[NOC:noc0]]
    %cta = arith.constant 2 : i32
    %crta = arith.constant 0 : i32
    %addr = arith.constant 262400 : i32
    %tile_size = arith.constant 8 : i32
    %tile = arith.constant 1 : i32
    %noc0 = arith.constant 0 : i8
    %args = ttkernel.TensorAccessorArgs(%cta, %crta)
    // CHECK: TensorAccessor [[ACC:v[0-9]+]] = TensorAccessor(
    %s = "ttkernel.TensorAccessor"(%args, %addr, %tile_size) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
    // CHECK: [[NOC]].async_read([[ACC]], CoreLocalMem<uint32_t>({{.*}}), [[ACC]].get_aligned_page_size(), {.page_id = static_cast<uint32_t>({{.*}})}, {});
    "ttkernel.noc_async_read_tile"(%tile, %s, %addr, %noc0) : (i32, !ttkernel.TensorAccessor, i32, i8) -> ()
    // CHECK: [[NOC]].async_write(CoreLocalMem<uint32_t>({{.*}}), [[ACC]], [[ACC]].get_aligned_page_size(), {} , {.page_id = static_cast<uint32_t>({{.*}})});
    "ttkernel.noc_async_write_tile"(%tile, %s, %addr, %noc0) : (i32, !ttkernel.TensorAccessor, i32, i8) -> ()
    // CHECK: return
    func.return
}
