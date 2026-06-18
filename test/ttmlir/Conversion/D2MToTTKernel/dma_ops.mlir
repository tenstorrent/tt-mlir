// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

module {
  func.func private @fabric_dma_write_to_dram_uses_bank_noc_addr() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %src_cb = d2m.get_cb(0) : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>>
    %dst = d2m.get_arg(0) resolution_stage = compile : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
    %src = d2m.wait %src_cb : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // CHECK-LABEL: func.func private @fabric_dma_write_to_dram_uses_bank_noc_addr
    // CHECK: %[[DRAM_BASE:[0-9]+]] = builtin.unrealized_conversion_cast
    // CHECK: %[[BANK_ID:[0-9]+]] = arith.index_cast {{.*}} : index to i32
    // CHECK: %[[DRAM_OFFSET:[0-9]+]] = arith.index_cast {{.*}} : index to i32
    // CHECK: %[[DRAM_ADDR:[0-9]+]] = arith.addi %[[DRAM_BASE]], %[[DRAM_OFFSET]] : i32
    // CHECK: %[[NOC_ADDR:[0-9]+]] = ttkernel.get_noc_addr_from_bank_id(%[[BANK_ID]], %[[DRAM_ADDR]]) : (i32, i32) -> !ttkernel.noc_addr
    // CHECK: ttkernel.experimental.fabric_fast_write_any_len({{.*}}, %[[NOC_ADDR]], {{.*}})
    %tx = d2m.dma_write %src[%c0], %dst[%c0, %c0, %c0] startDevice[%c0, %c1], <1> : (memref<1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>) -> !d2m.mem_tx<write>
    d2m.dma_wait %tx : !d2m.mem_tx<write>
    return
  }

  func.func private @local_dma_read_uses_translated_self_coords() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %src_cb = d2m.get_cb(0) : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>>
    %dst_cb = d2m.get_cb(1) : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>>
    %src = d2m.wait %src_cb : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x!ttcore.tile<32x32, f32>, #l1>
    %dst = d2m.reserve %dst_cb : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index

    // CHECK-LABEL: func.func private @local_dma_read_uses_translated_self_coords
    // CHECK: %[[MY_Y:[0-9]+]] = ttkernel.my_logical_y_
    // CHECK: %[[MY_X:[0-9]+]] = ttkernel.my_logical_x_
    // CHECK: %[[VIRT_Y:[0-9]+]] = ttkernel.experimental.convert_logical_y_to_translated(%[[MY_Y]]) : (index) -> index
    // CHECK: %[[VIRT_X:[0-9]+]] = ttkernel.experimental.convert_logical_x_to_translated(%[[MY_X]]) : (index) -> index
    // CHECK: ttkernel.noc_async_read core[%[[VIRT_X]], %[[VIRT_Y]]],
    %tx = d2m.dma_read %src[%c0], %dst[%c0], <1> : (memref<1x!ttcore.tile<32x32, f32>, #l1>, memref<1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
    d2m.dma_wait %tx : !d2m.mem_tx<read>
    return
  }

  func.func private @local_mcast_dma_write_loopback_uses_direct_coords() attributes {d2m.thread = #d2m.thread<datamovement, dm_core = 0>} {
    %src_cb = d2m.get_cb(0) : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>>
    %dst_cb = d2m.get_cb(1) : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>>
    %src = d2m.wait %src_cb : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x!ttcore.tile<32x32, f32>, #l1>
    %dst = d2m.reserve %dst_cb : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // CHECK-LABEL: func.func private @local_mcast_dma_write_loopback_uses_direct_coords
    // CHECK: %[[START_Y:[0-9]+]] = ttkernel.experimental.convert_logical_y_to_translated
    // CHECK: %[[START_X:[0-9]+]] = ttkernel.experimental.convert_logical_x_to_translated
    // CHECK: %[[END_Y:[0-9]+]] = ttkernel.experimental.convert_logical_y_to_translated
    // CHECK: %[[END_X:[0-9]+]] = ttkernel.experimental.convert_logical_x_to_translated
    // CHECK: %[[NUM_DESTS:[0-9]+]] = arith.index_cast {{.*}} : index to i32
    // CHECK: ttkernel.noc_async_write_multicast_loopback_src({{.*}}, {{.*}}, %[[NUM_DESTS]], start_xy[%[[END_X]], %[[END_Y]]], end_xy[%[[START_X]], %[[START_Y]]], {{.*}}, noc %{{[a-zA-Z0-9_]+}}, linked true)
    %tx = d2m.dma_write %src[%c0], %dst[%c0] core[%c1, %c2] mcast[%c2, %c3], <1> : (memref<1x!ttcore.tile<32x32, f32>, #l1>, memref<1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<write>
    d2m.dma_wait %tx : !d2m.mem_tx<write>
    return
  }

  func.func private @local_dma_write_uses_translated_self_coords() attributes {d2m.thread = #d2m.thread<datamovement>} {
    %src_cb = d2m.get_cb(0) : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>>
    %dst_cb = d2m.get_cb(1) : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>>
    %src = d2m.wait %src_cb : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x!ttcore.tile<32x32, f32>, #l1>
    %dst = d2m.reserve %dst_cb : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x!ttcore.tile<32x32, f32>, #l1>
    %c0 = arith.constant 0 : index

    // CHECK-LABEL: func.func private @local_dma_write_uses_translated_self_coords
    // CHECK: %[[SRC_ADDR:[0-9]+]] = ttkernel.get_read_ptr
    // CHECK: %[[DST_ADDR:[0-9]+]] = ttkernel.get_write_ptr
    // CHECK: %[[MY_Y:[0-9]+]] = ttkernel.my_logical_y_
    // CHECK: %[[MY_X:[0-9]+]] = ttkernel.my_logical_x_
    // CHECK: %[[VIRT_Y:[0-9]+]] = ttkernel.experimental.convert_logical_y_to_translated(%[[MY_Y]]) : (index) -> index
    // CHECK: %[[VIRT_X:[0-9]+]] = ttkernel.experimental.convert_logical_x_to_translated(%[[MY_X]]) : (index) -> index
    // CHECK: ttkernel.noc_async_write %[[SRC_ADDR]], core[%[[VIRT_X]], %[[VIRT_Y]]], %[[DST_ADDR]],
    %tx = d2m.dma_write %src[%c0], %dst[%c0], <1> : (memref<1x!ttcore.tile<32x32, f32>, #l1>, memref<1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<write>
    d2m.dma_wait %tx : !d2m.mem_tx<write>
    return
  }
}
