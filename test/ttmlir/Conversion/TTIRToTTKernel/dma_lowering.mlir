// RUN: ttmlir-opt --convert-ttir-to-ttkernel %s  | FileCheck %s

#l1_ = #tt.memory_space<l1>
#dram = #tt.memory_space<dram>
#system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_cores = {worker = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  1x0,  1x1,  1x2,  1x3,  1x4,  1x5,  1x6,  1x7,  2x0,  2x1,  2x2,  2x3,  2x4,  2x5,  2x6,  2x7,  3x0,  3x1,  3x2,  3x3,  3x4,  3x5,  3x6,  3x7,  4x0,  4x1,  4x2,  4x3,  4x4,  4x5,  4x6,  4x7,  5x0,  5x1,  5x2,  5x3,  5x4,  5x5,  5x6,  5x7,  6x0,  6x1,  6x2,  6x3,  6x4,  6x5,  6x6,  6x7,  7x0,  7x1,  7x2,  7x3,  7x4,  7x5,  7x6,  7x7] dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>

module attributes {tt.system_desc = #system_desc} {
tt.device @default_device = <workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

// Test 1: Local to local DMA within same core
// CHECK-LABEL: func.func @test_local_to_local_same_core
func.func @test_local_to_local_same_core(%arg0: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<2x2x!tt.tile<32x32, f32>, #l1_>) {
  // CHECK: ttkernel.get_read_ptr
  // CHECK: ttkernel.get_write_ptr
  // CHECK: ttkernel.get_noc_addr_xy
  // CHECK: ttkernel.noc_async_write
  %0 = ttir.dma %arg0, %arg1 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  return
}

// Test 2: Local to local DMA between different cores
// CHECK-LABEL: func.func @test_local_to_local_remote_core
func.func @test_local_to_local_remote_core(%arg0: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<2x2x!tt.tile<32x32, f32>, #l1_>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: ttkernel.get_read_ptr
  // CHECK: ttkernel.get_write_ptr
  // CHECK: ttkernel.get_noc_addr_xy
  // CHECK: ttkernel.noc_async_write
  %0 = ttir.dma %arg0, %arg1, core[%c1, %c2] : (memref<2x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  return
}

// Test 3: Local to remote multicast DMA with same source and destination (regular multicast)
// CHECK-LABEL: func.func @test_local_to_remote_multicast_regular
func.func @test_local_to_remote_multicast_regular(%arg0: memref<2x2x!tt.tile<32x32, f32>, #l1_>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  // CHECK: ttkernel.get_read_ptr
  // CHECK: ttkernel.get_write_ptr
  // CHECK: ttkernel.get_noc_multicast_addr
  // CHECK: ttkernel.noc_async_write_multicast
  %0 = ttir.dma %arg0, %arg0, core[%c1, %c2] mcast[%c3, %c4] : (memref<2x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  return
}

// Test 4: Local to remote multicast DMA with different source and destination (loopback source)
// CHECK-LABEL: func.func @test_local_to_remote_multicast_loopback
func.func @test_local_to_remote_multicast_loopback(%arg0: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<2x2x!tt.tile<32x32, f32>, #l1_>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  // CHECK: ttkernel.get_read_ptr
  // CHECK: ttkernel.get_write_ptr
  // CHECK: ttkernel.get_noc_multicast_addr
  // CHECK: ttkernel.noc_async_write_multicast_loopback_src
  %0 = ttir.dma %arg0, %arg1, core[%c1, %c2] mcast[%c3, %c4] : (memref<2x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  return
}

// Test 5: Remote to local DMA (L1 from another core)
// CHECK-LABEL: func.func @test_remote_l1_to_local
func.func @test_remote_l1_to_local(%dst: memref<2x2x!tt.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %src = ttir.get_global_operand(0) : memref<1x2x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 16384 + d3 * 4096)>, #l1_>
  // CHECK: ttkernel.get_write_ptr
  // CHECK: ttkernel.get_noc_addr_xy
  // CHECK: ttkernel.noc_async_read
  %0 = ttir.dma %src[%c0, %c1], %dst : (memref<1x2x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 16384 + d3 * 4096)>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  return
}

// Test 7: DMA wait operation for read
// CHECK-LABEL: func.func @test_dma_wait_read
func.func @test_dma_wait_read(%dst: memref<2x2x!tt.tile<32x32, f32>, #l1_>) {
  %src = ttir.get_global_operand(0) : memref<1x2x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 16384 + d3 * 4096)>, #l1_>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = ttir.dma %src[%c0, %c1], %dst : (memref<1x2x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 16384 + d3 * 4096)>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  // CHECK: ttkernel.noc_async_read_barrier
  ttir.dma_wait %0
  return
}

// Test 8: DMA wait operation for write
// CHECK-LABEL: func.func @test_dma_wait_write
func.func @test_dma_wait_write(%arg0: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<2x2x!tt.tile<32x32, f32>, #l1_>) {
  %0 = ttir.dma %arg0, %arg1 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  // CHECK: ttkernel.noc_async_write_barrier
  ttir.dma_wait %0
  return
}

}
