// RUN: ttmlir-opt --ttcore-register-device --ttir-generic-lower-dmas --convert-ttir-to-ttkernel -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

module {

// Test 1: Local to local DMA within same core
// CHECK-LABEL: func.func @test_local_to_local_same_core
func.func @test_local_to_local_same_core(%arg0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<datamovement>} {
  // CHECK: ttkernel.get_read_ptr
  // CHECK: ttkernel.get_write_ptr
  // CHECK: ttkernel.get_noc_addr
  // CHECK: ttkernel.noc_async_write
  %c0 = arith.constant 0 : index
  ttir.await %arg0 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  %0 = ttir.dma %arg0[%c0, %c0], %arg1[%c0, %c0] : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  ttir.dma_wait %0
  ttir.yield %arg1 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  return
}

// Test 2: Local to local DMA between different cores
// CHECK-LABEL: func.func @test_local_to_local_remote_core
func.func @test_local_to_local_remote_core(%arg0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<datamovement>} {
  %dst = ttir.get_global_operand(0) : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttir.await %arg0 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  // CHECK: ttkernel.get_read_ptr
  // CHECK: ttkernel.get_noc_addr
  // CHECK: ttkernel.noc_async_write
  %0 = ttir.dma %arg0[%c0, %c0], %dst[%c0, %c1, %c0, %c0] : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> !ttir.mem_tx
  ttir.dma_wait %0
  return
}

// Test 3: Local to remote multicast DMA with same source and destination (regular multicast)
// CHECK-LABEL: func.func @test_local_to_remote_multicast_regular
func.func @test_local_to_remote_multicast_regular(%arg0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<datamovement>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  // CHECK: ttkernel.get_write_ptr
  // CHECK: ttkernel.experimental::get_noc_multicast_addr
  // CHECK: ttkernel.noc_async_write_multicast
  %0 = ttir.dma %arg0[%c0, %c0], %arg0[%c0, %c0] core[%c1, %c2] mcast[%c3, %c4] : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  ttir.dma_wait %0
  ttir.yield %arg0 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  return
}

// Test 4: Local to remote multicast DMA with different source and destination (loopback source)
// CHECK-LABEL: func.func @test_local_to_remote_multicast_loopback
func.func @test_local_to_remote_multicast_loopback(%arg0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<datamovement>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  // CHECK: ttkernel.get_read_ptr
  // CHECK: ttkernel.get_write_ptr
  // CHECK: ttkernel.experimental::get_noc_multicast_addr
  // CHECK: ttkernel.noc_async_write_multicast_loopback_src
  ttir.await %arg0 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  %0 = ttir.dma %arg0[%c0, %c0], %arg1[%c0, %c0] core[%c1, %c2] mcast[%c3, %c4] : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  ttir.dma_wait %0
  ttir.yield %arg1 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  return
}

// Test 5: Remote to local DMA (L1 from another core)
// CHECK-LABEL: func.func @test_remote_l1_to_local
func.func @test_remote_l1_to_local(%dst: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<datamovement>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %src = ttir.get_global_operand(0) : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  // CHECK: ttkernel.get_noc_addr
  // CHECK: ttkernel.get_write_ptr
  // CHECK: ttkernel.noc_async_read
  %0 = ttir.dma %src[%c0, %c1, %c0, %c0], %dst[%c0, %c0] : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  ttir.dma_wait %0
  ttir.yield %dst : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  return
}

// Test 7: DMA wait operation for read
// CHECK-LABEL: func.func @test_dma_wait_read
func.func @test_dma_wait_read(%dst: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<datamovement>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %src = ttir.get_global_operand(0) : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  %0 = ttir.dma %src[%c0, %c1, %c0, %c0], %dst[%c0, %c0] : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  // CHECK: ttkernel.noc_async_read_barrier
  ttir.dma_wait %0
  ttir.yield %dst : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  return
}

// Test 8: DMA wait operation for write
// CHECK-LABEL: func.func @test_dma_wait_write
func.func @test_dma_wait_write(%arg0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<datamovement>} {
  %c0 = arith.constant 0 : index
  ttir.await %arg0 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  %0 = ttir.dma %arg0[%c0, %c0], %arg1[%c0, %c0] : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
  // CHECK: ttkernel.noc_async_write_barrier
  ttir.dma_wait %0
  ttir.yield %arg1 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  return
}

}
