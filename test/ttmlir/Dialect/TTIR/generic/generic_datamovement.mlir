// RUN: ttmlir-opt --ttcore-register-device --ttir-generic-generate-datamovement -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

func.func @add(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>,  #l1_>, %arg1: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>,  #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>,  #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>,  #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  // Look for 4 regions, one for each operand and one for the compute
  // Operand 0 (input)
  // CHECK: ^datamovement0
  // CHECK-NEXT: ttir.yield
  // Operand 1 (input)
  // CHECK: ^datamovement1
  // CHECK-NEXT: ttir.yield
  // Operand 2 (output)
  // CHECK: ^datamovement2
  // CHECK-NEXT: ttir.await
  // Compute
  // CHECK: ^compute
  // CHECK: ttir.await
  // CHECK: ttir.tile_add
  // CHECK: ttir.yield
  ^bb0(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        %0 = affine.load %cb0[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %1 = affine.load %cb1[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %2 = "ttir.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %2, %cb2[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      }
    }
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>,  #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>,  #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>,  #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>,  #l1_>
}

#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
#reduction = #ttcore.iterator_type<reduction>

func.func @matmul_single_core(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>,  #l1_>, %arg1: memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  // Look for 4 regions, one for each operand and one for the compute
  // Operand 0 (input)
  // CHECK: ^datamovement0
  // CHECK-NEXT: ttir.yield
  // Operand 1 (input)
  // CHECK: ^datamovement1
  // CHECK-NEXT: ttir.yield
  // Operand 2 (output)
  // CHECK: ^datamovement2
  // CHECK-NEXT: ttir.await
  // Compute
  // CHECK: ^compute
  // CHECK: ttir.await
  // CHECK: ttir.tile_matmul_block
  // CHECK: ttir.yield
  ^bb0(%cb0: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>,  #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

func.func @matmul_single_core_stream(%arg0: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, %arg1: memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  %0 = "ttir.stream_layout"(%arg0, %cb0_alloc) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #ttcore.view<map(4)>, #l1_>
  %1 = "ttir.stream_layout"(%arg1, %cb1_alloc) : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #ttcore.view<map(4)>, #l1_>
  // CHECK: ttir.generic
  // CHECK-NEXT: ins([[lhs:%[a-z0-9_]+]], [[rhs:%[a-z0-9_]+]] : {{.*}})
  // CHECK-NEXT: outs([[out:%[a-z0-9_]+]] : {{.*}})
  "ttir.generic"(%0, %1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  // Look for 4 regions, one for each operand and one for the compute
  // Operand 0 (input)
  // CHECK: ^datamovement0
  // CHECK-NEXT: ttir.dma [[lhs]]<#map1>, %cb0
  // CHECK-NEXT: ttir.dma_wait
  // CHECK-NEXT: ttir.yield %cb0
  // Operand 1 (input)
  // CHECK: ^datamovement1
  // CHECK-NEXT: ttir.dma [[rhs]]<#map2>, %cb1
  // CHECK-NEXT: ttir.dma_wait
  // CHECK-NEXT: ttir.yield %cb1
  // Operand 2 (output)
  // CHECK: ^datamovement2
  // CHECK-NEXT: ttir.await
  // Compute
  // CHECK: ^compute
  // CHECK: ttir.await
  // CHECK: ttir.tile_matmul_block
  // CHECK: ttir.yield
  ^bb0(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #ttcore.view<map(4)>, #l1_>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #ttcore.view<map(4)>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

func.func @matmul_multi_core(%arg0: memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, %arg1: memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>
  %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  %0 = "ttir.stream_layout"(%arg0, %cb0_alloc) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #ttcore.view<map(4)>, #l1_>
  %1 = "ttir.stream_layout"(%arg1, %cb1_alloc) : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>, memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  // CHECK: ttir.generic
  // CHECK-NEXT: ins([[lhs:%[a-z0-9_]+]], [[rhs:%[a-z0-9_]+]] : {{.*}})
  // CHECK-NEXT: outs([[out:%[a-z0-9_]+]] : {{.*}})
  "ttir.generic"(%0, %1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  // Look for 4 regions, one for each operand and one for the compute
  // Operand 0 (input)
  // CHECK: ^datamovement0
  // CHECK: ttir.dma [[lhs]]<#map1>, %cb0
  // CHECK-NEXT: ttir.dma_wait
  // CHECK-NEXT: ttir.semaphore_wait [[reader_ready_lhs:%[a-z0-9]+]]
  // CHECK-NEXT: ttir.dma %cb0, %cb0
  // CHECK-NEXT: ttir.dma_wait
  // CHECK-NEXT: ttir.semaphore_set [[writer_done_lhs:%[a-z0-9]+]]
  // CHECK-NEXT: else
  // CHECK-NEXT: ttir.semaphore_inc [[reader_ready_lhs]]
  // CHECK-NEXT: ttir.semaphore_wait [[writer_done_lhs]]
  // Operand 1 (input)
  // CHECK: ^datamovement1
  // CHECK: ttir.dma [[rhs]]<#map2>, %cb1
  // CHECK-NEXT: ttir.dma_wait
  // CHECK-NEXT: ttir.semaphore_wait [[reader_ready_rhs:%[a-z0-9]+]]
  // CHECK-NEXT: ttir.dma %cb1, %cb1
  // CHECK-NEXT: ttir.dma_wait
  // CHECK-NEXT: ttir.semaphore_set [[writer_done_rhs:%[a-z0-9]+]]
  // CHECK-NEXT: else
  // CHECK-NEXT: ttir.semaphore_inc [[reader_ready_rhs]]
  // CHECK-NEXT: ttir.semaphore_wait [[writer_done_rhs]]
  // Operand 2 (output)
  // CHECK: ^datamovement2
  // CHECK-NEXT: ttir.await
  // Compute
  // CHECK: ^compute
  // CHECK: ttir.await
  // CHECK: ttir.tile_matmul_block
  // CHECK: ttir.yield
  ^bb0(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<4x8x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #ttcore.view<map(4)>, #l1_>, memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
}

func.func @matmul_multi_core_dram_params(%arg0: memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, %arg1: memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #dram>) -> memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>
  %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  %0 = "ttir.stream_layout"(%arg0, %cb0_alloc) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #ttcore.view<map(4)>, #l1_>
  %1 = "ttir.stream_layout"(%arg1, %cb1_alloc) : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #dram>, memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>
  // CHECK: ttir.generic
  // CHECK-NEXT: ins([[lhs:%[a-z0-9_]+]], [[rhs:%[a-z0-9_]+]] : {{.*}})
  // CHECK-NEXT: outs([[out:%[a-z0-9_]+]] : {{.*}})
  "ttir.generic"(%0, %1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  // Look for 4 regions, one for each operand and one for the compute
  // Operand 0 (input)
  // CHECK: ^datamovement0
  // CHECK: ttir.dma [[lhs]]<#map1>, %cb0
  // CHECK-NEXT: ttir.dma_wait
  // CHECK-NEXT: ttir.semaphore_wait [[reader_ready_lhs:%[a-z0-9]+]]
  // CHECK-NEXT: ttir.dma %cb0, %cb0
  // CHECK-NEXT: ttir.dma_wait
  // CHECK-NEXT: ttir.semaphore_set [[writer_done_lhs:%[a-z0-9]+]]
  // CHECK-NEXT: else
  // CHECK-NEXT: ttir.semaphore_inc [[reader_ready_lhs]]
  // CHECK-NEXT: ttir.semaphore_wait [[writer_done_lhs]]
  // Operand 1 (input)
  // CHECK: ^datamovement1
  // CHECK: ttir.dma [[rhs]]<#map2>, %cb1
  // CHECK-NEXT: ttir.dma_wait
  // CHECK-NEXT: ttir.semaphore_wait [[reader_ready_rhs:%[a-z0-9]+]]
  // CHECK-NEXT: ttir.dma %cb1, %cb1
  // CHECK-NEXT: ttir.dma_wait
  // CHECK-NEXT: ttir.semaphore_set [[writer_done_rhs:%[a-z0-9]+]]
  // CHECK-NEXT: else
  // CHECK-NEXT: ttir.semaphore_inc [[reader_ready_rhs]]
  // CHECK-NEXT: ttir.semaphore_wait [[writer_done_rhs]]
  // Operand 2 (output)
  // CHECK: ^datamovement2
  // CHECK-NEXT: ttir.await
  // Compute
  // CHECK: ^compute
  // CHECK: ttir.await
  // CHECK: ttir.tile_matmul_block
  // CHECK: ttir.yield
  ^bb0(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<4x8x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #ttcore.view<map(4)>, #l1_>, memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
}

func.func @tilize(%arg0: memref<2x4x128x192xf32, #ttcore.shard<768x4>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 1, 1>}> ({
  // Look for 4 regions, one for each operand and one for the compute
  // Operand 0 (input)
  // CHECK: ^datamovement0
  // CHECK-NEXT: ttir.yield
  // Operand 1 (output)
  // CHECK: ^datamovement1
  // CHECK-NEXT: ttir.await
  // Compute
  // CHECK: ^compute
  // CHECK: ttir.await
  // CHECK: ttir.tile_tilize_block
  // CHECK: ttir.yield
  ^compute(%cb0: memref<128x192xf32, #l1_>, %cb1: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>):
    "ttir.tile_tilize_block"(%cb0, %cb1) : (memref<128x192xf32, #l1_>, memref<4x6x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x128x192xf32, #ttcore.shard<768x4>, #l1_>, memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> ()
  return %alloc : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>
}

func.func @untilize(%arg0: memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<2x4x128x192xf32, #ttcore.shard<768x4>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x128x192xf32, #ttcore.shard<768x4>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 1, 1>}> ({
  // Look for 4 regions, one for each operand and one for the compute
  // Operand 0 (input)
  // CHECK: ^datamovement0
  // CHECK-NEXT: ttir.yield
  // Operand 1 (output)
  // CHECK: ^datamovement1
  // CHECK-NEXT: ttir.await
  // Compute
  // CHECK: ^compute
  // CHECK: ttir.await
  // CHECK: ttir.tile_untilize_block
  // CHECK: ttir.yield
  ^compute(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<128x192xf32, #l1_>):
    "ttir.tile_untilize_block"(%cb0, %cb1) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<128x192xf32, #l1_>) -> ()
  }) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, memref<2x4x128x192xf32, #ttcore.shard<768x4>, #l1_>) -> ()
  return %alloc : memref<2x4x128x192xf32, #ttcore.shard<768x4>, #l1_>
}

!inT = memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #ttcore.memory_space<l1>>
!sbT = !inT
!stT = memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #ttcore.memory_space<l1>>
!cbT = memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>

func.func @inferDMAForWrite(%arg0 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #ttcore.memory_space<l1>>)  {

  %inA = memref.alloc() {address = 296352 : i64, alignment = 16 : i64} : !inT
  %sbA = memref.alloc() {address = 361888 : i64, alignment = 16 : i64} : !sbT
  %streamA = "ttir.stream_layout"(%inA, %sbA) : (!inT, !sbT) -> !stT

  %inB = memref.alloc() {address = 427424 : i64, alignment = 16 : i64} : !inT
  %sbB = memref.alloc() {address = 492960 : i64, alignment = 16 : i64} : !sbT
  %streamB = "ttir.stream_layout"(%inB, %sbB) : (!inT, !sbT) -> !stT

  %out = memref.alloc() {address = 558496 : i64, alignment = 16 : i64} : !inT
  %sbOut = memref.alloc() {address = 624032 : i64, alignment = 16 : i64} : !sbT
  %streamOut = "ttir.stream_layout"(%out, %sbOut) : (!inT, !sbT) -> !stT

  ttir.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#ttir.thread<compute>]}
      ins(%streamA, %streamB : !stT, !stT)
      outs(%streamOut : !stT)
  // CHECK: ^datamovement0
  // CHECK: {{%.*}} = ttir.dma {{%.*}}<#map>, {{%.*}}
  // CHECK: ttir.dma_wait
  // CHECK: ttir.yield
  // CHECK: ^datamovement1
  // CHECK: {{%.*}} = ttir.dma {{%.*}}<#map>, {{%.*}}
  // CHECK: ttir.dma_wait
  // CHECK: ttir.yield
  // CHECK: ^datamovement2
  // CHECK: ttir.await
  // CHECK: {{%.*}} = ttir.dma {{%.*}}, {{%.*}}<#map>
  // CHECK: ttir.dma_wait
  // CHECK: ^compute
  {
  ^compute0(%cb0 : !cbT, %cb1 : !cbT, %cb2 : !cbT):
  }

  return
}
