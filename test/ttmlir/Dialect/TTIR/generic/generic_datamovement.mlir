// RUN: ttmlir-opt --ttir-load-system-desc --ttir-implicit-device --ttir-generic-datamovement %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>

func.func @add(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 2, 1>, operand_cb_mapping = array<i64>}> ({
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
  ^bb0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        %0 = affine.load %cb0[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %1 = affine.load %cb1[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %2 = "ttir.tile_add"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
        affine.store %2, %cb2[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
      }
    }
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
}

#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
#reduction = #tt.iterator_type<reduction>

func.func @matmul_single_core(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x4x2x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>, operand_cb_mapping = array<i64>}> ({
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
  ^bb0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x4x2x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
}

func.func @matmul_single_core_stream(%arg0: memref<1x2x2x2x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<2x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
  %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
  %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
  %0 = "ttir.stream_layout"(%arg0, %cb0_alloc) : (memref<1x2x2x2x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> memref<1x2x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 2 * 4096 + d3 * 4096)>, #l1_>
  %1 = "ttir.stream_layout"(%arg1, %cb1_alloc) : (memref<2x1x2x2x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> memref<2x1x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 2 * 4096 + d3 * 4096)>, #l1_>
  // CHECK: "ttir.generic"([[lhs:%[a-z0-9_]+]], [[rhs:%[a-z0-9_]+]], [[out:%[a-z0-9_]+]])
  "ttir.generic"(%0, %1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>, operand_cb_mapping = array<i64>}> ({
  // Look for 4 regions, one for each operand and one for the compute
  // Operand 0 (input)
  // CHECK: ^datamovement0
  // CHECK-NEXT: ttir.dma [[lhs]] #map1, %cb0
  // CHECK-NEXT: ttir.dma_wait
  // CHECK-NEXT: ttir.yield %cb0
  // Operand 1 (input)
  // CHECK: ^datamovement1
  // CHECK-NEXT: ttir.dma [[rhs]] #map2, %cb1
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
  ^bb0(%cb0: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x2x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 2 * 4096 + d3 * 4096)>, #l1_>, memref<2x1x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 2 * 4096 + d3 * 4096)>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
}

func.func @matmul_multi_core(%arg0: memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<4x4x6x8x!tt.tile<32x32, f32>, #l1_>) -> memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_>
  %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>
  %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x6x8x!tt.tile<32x32, f32>, #l1_>
  %0 = "ttir.stream_layout"(%arg0, %cb0_alloc) : (memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>, memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>) -> memref<2x4x4x6x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 6 * 4096 + d3 * 4096)>, #l1_>
  %1 = "ttir.stream_layout"(%arg1, %cb1_alloc) : (memref<4x4x6x8x!tt.tile<32x32, f32>, #l1_>, memref<2x4x6x8x!tt.tile<32x32, f32>, #l1_>) -> memref<4x4x6x8x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 8 * 4096 + d3 * 4096)>, #l1_>
  // CHECK: "ttir.generic"([[lhs:%[a-z0-9_]+]], [[rhs:%[a-z0-9_]+]], [[out:%[a-z0-9_]+]])
  "ttir.generic"(%0, %1, %alloc) <{grid = #tt.grid<2x4>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>, operand_cb_mapping = array<i64>}> ({
  // Look for 4 regions, one for each operand and one for the compute
  // Operand 0 (input)
  // CHECK: ^datamovement0
  // CHECK: ttir.dma [[lhs]] #map1, %cb0
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
  // CHECK: ttir.dma [[rhs]] #map2, %cb1
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
  ^bb0(%cb0: memref<4x6x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!tt.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<4x6x!tt.tile<32x32, f32>, #l1_>, memref<6x8x!tt.tile<32x32, f32>, #l1_>, memref<4x8x!tt.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x4x6x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 6 * 4096 + d3 * 4096)>, #l1_>, memref<4x4x6x8x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 8 * 4096 + d3 * 4096)>, #l1_>, memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_>
}

func.func @tilize(%arg0: memref<2x4x128x192xf32, #l1_>) -> memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{grid = #tt.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>, operand_cb_mapping = array<i64>}> ({
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
  ^compute(%cb0: memref<128x192xf32, #l1_>, %cb1: memref<4x6x!tt.tile<32x32, f32>, #l1_>):
    "ttir.tile_tilize_block"(%cb0, %cb1) : (memref<128x192xf32, #l1_>, memref<4x6x!tt.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x128x192xf32, #l1_>, memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>
}

func.func @untilize(%arg0: memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>) -> memref<2x4x128x192xf32, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x128x192xf32, #l1_>
  "ttir.generic"(%arg0, %alloc) <{grid = #tt.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>, operand_cb_mapping = array<i64>}> ({
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
  ^compute(%cb0: memref<4x6x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<128x192xf32, #l1_>):
    "ttir.tile_untilize_block"(%cb0, %cb1) : (memref<4x6x!tt.tile<32x32, f32>, #l1_>, memref<128x192xf32, #l1_>) -> ()
  }) : (memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>, memref<2x4x128x192xf32, #l1_>) -> ()
  return %alloc : memref<2x4x128x192xf32, #l1_>
}
