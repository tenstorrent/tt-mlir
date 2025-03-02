// RUN: ttmlir-opt --ttir-load-system-desc --ttir-implicit-device --ttir-generic-datamovement %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>

func.func @add(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 2, 0, 1>, operand_cb_mapping = array<i64>}> ({
  // Look for 4 regions, one for each operand and one for the compute
  // Operand 0 (input)
  // CHECK: ^bb0
  // CHECK-NEXT: ttir.yield
  // Operand 1 (input)
  // CHECK: ^bb0
  // CHECK-NEXT: ttir.yield
  // Operand 2 (output)
  // CHECK: ^bb0
  // CHECK-NEXT: ttir.await
  // Compute
  // CHECK: ^bb0
  // CHECK: ttir.await
  // CHECK: ttir.tile_add
  // CHECK: ttir.yield
  ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        %0 = affine.load %arg2[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %1 = affine.load %arg3[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %2 = "ttir.tile_add"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
        affine.store %2, %arg4[%arg5, %arg6] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
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
  "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 0, 1>, operand_cb_mapping = array<i64>}> ({
  ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%arg2, %arg3, %arg4) : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x4x2x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
}

func.func @matmul_single_core_stream(%arg0: memref<1x2x2x2x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<2x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
  %cb0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
  %cb1 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
  %0 = "ttir.stream_layout"(%arg0, %cb0) : (memref<1x2x2x2x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> memref<1x2x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), stream>, #l1_>
  %1 = "ttir.stream_layout"(%arg1, %cb1) : (memref<2x1x2x2x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> memref<2x1x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), stream>, #l1_>
  "ttir.generic"(%0, %1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 0, 1>, operand_cb_mapping = array<i64>}> ({
  ^bb0(%arg2: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%arg2, %arg3, %arg4) : (memref<2x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x2x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), stream>, #l1_>, memref<2x1x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), stream>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
}

func.func @matmul_multi_core(%arg0: memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<4x4x6x8x!tt.tile<32x32, f32>, #l1_>) -> memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_>
  %cb0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>
  %cb1 = memref.alloc() {alignment = 64 : i64} : memref<2x4x6x8x!tt.tile<32x32, f32>, #l1_>
  %0 = "ttir.stream_layout"(%arg0, %cb0) : (memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>, memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>) -> memref<2x4x4x6x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), stream>, #l1_>
  %1 = "ttir.stream_layout"(%arg1, %cb1) : (memref<4x4x6x8x!tt.tile<32x32, f32>, #l1_>, memref<2x4x6x8x!tt.tile<32x32, f32>, #l1_>) -> memref<4x4x6x8x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), stream>, #l1_>
  "ttir.generic"(%0, %1, %alloc) <{grid = #tt.grid<2x4>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 0, 1>, operand_cb_mapping = array<i64>}> ({
  ^bb0(%arg2: memref<4x6x!tt.tile<32x32, f32>, #l1_>, %arg3: memref<6x8x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<4x8x!tt.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%arg2, %arg3, %arg4) : (memref<4x6x!tt.tile<32x32, f32>, #l1_>, memref<6x8x!tt.tile<32x32, f32>, #l1_>, memref<4x8x!tt.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x4x6x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), stream>, #l1_>, memref<4x4x6x8x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2, d3), stream>, #l1_>, memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_>
}
