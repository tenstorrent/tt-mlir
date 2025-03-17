// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for generic operation.

#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op Output grid shape must match the GenericOp grid shape

func.func @matmul(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{grid = #tt.grid<2x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
}

// -----

#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op GenericOp region must have at least as many arguments as the number of top-level operands

func.func @matmul(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
}

// -----

#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op GenericOp region arguments must be of MemRefType

func.func @matmul(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg4: tensor<2x4x!tt.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
}

// -----

#l1_ = #tt.memory_space<l1>
#dram_ = #tt.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op GenericOp region argument memory space must match the memory space of the corresponding operand

func.func @matmul(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #dram_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #dram_>
  "ttir.generic"(%arg0, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #dram_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #dram_>
}

// -----

#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op GenericOp region argument shape must match the shape of the corresponding operand

func.func @matmul(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg2: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
}

// -----

#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op all indexing maps must have the same number of dimensions

func.func @matmul(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map1], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg2: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
}
