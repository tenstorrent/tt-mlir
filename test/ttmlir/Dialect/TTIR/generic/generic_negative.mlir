// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for generic operation.

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op region must have at least as many arguments as the number of top-level operands

func.func @matmul(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
}

// -----

#l1_ = #ttcore.memory_space<l1>
#dram_ = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op region argument memory space must match the memory space of the corresponding operand

func.func @matmul(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #dram_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #dram_>
  "ttir.generic"(%arg0, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #dram_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #dram_>
}

// -----

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op region argument shape must match the shape of the corresponding operand

func.func @matmul(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>, threads = [#ttir.thread<compute>]}> ({
  ^bb0(%arg2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
}

// -----

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op all regions must have the same number of arguments

func.func @matmul(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<datamovement>, #ttir.thread<compute>], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
  }, {
  ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg5: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
}

// -----

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op all indexing maps must have the same number of dimensions

func.func @matmul(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
}

// -----

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op all regions must have the same argument types

func.func @matmul(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#ttir.thread<datamovement>, #ttir.thread<compute>], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg5: !ttir.semaphore):
  }, {
  ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg5: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
}

// -----

#l1_ = #ttcore.memory_space<l1>
#lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

// CHECK: error: 'ttir.generic' op grid shape mismatch between operand[1] grid_shape=[1, 2] and operand[2] grid_shape=[1, 1] at affine dim d1

func.func @matmul(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, %arg1: memref<1x2x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#lhs, #rhs, #out], iterator_types = [#parallel, #parallel, #reduction], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x2x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

// -----

#l1_ = #ttcore.memory_space<l1>
#lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

// CHECK: error: 'ttir.generic' op grid shape mismatch between operand[0] grid_shape=[1, 1] and operand[1] grid_shape=[2, 1] at affine dim d2

func.func @matmul(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, %arg1: memref<2x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#lhs, #rhs, #out], iterator_types = [#parallel, #parallel, #reduction], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg2: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<2x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

// -----

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'ttir.tile_matmul_block' op expected to be in a compute region

func.func @matmul(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>, threads = [#ttir.thread<datamovement>]}> ({
  ^bb0(%arg2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
  "ttir.tile_matmul_block"(%arg2, %arg4, %arg4) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

// -----

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op grid dim unexpected for operand[0] grid_shape=[1, 1] expected grid_shape=[1, 3] at affine dim d1

func.func @matmul(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{block_factors = [1, 3], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>, threads = [#ttir.thread<compute>]}> ({
  ^bb0(%arg2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
  "ttir.tile_matmul_block"(%arg2, %arg4, %arg4) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

// -----

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK: error: 'ttir.generic' op grid dim unexpected for operand[0] grid_shape=[1, 2] expected grid_shape=[1, 6] at affine dim d1

func.func @matmul(%arg0: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{block_factors = [1, 3], grid = #ttcore.grid<1x2>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>, threads = [#ttir.thread<compute>]}> ({
  ^bb0(%arg2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %arg4: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
  "ttir.tile_matmul_block"(%arg2, %arg4, %arg4) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}
