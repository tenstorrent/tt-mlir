// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-scratch-allocate 2>&1 -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test type aliases for convenience
!tile_bf16 = !ttcore.tile<32x32, bf16>
!tile_f32  = !ttcore.tile<32x32, f32>

!scratch_1d_bf16 = memref<8x!tile_bf16, #ttcore.memory_space<l1>>
!scratch_1d_f32  = memref<8x!tile_f32, #ttcore.memory_space<l1>>
!scratch_2d_bf16 = memref<4x4x!tile_bf16, #ttcore.memory_space<l1>>
!scratch_2d_f32  = memref<2x8x!tile_f32, #ttcore.memory_space<l1>>
!scratch_3d_bf16 = memref<2x2x4x!tile_bf16, #ttcore.memory_space<l1>>

// -----
// Test: Single 1D scratch allocation
// CHECK-LABEL: func @single_1d_scratch
func.func @single_1d_scratch() {
  // CHECK: %[[MASTER:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<8x!ttcore.tile<32x32, bf16>, #l1>
  // CHECK: memref.subview %[[MASTER]][0] [8] [1]
  // CHECK: memref.reinterpret_cast
  %scratch = d2m.scratch_allocate {slot = 0 : i64} : !scratch_1d_bf16
  // CHECK: memref.dealloc %[[MASTER]]
  return
}

// -----
// Test: Multiple 1D scratch allocations (same type)
// CHECK-LABEL: func @multiple_1d_same_type
func.func @multiple_1d_same_type() {
  // CHECK: %[[MASTER:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<24x!ttcore.tile<32x32, bf16>, #l1>
  // CHECK: memref.subview %[[MASTER]][0] [8] [1]
  // CHECK: memref.reinterpret_cast
  %scratch0 = d2m.scratch_allocate {slot = 0 : i64} : !scratch_1d_bf16
  // CHECK: memref.subview %[[MASTER]][8] [8] [1]
  // CHECK: memref.reinterpret_cast
  %scratch1 = d2m.scratch_allocate {slot = 1 : i64} : !scratch_1d_bf16
  // CHECK: memref.subview %[[MASTER]][16] [8] [1]
  // CHECK: memref.reinterpret_cast
  %scratch2 = d2m.scratch_allocate {slot = 2 : i64} : !scratch_1d_bf16
  // CHECK: memref.dealloc %[[MASTER]]
  return
}

// -----
// Test: Multiple allocations of different element types
// CHECK-LABEL: func @multiple_different_types
func.func @multiple_different_types() {
  // CHECK-DAG: memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<8x!ttcore.tile<32x32, bf16>, #l1>
  // CHECK-DAG: memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<8x!ttcore.tile<32x32, f32>, #l1>
  %scratch_bf16 = d2m.scratch_allocate {slot = 0 : i64} : !scratch_1d_bf16
  %scratch_f32 = d2m.scratch_allocate {slot = 1 : i64} : !scratch_1d_f32
  // CHECK-COUNT-2: memref.dealloc
  return
}

// -----
// Test: 2D scratch allocation (requires reinterpret_cast for reshaping)
// CHECK-LABEL: func @scratch_2d
func.func @scratch_2d() {
  // CHECK: %[[MASTER:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<16x!ttcore.tile<32x32, bf16>, #l1>
  // CHECK: %[[SV:.*]] = memref.subview %[[MASTER]][0] [16] [1]
  // CHECK: memref.reinterpret_cast %[[SV]]{{.*}}to offset: [0], sizes: [4, 4], strides: [4, 1]
  %scratch = d2m.scratch_allocate {slot = 0 : i64} : !scratch_2d_bf16
  // CHECK: memref.dealloc %[[MASTER]]
  return
}

// -----
// Test: Multiple 2D scratch allocations
// CHECK-LABEL: func @multiple_2d_scratch
func.func @multiple_2d_scratch() {
  // CHECK: %[[MASTER:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<32x!ttcore.tile<32x32, bf16>, #l1>
  // CHECK: %[[SV0:.*]] = memref.subview %[[MASTER]][0] [16] [1]
  // CHECK: memref.reinterpret_cast %[[SV0]]{{.*}}to offset: [0], sizes: [4, 4], strides: [4, 1]
  %scratch0 = d2m.scratch_allocate {slot = 0 : i64} : !scratch_2d_bf16
  // CHECK: %[[SV1:.*]] = memref.subview %[[MASTER]][16] [16] [1]
  // CHECK: memref.reinterpret_cast %[[SV1]]{{.*}}to offset: [0], sizes: [4, 4], strides: [4, 1]
  %scratch1 = d2m.scratch_allocate {slot = 1 : i64} : !scratch_2d_bf16
  // CHECK: memref.dealloc %[[MASTER]]
  return
}

// -----
// Test: 3D scratch allocation
// CHECK-LABEL: func @scratch_3d
func.func @scratch_3d() {
  // CHECK: %[[MASTER:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<16x!ttcore.tile<32x32, bf16>, #l1>
  // CHECK: %[[SV:.*]] = memref.subview %[[MASTER]][0] [16] [1]
  // CHECK: memref.reinterpret_cast %[[SV]]{{.*}}to offset: [0], sizes: [2, 2, 4], strides: [8, 4, 1]
  %scratch = d2m.scratch_allocate {slot = 0 : i64} : !scratch_3d_bf16
  // CHECK: memref.dealloc %[[MASTER]]
  return
}

// -----
// Test: No scratch allocations (pass should be a no-op)
// CHECK-LABEL: func @no_scratch_allocations
func.func @no_scratch_allocations() {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.subview
  // CHECK: return
  return
}

// -----
// Test: Multiple functions with separate scratch regions
// CHECK-LABEL: func @func_with_scratch_a
func.func @func_with_scratch_a() {
  // CHECK: memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<8x!ttcore.tile<32x32, bf16>, #l1>
  %scratch = d2m.scratch_allocate {slot = 0 : i64} : !scratch_1d_bf16
  // CHECK: memref.dealloc
  return
}

// CHECK-LABEL: func @func_with_scratch_b
func.func @func_with_scratch_b() {
  // Each function gets its own master scratchpad
  // CHECK: memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<16x!ttcore.tile<32x32, f32>, #l1>
  %scratch = d2m.scratch_allocate {slot = 0 : i64} : !scratch_2d_f32
  // CHECK: memref.dealloc
  return
}

// -----
// Test: Mixed 1D and 2D allocations of same element type
// CHECK-LABEL: func @mixed_dimensions_same_type
func.func @mixed_dimensions_same_type() {
  // Total elements: 8 (1D) + 16 (2D) = 24
  // CHECK: %[[MASTER:.*]] = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64} : memref<24x!ttcore.tile<32x32, bf16>, #l1>
  // CHECK: memref.subview %[[MASTER]][0] [8] [1]
  %scratch_1d = d2m.scratch_allocate {slot = 0 : i64} : !scratch_1d_bf16
  // CHECK: memref.subview %[[MASTER]][8] [16] [1]
  // CHECK: memref.reinterpret_cast{{.*}}to offset: [0], sizes: [4, 4], strides: [4, 1]
  %scratch_2d = d2m.scratch_allocate {slot = 1 : i64} : !scratch_2d_bf16
  // CHECK: memref.dealloc %[[MASTER]]
  return
}

// Test: Mixed types and dimensions
// CHECK-LABEL: func @mixed_types_and_dimensions
func.func @mixed_types_and_dimensions() {
  // Should create two master scratchpads: one for bf16, one for f32
  // CHECK-DAG: memref.alloc(){{.*}}: memref<24x!ttcore.tile<32x32, bf16>, #l1>
  // CHECK-DAG: memref.alloc(){{.*}}: memref<16x!ttcore.tile<32x32, f32>, #l1>
  %scratch_1d_bf16 = d2m.scratch_allocate {slot = 0 : i64} : !scratch_1d_bf16
  %scratch_2d_f32 = d2m.scratch_allocate {slot = 1 : i64} : !scratch_2d_f32
  %scratch_2d_bf16 = d2m.scratch_allocate {slot = 2 : i64} : !scratch_2d_bf16
  // CHECK-COUNT-2: memref.dealloc
  return
}

// -----
// Test: 1D scratch with store and load operations
// CHECK-LABEL: func @scratch_1d_with_store_load
func.func @scratch_1d_with_store_load(%arg0: !tile_bf16) -> !tile_bf16 {
  // CHECK: %[[MASTER:.*]] = memref.alloc()
  // CHECK: %[[SV:.*]] = memref.subview %[[MASTER]]
  // CHECK: %[[RC:.*]] = memref.reinterpret_cast %[[SV]]
  %scratch = d2m.scratch_allocate {slot = 0 : i64} : !scratch_1d_bf16

  // Store a tile into scratch
  // CHECK: memref.store %arg0, %[[RC]][%c0]
  %c0 = arith.constant 0 : index
  memref.store %arg0, %scratch[%c0] : !scratch_1d_bf16

  // Load it back
  // CHECK: %[[LOADED:.*]] = memref.load %[[RC]][%c0]
  %loaded = memref.load %scratch[%c0] : !scratch_1d_bf16

  // CHECK: memref.dealloc %[[MASTER]]
  // CHECK: return %[[LOADED]]
  return %loaded : !tile_bf16
}

// -----
// Test: 2D scratch with affine store and load operations
// CHECK-LABEL: func @scratch_2d_with_affine_ops
func.func @scratch_2d_with_affine_ops(%arg0: !tile_bf16) -> !tile_bf16 {
  // CHECK: %[[MASTER:.*]] = memref.alloc()
  // CHECK: %[[SV:.*]] = memref.subview %[[MASTER]]
  // CHECK: %[[RC:.*]] = memref.reinterpret_cast %[[SV]]{{.*}}sizes: [4, 4]
  %scratch = d2m.scratch_allocate {slot = 0 : i64} : !scratch_2d_bf16

  // Store a tile into 2D scratch at [1, 2]
  // CHECK: affine.store %arg0, %[[RC]][1, 2]
  affine.store %arg0, %scratch[1, 2] : !scratch_2d_bf16

  // Load it back from [1, 2]
  // CHECK: %[[LOADED:.*]] = affine.load %[[RC]][1, 2]
  %loaded = affine.load %scratch[1, 2] : !scratch_2d_bf16

  // CHECK: memref.dealloc %[[MASTER]]
  // CHECK: return %[[LOADED]]
  return %loaded : !tile_bf16
}
