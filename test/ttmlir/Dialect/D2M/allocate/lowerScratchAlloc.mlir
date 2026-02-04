// UNSUPPORTED: true
// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-scratch-allocate %s | FileCheck %s

// Type aliases
!tile_bf16 = !ttcore.tile<32x32, bf16>
!tile_f32  = !ttcore.tile<32x32, f32>

!scratch_1d_bf16 = memref<8x!tile_bf16, #ttcore.memory_space<l1>>
!scratch_2d_bf16 = memref<4x4x!tile_bf16, #ttcore.memory_space<l1>>
!scratch_1d_f32  = memref<8x!tile_f32, #ttcore.memory_space<l1>>

// Scratch buffer type (byte buffer)
!scratch_buffer = memref<65536xi8, #ttcore.memory_space<l1>>

// -----
// Test: Single scratch allocation
// CHECK-LABEL: func @single_scratch
func.func @single_scratch() {
  %buf = memref.alloc() : !scratch_buffer
  d2m.scratch_init %buf : !scratch_buffer
  // CHECK: %[[BUF:.*]] = memref.alloc() : memref<65536xi8, #l1>
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[VIEW:.*]] = memref.view %[[BUF]][%[[C0]]][] : memref<65536xi8, #l1> to memref<8x!ttcore.tile<32x32, bf16>, #l1>
  %scratch = d2m.scratch_allocate {slot = 0 : i64} : !scratch_1d_bf16
  // CHECK-NOT: d2m.scratch_init
  // CHECK-NOT: d2m.scratch_allocate
  return
}

// -----
// Test: Multiple scratch allocations (sequential offsets)
// CHECK-LABEL: func @multiple_scratch
func.func @multiple_scratch() {
  %buf = memref.alloc() : !scratch_buffer
  d2m.scratch_init %buf : !scratch_buffer
  // CHECK: %[[BUF:.*]] = memref.alloc() : memref<65536xi8, #l1>
  // First allocation at offset 0
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: memref.view %[[BUF]][%[[C0]]][] : memref<65536xi8, #l1> to memref<8x!ttcore.tile<32x32, bf16>, #l1>
  %scratch0 = d2m.scratch_allocate {slot = 0 : i64} : !scratch_1d_bf16
  // Second allocation at offset 16384 (8 tiles * 2048 bytes/tile)
  // CHECK: %[[C16384:.*]] = arith.constant 16384 : index
  // CHECK: memref.view %[[BUF]][%[[C16384]]][] : memref<65536xi8, #l1> to memref<8x!ttcore.tile<32x32, bf16>, #l1>
  %scratch1 = d2m.scratch_allocate {slot = 1 : i64} : !scratch_1d_bf16
  return
}

// -----
// Test: Mixed types (bf16 and f32)
// CHECK-LABEL: func @mixed_types
func.func @mixed_types() {
  %buf = memref.alloc() : !scratch_buffer
  d2m.scratch_init %buf : !scratch_buffer
  // CHECK: %[[BUF:.*]] = memref.alloc()
  // bf16: 8 tiles * 2048 bytes = 16384 bytes
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: memref.view %[[BUF]][%[[C0]]][] : memref<65536xi8, #l1> to memref<8x!ttcore.tile<32x32, bf16>, #l1>
  %scratch_bf16 = d2m.scratch_allocate {slot = 0 : i64} : !scratch_1d_bf16
  // f32: 8 tiles * 4096 bytes = 32768 bytes, starts at 16384
  // CHECK: %[[C16384:.*]] = arith.constant 16384 : index
  // CHECK: memref.view %[[BUF]][%[[C16384]]][] : memref<65536xi8, #l1> to memref<8x!ttcore.tile<32x32, f32>, #l1>
  %scratch_f32 = d2m.scratch_allocate {slot = 1 : i64} : !scratch_1d_f32
  return
}

// -----
// Test: 2D scratch allocation
// CHECK-LABEL: func @scratch_2d
func.func @scratch_2d() {
  %buf = memref.alloc() : !scratch_buffer
  d2m.scratch_init %buf : !scratch_buffer
  // CHECK: %[[BUF:.*]] = memref.alloc()
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: memref.view %[[BUF]][%[[C0]]][] : memref<65536xi8, #l1> to memref<4x4x!ttcore.tile<32x32, bf16>, #l1>
  %scratch = d2m.scratch_allocate {slot = 0 : i64} : !scratch_2d_bf16
  return
}

// -----
// Test: Scratch with actual load/store usage
// CHECK-LABEL: func @scratch_with_usage
func.func @scratch_with_usage(%arg0: !tile_bf16) -> !tile_bf16 {
  %buf = memref.alloc() : !scratch_buffer
  d2m.scratch_init %buf : !scratch_buffer
  // CHECK: %[[BUF:.*]] = memref.alloc()
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[VIEW:.*]] = memref.view %[[BUF]][%[[C0]]][]
  %scratch = d2m.scratch_allocate {slot = 0 : i64} : !scratch_1d_bf16

  %idx = arith.constant 0 : index
  // CHECK: memref.store %arg0, %[[VIEW]][%{{.*}}]
  memref.store %arg0, %scratch[%idx] : !scratch_1d_bf16
  // CHECK: %[[LOADED:.*]] = memref.load %[[VIEW]][%{{.*}}]
  %loaded = memref.load %scratch[%idx] : !scratch_1d_bf16

  // CHECK: return %[[LOADED]]
  return %loaded : !tile_bf16
}

// -----
// Test: No scratch_init, no scratch_allocate (no-op)
// CHECK-LABEL: func @no_scratch
func.func @no_scratch() {
  // CHECK-NOT: memref.view
  return
}
