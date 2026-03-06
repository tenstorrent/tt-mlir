// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --convert-d2m-to-ttnn -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Test for d2m.create_global_semaphore and d2m.reset_global_semaphore conversion to TTNN.
// This test uses --convert-d2m-to-ttnn directly instead of the full backend pipeline
// to avoid canonicalization removing the ops.

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @test_global_semaphore
  func.func @test_global_semaphore() {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK-NOT: memref.alloc
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #ttcore.memory_space<l1>>
    // CHECK: %[[SEM:.*]] = "ttnn.create_global_semaphore"() <{core_range = #ttnn.core_range<(0,0), (7,7)>, initial_value = 0 : ui32}> : () -> !ttnn.global_semaphore
    %sem = d2m.create_global_semaphore(%alloc) {value = 0 : ui32}
      : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #ttcore.memory_space<l1>> -> !d2m.global_semaphore
    // CHECK: "ttnn.reset_global_semaphore"(%[[SEM]]) <{value = 0 : ui32}> : (!ttnn.global_semaphore) -> ()
    d2m.reset_global_semaphore(%sem) {value = 0 : ui32} : !d2m.global_semaphore
    // CHECK-NOT: memref.dealloc
    memref.dealloc %alloc : memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #ttcore.memory_space<l1>>
    return
  }
}
