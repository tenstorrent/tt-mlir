// RUN: ttmlir-opt %s | FileCheck %s

// This test verifies that various remote_load and remote_store assembly syntaxes
// are parsed correctly for both CB-mode and result-mode variants.

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

//===----------------------------------------------------------------------===//
// RemoteLoadOp Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_remote_load_with_cb
func.func @test_remote_load_with_cb(
    %remote_src: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
    %cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // RemoteLoadOp with CB (no result, no localBuffer) - basic case
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}} : memref<{{.*}}> into !d2m.cb<memref<{{.*}}>
  d2m.remote_load %remote_src[%c0, %c1] into %cb : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>

  return
}

// CHECK-LABEL: @test_remote_load_with_cb_multicast
func.func @test_remote_load_with_cb_multicast(
    %remote_src: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
    %cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // RemoteLoadOp with CB and multicast (low-level multicast, no localBuffer)
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}} mcore[%{{.*}}, %{{.*}}] mshape[%{{.*}}, %{{.*}}] : memref<{{.*}}> into !d2m.cb<memref<{{.*}}>
  d2m.remote_load %remote_src[%c0, %c1] into %cb mcore[%c0, %c0] mshape[%c1, %c2] : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>

  return
}

// CHECK-LABEL: @test_remote_load_with_result
func.func @test_remote_load_with_result(
    %remote_src: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

  // RemoteLoadOp with result (no CB)
  // CHECK: %{{.*}} = d2m.remote_load %{{.*}} %{{.*}}[%{{.*}}, %{{.*}}] : memref<{{.*}}>, memref<{{.*}}> -> memref<{{.*}}>
  %result = d2m.remote_load %buffer %remote_src[%c0, %c1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

  return %result : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
}

// CHECK-LABEL: @test_remote_load_with_result_multicast
func.func @test_remote_load_with_result_multicast(
    %remote_src: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

  // RemoteLoadOp with result and multicast (low-level multicast)
  // CHECK: %{{.*}} = d2m.remote_load %{{.*}} %{{.*}}[%{{.*}}, %{{.*}}] mcore[%{{.*}}, %{{.*}}] mshape[%{{.*}}, %{{.*}}] : memref<{{.*}}>, memref<{{.*}}> -> memref<{{.*}}>
  %result = d2m.remote_load %buffer %remote_src[%c0, %c1] mcore[%c0, %c0] mshape[%c1, %c2] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

  return %result : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
}

//===----------------------------------------------------------------------===//
// RemoteStoreOp Tests - Implicit Form (local buffer)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_remote_store_with_local_buffer
func.func @test_remote_store_with_local_buffer(
    %remote_dst: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
    %local_buffer: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // RemoteStoreOp with local buffer (implicit form) - basic case
  // Result is required but must be unused when using memref types
  // CHECK: %{{.*}} = d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] %{{.*}} : memref<{{.*}}>, memref<{{.*}}> -> memref<{{.*}}>
  %result = d2m.remote_store %remote_dst[%c0, %c1] %local_buffer : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>

  return
}

//===----------------------------------------------------------------------===//
// RemoteStoreOp Tests - Explicit CB Form (circular buffer)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_remote_store_with_cb
func.func @test_remote_store_with_cb(
    %remote_dst: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
    %cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // RemoteStoreOp with CB (explicit CB form) - basic case
  // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %{{.*}} : memref<{{.*}}> from !d2m.cb<memref<{{.*}}>
  d2m.remote_store %remote_dst[%c0, %c1] from %cb : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>

  return
}
