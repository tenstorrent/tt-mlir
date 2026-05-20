// RUN: ttmlir-opt %s | FileCheck %s

// Parse/print round-trip for d2m.gather_core (memref + tensor forms) and for
// the cross-core local-L1 form of d2m.dma_read (shard-level + fully-indexed).
// All ops live inside func.func per the D2M_GenericParent interface.

#l1_ = #ttcore.memory_space<l1>

//===----------------------------------------------------------------------===//
// d2m.gather_core
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_gather_core_memref
func.func @test_gather_core_memref(
    %src: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
    %dst: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // CHECK: d2m.gather_core %{{.*}} into %{{.*}} group[%{{.*}}, %{{.*}}] shape[%{{.*}}, %{{.*}}] collector[%{{.*}}, %{{.*}}] : memref<2x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>, memref<2x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>
  d2m.gather_core %src into %dst
    group [%c0, %c0] shape [%c1, %c2] collector [%c0, %c0]
    : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
      memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

  return
}

// CHECK-LABEL: @test_gather_core_tensor
func.func @test_gather_core_tensor(
    %src: tensor<2x4x!ttcore.tile<32x32, f32>>,
    %dst: tensor<2x4x!ttcore.tile<32x32, f32>>) -> tensor<2x4x!ttcore.tile<32x32, f32>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // Tensor form is DPS: the result aliases %dst.
  // CHECK: %{{.*}} = d2m.gather_core %{{.*}} into %{{.*}} group[%{{.*}}, %{{.*}}] shape[%{{.*}}, %{{.*}}] collector[%{{.*}}, %{{.*}}] : tensor<{{.*}}>, tensor<{{.*}}> -> tensor<{{.*}}>
  %0 = d2m.gather_core %src into %dst
    group [%c0, %c0] shape [%c1, %c2] collector [%c0, %c0]
    : tensor<2x4x!ttcore.tile<32x32, f32>>,
      tensor<2x4x!ttcore.tile<32x32, f32>>
    -> tensor<2x4x!ttcore.tile<32x32, f32>>

  return %0 : tensor<2x4x!ttcore.tile<32x32, f32>>
}

//===----------------------------------------------------------------------===//
// d2m.gather_core: explicit CB forms (introduced by SplitUnifiedThread)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_gather_core_src_cb_dst_cb
func.func @test_gather_core_src_cb_dst_cb(
    %srcCb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>,
    %dstCb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // `from` toggles src-side to the CB; dst type also being a CB toggles dst-side.
  // CHECK: d2m.gather_core from %{{.*}} into %{{.*}} group[%{{.*}}, %{{.*}}] shape[%{{.*}}, %{{.*}}] collector[%{{.*}}, %{{.*}}] : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>>, !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>>
  d2m.gather_core from %srcCb into %dstCb
    group [%c0, %c0] shape [%c1, %c2] collector [%c0, %c0]
    : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>,
      !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>

  return
}

// CHECK-LABEL: @test_gather_core_src_cb_dst_memref
func.func @test_gather_core_src_cb_dst_memref(
    %srcCb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>,
    %dst: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // `from` keyword on src, no keyword on dst; dst type is a plain memref.
  // CHECK: d2m.gather_core from %{{.*}} into %{{.*}} group[%{{.*}}, %{{.*}}] shape[%{{.*}}, %{{.*}}] collector[%{{.*}}, %{{.*}}] : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>>, memref<2x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>
  d2m.gather_core from %srcCb into %dst
    group [%c0, %c0] shape [%c1, %c2] collector [%c0, %c0]
    : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>,
      memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

  return
}

// CHECK-LABEL: @test_gather_core_src_memref_dst_cb
func.func @test_gather_core_src_memref_dst_cb(
    %src: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
    %dstCb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // No `from` keyword on src; dst type is a CB.
  // CHECK: d2m.gather_core %{{.*}} into %{{.*}} group[%{{.*}}, %{{.*}}] shape[%{{.*}}, %{{.*}}] collector[%{{.*}}, %{{.*}}] : memref<2x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>, !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>>
  d2m.gather_core %src into %dstCb
    group [%c0, %c0] shape [%c1, %c2] collector [%c0, %c0]
    : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
      !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>

  return
}

//===----------------------------------------------------------------------===//
// d2m.dma_read core[...]
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_dma_read_cross_core_shard_level
func.func @test_dma_read_cross_core_shard_level(
    %src: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
    %dst: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // Shard-level cross-core read: empty srcIndices, empty dstIndices, numElems=0.
  // CHECK: d2m.dma_read %{{.*}}[] core[%{{.*}}, %{{.*}}], %{{.*}}, <0> : (memref<{{.*}}>, memref<{{.*}}>) -> !d2m.mem_tx<read>
  %tx = d2m.dma_read %src[] core[%c1, %c2], %dst, <0>
        : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
           memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx<read>
  d2m.dma_wait %tx : !d2m.mem_tx<read>
  return
}

// CHECK-LABEL: @test_dma_read_cross_core_fully_indexed
func.func @test_dma_read_cross_core_fully_indexed(
    %src: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
    %dst: memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // Fully-indexed cross-core read: single linearized offset on each side, numElems>0.
  // CHECK: d2m.dma_read %{{.*}}[%{{.*}}] core[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], <8> : (memref<{{.*}}>, memref<{{.*}}>) -> !d2m.mem_tx<read>
  %tx = d2m.dma_read %src[%c0] core[%c1, %c2], %dst[%c0], <8>
        : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
           memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx<read>
  d2m.dma_wait %tx : !d2m.mem_tx<read>
  return
}
