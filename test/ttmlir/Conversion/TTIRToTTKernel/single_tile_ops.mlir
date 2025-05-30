// RUN: ttmlir-opt --tt-register-device --ttir-insert-dst-register-access --lower-affine --convert-ttir-to-ttkernel --canonicalize %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #tt.memory_space<l1>

module {
  //===----------------------------------------------------------------------===//
  // TTIR FPU operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: func.func @test_matmul_lowering
  func.func @test_matmul_lowering(%arg0: memref<1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    %2 = affine.load %arg2[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_matmul
    // CHECK: ttkernel.mm_init
    // CHECK: ttkernel.mm_init_short
    // CHECK: ttkernel.matmul_tiles
    %3 = "ttir.tile_matmul"(%0, %1, %2) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %3, %arg2[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_add_lowering
  func.func @test_add_lowering(%arg0: memref<1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_add
    // CHECK: ttkernel.binary_op_init_common
    // CHECK: ttkernel.add_tiles_init
    // CHECK: ttkernel.add_tiles
    %2 = "ttir.tile_add"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %2, %arg2[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_mul_lowering
  func.func @test_mul_lowering(%arg0: memref<1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_mul
    // CHECK: ttkernel.binary_op_init_common
    // CHECK: ttkernel.mul_tiles_init
    // CHECK: ttkernel.mul_tiles
    %2 = "ttir.tile_mul"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %2, %arg2[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    return
  }

  //===----------------------------------------------------------------------===//
  // TTIR SFPU operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: func.func @test_max_lowering
  func.func @test_max_lowering(%arg0: memref<1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_max
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init
    // CHECK: ttkernel.copy_tile
    // CHECK: ttkernel.copy_tile_init
    // CHECK: ttkernel.copy_tile
    // CHECK: ttkernel.max_tile_init
    // CHECK: ttkernel.max_tile
    %2 = "ttir.tile_maximum"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %2, %arg2[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    return
  }
}
