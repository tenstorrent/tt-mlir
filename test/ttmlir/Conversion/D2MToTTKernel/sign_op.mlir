// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access --lower-affine --d2m-generic-linearize-memref --lower-affine --convert-d2m-to-ttkernel --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @test_sign_lowering
  func.func @test_sign_lowering(%arg0: memref<1x!ttcore.tile<32x32, f16>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f16>, #l1_>) attributes {d2m.thread = #d2m.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f16>, #l1_>
    // CHECK-NOT: d2m.tile_sign
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %[[DST_IDX:.+]]) :
    // CHECK: ttkernel.sign_tile_init() :
    // CHECK: ttkernel.sign_tile(%[[DST_IDX]])
    %1 = "d2m.tile_sign"(%0) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f16>, #l1_>
    return
  }
}
