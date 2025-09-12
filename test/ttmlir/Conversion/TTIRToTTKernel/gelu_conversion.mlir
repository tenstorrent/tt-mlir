// RUN: ttmlir-opt --ttcore-register-device --ttir-insert-dst-register-access --lower-affine --ttir-generic-linearize-memref --lower-affine --convert-ttir-to-ttkernel --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @test_gelu_lowering
  func.func @test_gelu_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_gelu
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.gelu_tile_init
    // CHECK: ttkernel.gelu_tile
    %1 = "ttir.tile_gelu"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }
}
