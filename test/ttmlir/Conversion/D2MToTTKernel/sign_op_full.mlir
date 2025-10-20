// UNSUPPORTED: true
// TODO: Need to figure out the correct pass pipeline to go from TTIR all the way to TTKernel.
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --convert-d2m-to-ttkernel -o %t %s
// RUN: FileCheck %s --input-file=%t

!ttype_f32 = tensor<128x96xf32>
!ttype_f16 = tensor<128x96xf16>

module {
  // CHECK-LABEL: func @test_sign_f16
  func.func @test_sign_f16(%arg: !ttype_f16, %out: !ttype_f16) -> (!ttype_f16) {
    // CHECK-NOT: d2m.tile_sign
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init
    // CHECK: ttkernel.copy_tile
    // CHECK: ttkernel.sign_tile_init
    // CHECK: ttkernel.sign_tile
    %0 = "ttir.sign"(%arg, %out) : (!ttype_f16, !ttype_f16) -> !ttype_f16
    return %0 : !ttype_f16
  }

  // CHECK-LABEL: func @test_sign_f32
  func.func @test_sign_f32(%arg: !ttype_f32, %out: !ttype_f32) -> (!ttype_f32) {
    // CHECK-NOT: d2m.tile_sign
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init
    // CHECK: ttkernel.copy_tile
    // CHECK: ttkernel.typecast_tile
    // CHECK: ttkernel.sign_tile_init
    // CHECK: ttkernel.sign_tile
    // CHECK: ttkernel.typecast_tile
    %0 = "ttir.sign"(%arg, %out) : (!ttype_f32, !ttype_f32) -> !ttype_f32
    return %0 : !ttype_f32
  }
}
