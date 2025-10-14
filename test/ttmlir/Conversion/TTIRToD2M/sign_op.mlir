// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m -o %t %s
// RUN: FileCheck %s --input-file=%t

!ttype_f32 = tensor<128x96xf32>
!ttype_f16 = tensor<128x96xf16>
!ttype_bf16 = tensor<128x96xbf16>

module {
  // CHECK-LABEL: func @test_sign_f32
  func.func @test_sign_f32(%arg: !ttype_f32, %out: !ttype_f32) -> (!ttype_f32) {
    // CHECK: linalg.generic
    // CHECK: ^bb0([[IN:%.*]]: !ttcore.tile<32x32, f32>,
    // CHECK: [[CAST_IN:%.*]] = "d2m.tile_typecast"([[IN]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f16>
    // CHECK: [[SIGN:%.*]] = "d2m.tile_sign"([[CAST_IN]]) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
    // CHECK: [[CAST_OUT:%.*]] = "d2m.tile_typecast"([[SIGN]]) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f32>
    // CHECK: linalg.yield [[CAST_OUT]] : !ttcore.tile<32x32, f32>
    %0 = "ttir.sign"(%arg, %out) : (!ttype_f32, !ttype_f32) -> !ttype_f32
    return %0 : !ttype_f32
  }

  // CHECK-LABEL: func @test_sign_f16
  func.func @test_sign_f16(%arg: !ttype_f16, %out: !ttype_f16) -> (!ttype_f16) {
    // CHECK: linalg.generic
    // CHECK: ^bb0([[IN:%.*]]: !ttcore.tile<32x32, f16>,
    // CHECK-NOT: d2m.tile_typecast
    // CHECK: [[SIGN:%.*]] = "d2m.tile_sign"([[IN]]) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
    // CHECK: linalg.yield [[SIGN]] : !ttcore.tile<32x32, f16>
    %0 = "ttir.sign"(%arg, %out) : (!ttype_f16, !ttype_f16) -> !ttype_f16
    return %0 : !ttype_f16
  }

  // CHECK-LABEL: func @test_sign_bf16
  func.func @test_sign_bf16(%arg: !ttype_bf16, %out: !ttype_bf16) -> (!ttype_bf16) {
    // CHECK: linalg.generic
    // CHECK: ^bb0([[IN:%.*]]: !ttcore.tile<32x32, bf16>,
    // CHECK: [[CAST_IN:%.*]] = "d2m.tile_typecast"([[IN]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, f16>
    // CHECK: [[SIGN:%.*]] = "d2m.tile_sign"([[CAST_IN]]) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
    // CHECK: [[CAST_OUT:%.*]] = "d2m.tile_typecast"([[SIGN]]) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, bf16>
    // CHECK: linalg.yield [[CAST_OUT]] : !ttcore.tile<32x32, bf16>
    %0 = "ttir.sign"(%arg, %out) : (!ttype_bf16, !ttype_bf16) -> !ttype_bf16
    return %0 : !ttype_bf16
  }
}
