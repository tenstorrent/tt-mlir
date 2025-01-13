// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @test_empty_int8() -> tensor<64x128xi8> {
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xi8>}> : () -> tensor<64x128xi8>
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    return %0 : tensor<64x128xi8>
  }

  func.func @test_empty_int16() -> tensor<64x128xi16> {
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xi16>}> : () -> tensor<64x128xi16>
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    return %0 : tensor<64x128xi16>
  }

  func.func @test_empty_int() -> tensor<64x128xi32> {
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    return %0 : tensor<64x128xi32>
  }

  func.func @test_empty_bfloat16() -> tensor<64x128xbf16> {
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<64x128xbf16>}> : () -> tensor<64x128xbf16>
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    return %0 : tensor<64x128xbf16>
  }

  func.func @test_empty_float() -> tensor<64x128xf32> {
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    return %0 : tensor<64x128xf32>
  }

  func.func @test_full_int8() -> tensor<64x128xi8> {
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fillValue = 1.000000e+00 : f32
    // CHECK-SAME: tensor<64x128xi8
    %0 = "ttir.constant"() <{value = dense<1> : tensor<64x128xi8>}> : () -> tensor<64x128xi8>
    return %0 : tensor<64x128xi8>
  }

  func.func @test_full_int16() -> tensor<64x128xi16> {
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fillValue = 1.000000e+00 : f32
    // CHECK-SAME: tensor<64x128xi16
    %0 = "ttir.constant"() <{value = dense<1> : tensor<64x128xi16>}> : () -> tensor<64x128xi16>
    return %0 : tensor<64x128xi16>
  }

  func.func @test_full_int() -> tensor<64x128xi32> {
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fillValue = 1.000000e+00 : f32
    // CHECK-SAME: tensor<64x128xi32
    %0 = "ttir.constant"() <{value = dense<1> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    return %0 : tensor<64x128xi32>
  }

  func.func @test_full_bfloat16() -> tensor<64x128xbf16> {
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fillValue = 1.000000e+00 : f32
    // CHECK-SAME: tensor<64x128xbf16
    %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<64x128xbf16>}> : () -> tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }

  func.func @test_full_float() -> tensor<64x128xf32> {
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fillValue = 1.000000e+00 : f32
    // CHECK-SAME: tensor<64x128xf32
    %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
