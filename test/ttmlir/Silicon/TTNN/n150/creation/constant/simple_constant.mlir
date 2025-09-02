// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module @sysmem_creation attributes {} {
  func.func @test_empty_int() -> tensor<64x128xi32> {
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    // CHECK: = "ttnn.full"
    return %0 : tensor<64x128xi32>
  }

  func.func @test_empty_uint() -> tensor<64x128xui32> {
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xui32>}> : () -> tensor<64x128xui32>
    // CHECK: = "ttnn.full"
    return %0 : tensor<64x128xui32>
  }

  func.func @test_empty_float() -> tensor<64x128xf32> {
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    // CHECK: = "ttnn.full"
    return %0 : tensor<64x128xf32>
  }

  func.func @test_empty_float_scalar() -> tensor<1x1xf32> {
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
    // CHECK: = "ttnn.full"
    return %0 : tensor<1x1xf32>
  }

  func.func @test_neginf_float() -> tensor<1xf32> {
    %0 = "ttir.constant"() <{value = dense<0xFF800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }

  func.func @test_full_int() -> tensor<64x128xi32> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    // CHECK: = "ttnn.full"
    return %0 : tensor<64x128xi32>
  }

  func.func @test_full_uint() -> tensor<64x128xui32> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<64x128xui32>}> : () -> tensor<64x128xui32>
    // CHECK: = "ttnn.full"
    return %0 : tensor<64x128xui32>
  }

  func.func @test_full_float() -> tensor<64x128xf32> {
    %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    // CHECK: = "ttnn.full"
    return %0 : tensor<64x128xf32>
  }

  func.func @test_full_scalar() -> tensor<1x1xf32> {
    %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
    // CHECK: = "ttnn.full"
    return %0 : tensor<1x1xf32>
  }

  // Tests of ttir.constant where the value is a non-splat tensor.
  func.func @test_constant_f32() -> tensor<2x3xf32> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: -1.100000e+00, 2.200000e+00, -3.300000e+00
    // CHECK-SAME: 4.400000e+00, -5.500000e+00, 6.600000e+00
    %0 = "ttir.constant"() <{value = dense<[[-1.1, 2.2, -3.3], [4.4, -5.5, 6.6]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  func.func @test_constant_bf16() -> tensor<1x4xbf16> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: -1.101560e+00, 2.203130e+00, -3.296880e+00, 4.406250e+00
    %0 = "ttir.constant"() <{value = dense<[[-1.1, 2.2, -3.3, 4.4]]> : tensor<1x4xbf16>}> : () -> tensor<1x4xbf16>
    return %0 : tensor<1x4xbf16>
  }

  func.func @test_constant_ui32() -> tensor<1x1x3xui32> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: 1, 2, 3
    %0 = "ttir.constant"() <{value = dense<[[[1, 2, 3]]]> : tensor<1x1x3xui32>}> : () -> tensor<1x1x3xui32>
    return %0 : tensor<1x1x3xui32>
  }

  func.func @test_constant_ui16() -> tensor<4xui16> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: 1, 2, 3, 4
    %0 = "ttir.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xui16>}> : () -> tensor<4xui16>
    return %0 : tensor<4xui16>
  }

  func.func @test_constant_ui8() -> tensor<3x1xui8> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: 1
    // CHECK-SAME: 2
    // CHECK-SAME: 3
    %0 = "ttir.constant"() <{value = dense<[[1], [2], [3]]> : tensor<3x1xui8>}> : () -> tensor<3x1xui8>
    return %0 : tensor<3x1xui8>
  }
}
