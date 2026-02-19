// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @test_empty_int8() -> tensor<64x128xi8> {
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xi8>}> : () -> tensor<64x128xi8>
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK-SAME: -> tensor<64x128xui8
    return %0 : tensor<64x128xi8>
  }

  func.func @test_empty_int16() -> tensor<64x128xi16> {
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xi16>}> : () -> tensor<64x128xi16>
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK-SAME: -> tensor<64x128xui16
    return %0 : tensor<64x128xi16>
  }

  func.func @test_empty_int() -> tensor<64x128xi32> {
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK-SAME: -> tensor<64x128xsi32
    return %0 : tensor<64x128xi32>
  }

  func.func @test_empty_uint() -> tensor<64x128xui32> {
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xui32>}> : () -> tensor<64x128xui32>
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK-SAME: -> tensor<64x128xui32
    return %0 : tensor<64x128xui32>
  }

  func.func @test_empty_bfloat16() -> tensor<64x128xbf16> {
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<64x128xbf16>}> : () -> tensor<64x128xbf16>
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fill_value = 0.000000e+00 : f32
    // CHECK-SAME: -> tensor<64x128xbf16
    return %0 : tensor<64x128xbf16>
  }

  func.func @test_empty_float() -> tensor<64x128xf32> {
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fill_value = 0.000000e+00 : f32
    // CHECK-SAME: -> tensor<64x128xf32
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  func.func @test_full_int8() -> tensor<64x128xi8> {
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fill_value = 1 : i32
    // CHECK-SAME: -> tensor<64x128xui8
    %0 = "ttir.constant"() <{value = dense<1> : tensor<64x128xi8>}> : () -> tensor<64x128xi8>
    return %0 : tensor<64x128xi8>
  }

  func.func @test_full_int16() -> tensor<64x128xi16> {
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fill_value = 1 : i32
    // CHECK-SAME: -> tensor<64x128xui16
    %0 = "ttir.constant"() <{value = dense<1> : tensor<64x128xi16>}> : () -> tensor<64x128xi16>
    return %0 : tensor<64x128xi16>
  }

  func.func @test_full_int() -> tensor<64x128xi32> {
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fill_value = 1 : i32
    // CHECK-SAME: -> tensor<64x128xsi32
    %0 = "ttir.constant"() <{value = dense<1> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    return %0 : tensor<64x128xi32>
  }

  func.func @test_full_uint() -> tensor<64x128xui32> {
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fill_value = 1 : i32
    // CHECK-SAME: -> tensor<64x128xui32
    %0 = "ttir.constant"() <{value = dense<1> : tensor<64x128xui32>}> : () -> tensor<64x128xui32>
    return %0 : tensor<64x128xui32>
  }

  func.func @test_full_bfloat16() -> tensor<64x128xbf16> {
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fill_value = 1.000000e+00 : f32
    // CHECK-SAME: -> tensor<64x128xbf16
    %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<64x128xbf16>}> : () -> tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }

  func.func @test_full_float() -> tensor<64x128xf32> {
    // CHECK: %{{[0-9]+}} = "ttnn.full"
    // CHECK-SAME: fill_value = 1.000000e+00 : f32
    // CHECK-SAME: -> tensor<64x128xf32
    %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  // Tests of ttir.constant where the value is a non-splat tensor.
  func.func @test_constant_f32() -> tensor<2x3xf32> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: -1.100000e+00, 2.200000e+00, -3.300000e+00
    // CHECK-SAME: 4.400000e+00, -5.500000e+00, 6.600000e+00
    // CHECK-SAME: tensor<2x3xf32>
    // CHECK-SAME: -> tensor<2x3xf32
    %0 = "ttir.constant"() <{value = dense<[[-1.1, 2.2, -3.3], [4.4, -5.5, 6.6]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  func.func @test_constant_bf16() -> tensor<1x4xbf16> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: -1.101560e+00, 2.203130e+00, -3.296880e+00, 4.406250e+00
    // CHECK-SAME: tensor<1x4xbf16>
    // CHECK-SAME: -> tensor<1x4xbf16
    %0 = "ttir.constant"() <{value = dense<[[-1.1, 2.2, -3.3, 4.4]]> : tensor<1x4xbf16>}> : () -> tensor<1x4xbf16>
    return %0 : tensor<1x4xbf16>
  }

  func.func @test_constant_ui32() -> tensor<1x1x3xui32> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: 1, 2, 3
    // CHECK-SAME: tensor<1x1x3xui32>
    // CHECK-SAME: -> tensor<1x1x3xui32
    %0 = "ttir.constant"() <{value = dense<[[[1, 2, 3]]]> : tensor<1x1x3xui32>}> : () -> tensor<1x1x3xui32>
    return %0 : tensor<1x1x3xui32>
  }

  func.func @test_constant_ui16() -> tensor<4xui16> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: 1, 2, 3, 4
    // CHECK-SAME: tensor<4xui16>
    // CHECK-SAME: -> tensor<4xui16
    %0 = "ttir.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xui16>}> : () -> tensor<4xui16>
    return %0 : tensor<4xui16>
  }

  func.func @test_constant_ui8() -> tensor<3x1xui8> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: 1
    // CHECK-SAME: 2
    // CHECK-SAME: 3
    // CHECK-SAME: tensor<3x1xui8>
    // CHECK-SAME: -> tensor<3x1xui8
    %0 = "ttir.constant"() <{value = dense<[[1], [2], [3]]> : tensor<3x1xui8>}> : () -> tensor<3x1xui8>
    return %0 : tensor<3x1xui8>
  }

  func.func @test_constant_i32() -> tensor<1x1x3xi32> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: -1, 2, 3
    // CHECK-SAME: tensor<1x1x3xsi32>
    // CHECK-SAME: -> tensor<1x1x3xsi32
    %0 = "ttir.constant"() <{value = dense<[[[-1, 2, 3]]]> : tensor<1x1x3xi32>}> : () -> tensor<1x1x3xi32>
    return %0 : tensor<1x1x3xi32>
  }

  func.func @test_constant_i64() -> tensor<1x1x3xi64> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: -1, 2, 3
    // CHECK-SAME: tensor<1x1x3xsi32>
    // CHECK-SAME: -> tensor<1x1x3xsi32
    %0 = "ttir.constant"() <{value = dense<[[[-1, 2, 3]]]> : tensor<1x1x3xi64>}> : () -> tensor<1x1x3xi64>
    return %0 : tensor<1x1x3xi64>
  }

  func.func @test_dense_attr() -> tensor<1x2xf64> {
    %0 = "ttir.constant"() <{value = dense_resource<dense_attr> : tensor<1x2xf64>}> : () -> tensor<1x2xf64>
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense_resource<dense_attr>
    return %0 : tensor<1x2xf64>
  }
  func.func @test_constant_complex() -> tensor<1x2xcomplex<f32>> {
    // CHECK: "ttnn.constant"
    // CHECK-SAME: value = dense
    // CHECK-SAME: [2.000000e+00, 0.000000e+00], [3.000000e+00, 0.000000e+00]
    // CHECK-SAME: -> tensor<1x2x2xf32
    %0 = "ttir.constant"() <{value = dense<[ [(2.0, 0.0), (3.0, 0.0)]]> : tensor<1x2xcomplex<f32>>}> : () -> tensor<1x2xcomplex<f32>>
    return %0 : tensor<1x2xcomplex<f32>>
  }
}
{-#
    dialect_resources: {
        builtin: {
            // This should encode for two bfloat16 values which are both 2.0
            // 0x020000000 is a hex string blob
            // 0x0040 is 2.0 in bfloat16
            // 0x00400040 is 2.0, 2.0
            dense_attr: "0x0200000000400040"
        }
    }
#-}
