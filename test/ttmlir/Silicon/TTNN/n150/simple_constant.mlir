// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module @sysmem_creation attributes {} {
  func.func @test_empty_int() -> tensor<64x128xi32> {
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    return %0 : tensor<64x128xi32>
  }

  func.func @test_empty_uint() -> tensor<64x128xui32> {
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xui32>}> : () -> tensor<64x128xui32>
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    return %0 : tensor<64x128xui32>
  }

  func.func @test_empty_float() -> tensor<64x128xf32> {
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    return %0 : tensor<64x128xf32>
  }

  func.func @test_empty_float_scalar() -> tensor<1x1xf32> {
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    return %0 : tensor<1x1xf32>
  }

  func.func @test_full_int() -> tensor<64x128xi32> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    return %0 : tensor<64x128xi32>
  }

  func.func @test_full_uint() -> tensor<64x128xui32> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<64x128xui32>}> : () -> tensor<64x128xui32>
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    return %0 : tensor<64x128xui32>
  }

  func.func @test_full_float() -> tensor<64x128xf32> {
    %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    return %0 : tensor<64x128xf32>
  }

  func.func @test_full_scalar() -> tensor<1x1xf32> {
    %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    return %0 : tensor<1x1xf32>
  }
}
