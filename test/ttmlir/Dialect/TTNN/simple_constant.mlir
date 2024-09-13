// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @test_empty_int() -> tensor<64x128xi32> {
    %0 = "ttir.constant"() <{value = dense<0> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    return %0 : tensor<64x128xi32>
  }

  func.func @test_empty_float() -> tensor<64x128xf32> {
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    return %0 : tensor<64x128xf32>
  }

  func.func @test_full_int() -> tensor<64x128xi32> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    return %0 : tensor<64x128xi32>
  }

  func.func @test_full_float() -> tensor<64x128xf32> {
    %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    return %0 : tensor<64x128xf32>
  }
}
