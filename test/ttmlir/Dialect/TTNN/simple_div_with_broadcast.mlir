// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<1x128xf32>) -> tensor<64x128xf32> {
    // CHECK: %[[MULTIPLY_OUT:[0-9]+]] = "ttnn.empty"{{.+}} -> tensor<64x128xf32, {{.+}}
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: %[[RECIP_OUT:[0-9]+]] = "ttnn.empty"{{.+}} -> tensor<1x128xf32, {{.+}}
    // CHECK: %[[RECIP_RESULT:[0-9]+]] = "ttnn.reciprocal"(%{{[0-9]+}}, %[[RECIP_OUT]]){{.+}} -> tensor<1x128xf32, {{.+}}
    // CHECK: %{{[0-9]+}} = "ttnn.multiply"(%{{[0-9]+}}, %[[RECIP_RESULT]], %[[MULTIPLY_OUT]]){{.+}} -> tensor<64x128xf32, {{.+}}
    %1 = "ttir.div"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<1x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
