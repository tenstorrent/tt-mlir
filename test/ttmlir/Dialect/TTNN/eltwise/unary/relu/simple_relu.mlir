// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<any_device>
#l1 = #ttnn.buffer_type<l1>
#system = #ttnn.buffer_type<system_memory>
#tensor_config = #ttnn.tensor_config<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system>>
#tensor_config1 = #ttnn.tensor_config<(d0, d1) -> (d0, d1), <8x8>, memref<8x16xf32, #system>>
#tensor_config2 = #ttnn.tensor_config<(d0, d1) -> (d0, d1), <8x8>, memref<8x16xf32, #l1>, interleaved>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32, #tensor_config>) -> tensor<64x128xf32, #tensor_config1> {
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<64x128xf32, #tensor_config1>
    // CHECK: %[[C:.*]] = "ttnn.relu"[[C:.*]]
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32, #tensor_config>, tensor<64x128xf32, #tensor_config1>) -> tensor<64x128xf32, #tensor_config1>
    return %1 : tensor<64x128xf32, #tensor_config1>
  }
}
