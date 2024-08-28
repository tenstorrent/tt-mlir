// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<256x512xf32>, %arg1: tensor<256x512xf32>) -> tensor<256x512xf32> {
    // CHECK: #[[LAYOUT_1:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<32x64xf32, #l1_>>
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<256x512xf32>
    // CHECK: %[[C:.*]] = "ttnn.multiply"[[C:.*]] -> tensor<256x512xf32, #[[LAYOUT_1]]>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<256x512xf32>, tensor<256x512xf32>, tensor<256x512xf32>) -> tensor<256x512xf32>
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %1 : tensor<256x512xf32>
  }
}
