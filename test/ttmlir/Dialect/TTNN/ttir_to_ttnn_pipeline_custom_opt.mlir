// RUN: ttmlir-opt --ttir-load-system-desc --ttir-to-ttnn-backend-pipeline="enable-grid-set=false" %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: #layout1 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>>
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.multiply"[[C:.*]] -> tensor<64x128xf32, #layout1>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.to_memory_config"[[C:.*]]
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %1 : tensor<64x128xf32>
  }
}
