// RUN: ttmlir-opt --ttir-load-system-desc --ttir-implicit-device --ttir-layout --ttir-grid-set %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: #layout2 = #tt.layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<8x16xf32, #l1_>>
    // CHECK: %[[C:.*]] = "ttir.to_layout"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.to_layout"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.multiply"[[C:.*]] -> tensor<64x128xf32, #layout2>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
