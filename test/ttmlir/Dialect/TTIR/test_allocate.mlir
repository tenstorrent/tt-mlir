// RUN: ttmlir-opt --ttir-load-system-desc --ttir-implicit-device --ttir-layout --ttir-allocate %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: %[[C:.*]] = "ttir.alloc"[[C:.*]]
    // CHECK-NOT: %[[C:.*]] = tensor.empty() : tensor<64x128xf32>
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: "ttir.dealloc"[[C:.*]]
    return %1 : tensor<64x128xf32>
  }
}
