// RUN: ttmlir-opt --ttir-load-system-desc --ttir-implicit-device --ttir-allocate %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#l1_ = #tt.memory_space<l1>
#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32, #layout>, %arg1: tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout> {
    // CHECK: %[[C:.*]] = "ttir.alloc"[[C:.*]]
    // CHECK-NOT: %[[C:.*]] = tensor.empty() : tensor<64x128xf32>
    %0 = tensor.empty() : tensor<64x128xf32, #layout>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout>
    return %1 : tensor<64x128xf32, #layout>
  }
}
