// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<any_device>
#l1 = #tt.memory_space<l1>
#system = #tt.memory_space<system>
#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #system>, none_layout>
#layout1 = #tt.layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<8x16xf32, #system>, none_layout>
#layout2 = #tt.layout<(d0, d1) -> (d0, d1), undef, <8x8>, memref<8x16xf32, #l1>, interleaved>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout1> {
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<64x128xf32, #layout1>
    // CHECK: %[[C:.*]] = "ttnn.relu"[[C:.*]]
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %1 : tensor<64x128xf32, #layout1>
  }
}
