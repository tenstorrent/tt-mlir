// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-implicit-device --ttir-allocate --convert-ttir-to-ttmetal %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

#l1_ = #tt.memory_space<l1>

#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<4x16xf32, #l1_>, none_layout>
#layout1 = #tt.layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<2x8xf32, #l1_>, none_layout>
func.func @simple(%arg0: tensor<4x16xf32, #layout>) -> tensor<4x16xf32, #layout1> {
  %0 = tensor.empty() : tensor<4x16xf32, #layout1>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<4x16xf32, #layout>, tensor<4x16xf32, #layout1>) -> tensor<4x16xf32, #layout1>
  return %1 : tensor<4x16xf32, #layout1>
}

#untilized = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>, none_layout>
#tilized = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<2x4x!tt.tile<32 x 32, f32>, #l1_>, none_layout>
func.func @tilize(%arg0: tensor<64x128xf32, #untilized>) -> tensor<64x128xf32, #untilized> {
  %0 = tensor.empty() : tensor<64x128xf32, #tilized>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %1 = "ttir.to_layout"(%arg0, %0) : (tensor<64x128xf32, #untilized>, tensor<64x128xf32, #tilized>) -> tensor<64x128xf32, #tilized>
  %2 = tensor.empty() : tensor<64x128xf32, #untilized>
  // CHECK: %[[C:.*]] = "ttmetal.dispatch"[[C:.*]]
  %3 = "ttir.to_layout"(%1, %2) : (tensor<64x128xf32, #tilized>, tensor<64x128xf32, #untilized>) -> tensor<64x128xf32, #untilized>
  return %3 : tensor<64x128xf32, #untilized>
}
