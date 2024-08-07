// RUN: ttmlir-opt --ttir-load-system-desc --ttir-implicit-device --ttir-allocate --convert-ttir-to-ttmetal %s | FileCheck %s
#l1_ = #tt.memory_space<l1>
#layout = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>>
#layout1 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x4>, memref<64x32xf32, #l1_>>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout1> {
    // CHECK: %[[C:.*]] = "ttmetal.alloc"[[C:.*]]
    %0 = tensor.empty() : tensor<64x128xf32, #layout1>
    %1 = "ttir.to_layout"(%arg0, %0) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    return %1 : tensor<64x128xf32, #layout1>
  }
}
