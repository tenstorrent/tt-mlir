// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir

func.func @gelu(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  // CHECK: emitc.call_opaque "init_sfpu"
  // CHECK: emitc.call_opaque "copy_tile_init"(%[[CB0:.+]]) :
  // CHECK-NEXT: emitc.call_opaque "copy_tile"(%[[CB0]], %{{.+}}, %{{.+}})
  // CHECK: emitc.call_opaque "gelu_tile_init"
  // CHECK-NEXT: emitc.call_opaque "gelu_tile"
  %1 = "ttir.gelu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
