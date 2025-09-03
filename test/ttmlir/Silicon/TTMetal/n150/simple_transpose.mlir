// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %basename_t.ttm %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir


func.func @transpose(%arg0: tensor<128x32xf32>) -> tensor<32x128xf32> {
  %0 = ttir.empty() : tensor<32x128xf32>
  %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<128x32xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
  // CHECK: emitc.call_opaque "transpose_wh_init"
  // CHECK: call_opaque "transpose_wh_tile"
  return %1 : tensor<32x128xf32>
}
