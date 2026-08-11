
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-const-eval=false enable-trace=true" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: FileCheck %s --input-file=%t2.mlir

func.func @round_test(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: emitc.call_opaque "ttnn::round"
  // CHECK-SAME: args = [0 : index, #emitc.opaque<"::std::nullopt">, #emitc.opaque<"::ttnn::MemoryConfig
  %1 = "ttir.round"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
