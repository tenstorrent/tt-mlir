// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=4,1,1" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// UNSUPPORTED: true
// REQUIRES: multi-chip
func.func @forward(%arg0: tensor<1x1x32x32xf32>) -> tensor<1x1x32x128xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<1x1x32x128xf32>
  // CHECK: %[[C:.*]] = "ttnn.all_gather"[[C:.*]]
  %1 = "ttir.all_gather"(%arg0, %0) <{dim = 3 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x32x128xf32>) -> tensor<1x1x32x128xf32>
  return %1 : tensor<1x1x32x128xf32>
}
