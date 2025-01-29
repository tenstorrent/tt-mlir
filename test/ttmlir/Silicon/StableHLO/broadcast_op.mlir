// REQUIRES: stablehlo, num-chips-1 || num-chips-2
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module {
  func.func public @main(%arg0: tensor<1x1xf32>, %arg1: tensor<64x128xf32>) -> (tensor<64x128xf32>) {
    // CHECK-NOT: broadcast
    // CHECK: ttnn.add
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<64x128xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
