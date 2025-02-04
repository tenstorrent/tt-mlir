// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16> {
    %output_shape = stablehlo.constant dense<[1, 1, 32, 128]> : tensor<4xi64>
    // CHECK: ttnn.arange
    %0 = "stablehlo.dynamic_iota"(%output_shape) {iota_dimension = 3: i64} : (tensor<4xi64>) -> tensor<1x1x32x128xbf16>
    %2 = "stablehlo.multiply"(%arg0, %0) : (tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    return %2 : tensor<1x1x32x128xbf16>
  }
}
