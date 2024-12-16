// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = tensor.empty() : tensor<32x32xbf16>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}

func.func @subtract(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = tensor.empty() : tensor<32x32xbf16>
  %1 = "ttir.subtract"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}
