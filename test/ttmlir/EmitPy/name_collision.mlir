// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir


func.func @input(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  return %arg0 : tensor<64x64xbf16>
}

func.func @input_0(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  return %arg0 : tensor<64x64xbf16>
}

func.func @var_0(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  return %arg0 : tensor<64x64xbf16>
}

func.func @v_0(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  return %arg0 : tensor<64x64xbf16>
}

func.func @const0_0(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  return %arg0 : tensor<64x64xbf16>
}

func.func @ttnn_add_0(%arg0: tensor<64x64xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = ttir.empty() : tensor<64x64xbf16>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %1 : tensor<64x64xbf16>
}

func.func @main(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = call @input(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %1 = call @input_0(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %2 = call @var_0(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %3 = call @v_0(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %4 = call @const0_0(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %5 = call @ttnn_add_0(%0, %1) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %5 : tensor<64x64xbf16>
}
