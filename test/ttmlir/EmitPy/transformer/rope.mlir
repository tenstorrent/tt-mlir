// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir

module {
  func.func @main(%arg0: tensor<1x1024x64xbf16>, %arg1: tensor<1x32x1024x64xbf16>, %arg2: tensor<1x1024x64xbf16>) -> tensor<1x32x1024x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x1024x64xbf16>
    %1 = ttir.empty() : tensor<1x1x1024x64xbf16>
    %2 = "ttir.reshape"(%arg2, %0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x1024x64xbf16>, tensor<1x1x1024x64xbf16>) -> tensor<1x1x1024x64xbf16>
    %3 = "ttir.reshape"(%arg0, %1) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x1024x64xbf16>, tensor<1x1x1024x64xbf16>) -> tensor<1x1x1024x64xbf16>
    %4 = "ttir.rotary_embedding"(%arg1, %2, %3) : (tensor<1x32x1024x64xbf16>, tensor<1x1x1024x64xbf16>, tensor<1x1x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    return %4 : tensor<1x32x1024x64xbf16>
  }
}
