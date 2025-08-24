// RUN: not ttmlir-opt --tt-populate-argument-types="argument-types=main=input,parameter,parameter,constant" %s 2>&1 | FileCheck %s
module attributes {} {
  // CHECK: error: Function: "main" was provided in the argument types map, however it was not found in module!
  func.func @forward(%arg0: tensor<1x32x32x64xbf16> {ttir.name = "input_activations"}, %arg1: tensor<64x64x3x3xbf16> {ttir.name = "weights1"}, %arg2: tensor<1x1x1x64xbf16> {ttir.name = "weights2"}, %arg3: tensor<1x32x32x64xbf16> {ttir.name = "const_0"}) -> (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x32x32x64xbf16>) {
    return %arg0, %arg1, %arg2, %arg3 : tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x32x32x64xbf16>
  }
}
