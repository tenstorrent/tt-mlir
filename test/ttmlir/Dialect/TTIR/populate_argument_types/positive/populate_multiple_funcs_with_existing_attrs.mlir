// RUN: ttmlir-opt --tt-populate-argument-types="argument-types=forward1=input,parameter,parameter,constant;forward2=input,input,constant,parameter" %s | FileCheck %s
module attributes {} {
  // CHECK: tt.argument_type = #tt.argument_type<input>
  // CHECK: ttir.name = "input_activations"
  // CHECK: tt.argument_type = #tt.argument_type<parameter>
  // CHECK: ttir.name = "weights1"
  // CHECK: tt.argument_type = #tt.argument_type<parameter>
  // CHECK: ttir.name = "weights2"
  // CHECK: tt.argument_type = #tt.argument_type<constant>
  // CHECK: ttir.name = "const_0"
  func.func @forward1(%arg0: tensor<1x32x32x64xbf16> {ttir.name = "input_activations"}, %arg1: tensor<64x64x3x3xbf16> {ttir.name = "weights1"}, %arg2: tensor<1x1x1x64xbf16> {ttir.name = "weights2"}, %arg3: tensor<1x32x32x64xbf16> {ttir.name = "const_0"}) -> (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x32x32x64xbf16>) {
    return %arg0, %arg1, %arg2, %arg3 : tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x32x32x64xbf16>
  }

  // CHECK: tt.argument_type = #tt.argument_type<input>
  // CHECK: ttir.name = "input_activations"
  // CHECK: tt.argument_type = #tt.argument_type<input>
  // CHECK: ttir.name = "weights1"
  // CHECK: tt.argument_type = #tt.argument_type<constant>
  // CHECK: ttir.name = "weights2"
  // CHECK: tt.argument_type = #tt.argument_type<parameter>
  // CHECK: ttir.name = "const_0"
  func.func @forward2(%arg0: tensor<1x32x32x64xbf16> {ttir.name = "input_activations"}, %arg1: tensor<64x64x3x3xbf16> {ttir.name = "weights1"}, %arg2: tensor<1x1x1x64xbf16> {ttir.name = "weights2"}, %arg3: tensor<1x32x32x64xbf16> {ttir.name = "const_0"}) -> (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x32x32x64xbf16>) {
    return %arg0, %arg1, %arg2, %arg3 : tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x32x32x64xbf16>
  }
}
