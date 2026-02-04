// RUN: ttmlir-opt --tt-populate-argument-types="argument-types=forward=input,kv_cache,kv_cache,parameter" -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  // CHECK: ttcore.argument_type = #ttcore.argument_type<input>
  // CHECK: ttir.name = "input_activations"
  // CHECK: ttcore.argument_type = #ttcore.argument_type<kv_cache>
  // CHECK: ttir.name = "key_cache"
  // CHECK: ttcore.argument_type = #ttcore.argument_type<kv_cache>
  // CHECK: ttir.name = "value_cache"
  // CHECK: ttcore.argument_type = #ttcore.argument_type<parameter>
  // CHECK: ttir.name = "weights"
  func.func @forward(
    %arg0: tensor<1x1x512xbf16> {ttir.name = "input_activations"},
    %arg1: tensor<1x32x64x512xbf16> {ttir.name = "key_cache"},
    %arg2: tensor<1x32x64x512xbf16> {ttir.name = "value_cache"},
    %arg3: tensor<512x512xbf16> {ttir.name = "weights"}
  ) -> tensor<1x1x512xbf16> {
    return %arg0 : tensor<1x1x512xbf16>
  }
}
