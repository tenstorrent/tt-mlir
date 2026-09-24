// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --ttnn-common-to-emitc-pipeline="target-dylib=true" %s -o %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t.mlir -o %t.cpp
// RUN: FileCheck %s --input-file=%t.cpp

// Composite inlining can leave cached helpers after their callers (#9121).
// Keep this order and check declarations for both cached-call signatures.
// CHECK: #include "ttnn-precompiled.hpp"
// CHECK-NEXT: ::std::vector<::ttnn::Tensor> prepare_with_input(::std::vector<::ttnn::Tensor>);
// CHECK-NEXT: ::std::vector<::ttnn::Tensor> prepare_without_input();
// CHECK-NOT: prepare_with_input(::std::vector<::ttnn::Tensor>);
// CHECK-NOT: prepare_without_input();
// CHECK: ::std::vector<::ttnn::Tensor> forward(
// CHECK: = &prepare_with_input;
// CHECK: = &prepare_without_input;
// CHECK: ::std::vector<::ttnn::Tensor> prepare_with_input(::std::vector<::ttnn::Tensor> {{.*}}) {
// CHECK: ::std::vector<::ttnn::Tensor> prepare_without_input() {

#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<32x32xbf16, #layout>) -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) attributes {tt.function_type = "forward_device"} {
    %0 = "ttcore.load_cached"(%arg0) <{callee = @prepare_with_input}> : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    %1 = "ttcore.load_cached"() <{callee = @prepare_without_input}> : () -> tensor<32x32xbf16, #layout>
    return %0, %1 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
  }
  func.func @prepare_with_input(%arg0: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> attributes {tt.function_type = "const_eval"} {
    return %arg0 : tensor<32x32xbf16, #layout>
  }
  func.func @prepare_without_input() -> tensor<32x32xbf16, #layout> attributes {tt.function_type = "const_eval"} {
    %0 = "ttnn.full"() <{shape = #ttnn.shape<32x32>, fill_value = 1.0 : f32}> : () -> tensor<32x32xbf16, #layout>
    return %0 : tensor<32x32xbf16, #layout>
  }
}
