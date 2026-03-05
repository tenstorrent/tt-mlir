// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// Verify the Python output when split-files is enabled (default):
// 1. Two file sections are emitted: "main" and "consteval"
// 2. Imports appear under each file label
// 3. Main file: forward() calls consteval_forward(), uses _cached_forward global
// 4. Consteval file: forward_const_eval_0() and consteval_forward() with caching
// 5. consteval_forward's dict argument is named "caching_dict"

// CHECK-LABEL: # File: "main"
// CHECK: import ttnn
// CHECK: import utils
// CHECK: from consteval import consteval_forward
// CHECK: _cached_forward = {}
// CHECK-LABEL: def forward(input):
// CHECK: global _cached_forward
// CHECK: _cached_forward = consteval_forward(_cached_forward, input)
// CHECK: ttnn.add(
// CHECK: ttnn.add(
// CHECK-LABEL: # File: "consteval"
// CHECK: import ttnn
// CHECK: import utils
// CHECK-LABEL: def forward_const_eval_0(input):
// CHECK: ttnn.add(
// CHECK-LABEL: def consteval_forward(caching_dict, input_1):
// CHECK: if not caching_dict:
// CHECK: forward_const_eval_0(
// CHECK: return caching_dict

module {
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                     %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %2 : tensor<32x32xbf16>
  }
}
