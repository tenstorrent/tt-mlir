// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="split-files=false" -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// Verify the Python output when split-files is disabled:
// 1. No file section labels are emitted
// 2. Imports appear at the top
// 3. cpu_hoisted_const_eval and forward_const_eval_0() are defined at module level
// 4. forward() calls consteval_forward wrapper with global _cached_forward
// 5. consteval_forward() contains the caching if-guard and its dict argument is named "ce_cache"

// CHECK-NOT: # File:
// CHECK: import ttnn
// CHECK: import utils
// CHECK-NOT: from consteval import
// CHECK-LABEL: def cpu_hoisted_const_eval_{{.*}}(
// CHECK:   ttir_cpu.add(
// CHECK-LABEL: def forward_const_eval_0(
// CHECK: _cached_forward = {}
// CHECK: def forward(input
// CHECK:   global _cached_forward
// CHECK:   _cached_forward = consteval_forward(
// CHECK:   ttnn.add(
// CHECK:   ttnn.add(
// CHECK-NOT: # File:
// CHECK-LABEL: def consteval_forward(ce_cache
// CHECK:   if not ce_cache:
// CHECK:     forward_const_eval_0(

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
