// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="enable-cpu-hoisted-const-eval=false" -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// Verify that when cpu-hoisted-const-eval is disabled and split-files is
// enabled (default), the consteval functions still emit proper MemoryConfig
// values instead of None. This is a regression test for the case where
// DeviceOp is invisible from inside the consteval FileOp due to the
// IsolatedFromAbove trait.

// CHECK-LABEL: # File: "main"
// CHECK: def forward(
// CHECK:   consteval_forward(
// CHECK:   ttnn.add(

// CHECK-LABEL: # File: "consteval"
// CHECK-LABEL: def forward_const_eval_0(
// CHECK:   ttnn.add({{.*}}memory_config=ttnn.MemoryConfig

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
