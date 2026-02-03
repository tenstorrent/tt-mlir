// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="enable-const-eval=true target-module=true" -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// Verify that with target-module=true and const-eval enabled:
// 1. The entry function has signature forward(input, device)
// 2. Execute functions that call for const-eval logic receive device and do not call DeviceGetter
// 2. Const-eval functions also receive device and do not call DeviceGetter
// 3. Const-eval wrapper calls pass device when available
//
// CHECK-LABEL: # File: "main"
// CHECK-LABEL: def forward(input, device):

// CHECK-LABEL: # File: "consteval"
// CHECK-LABEL: def forward_const_eval_0(
// CHECK-SAME: device
// CHECK-NOT: utils.DeviceGetter.get_device
// CHECK-LABEL: def consteval_forward(
// CHECK-SAME: device
// CHECK: utils.constEvalFuncWrapper(
// CHECK-SAME: device
// CHECK-NOT: utils.DeviceGetter.get_device

module {
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                     %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    %1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %7 = "ttir.subtract"(%3, %5) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %9 = "ttir.multiply"(%1, %7) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }
}
