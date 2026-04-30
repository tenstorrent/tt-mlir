// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that ui8 inputs to concat are promoted to i32 by the workaround pass.
// tilize_with_val_padding (used internally by concat) does not support ui8/i8.

module {
  func.func @concat_ui8(%arg0: tensor<32x32xui8>, %arg1: tensor<32x32xui8>) -> tensor<32x64xui8> {
    // CHECK-LABEL: func.func @concat_ui8
    // CHECK: ttnn.to_layout
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<32x32xui8
    // CHECK-SAME: tensor<32x32xsi32
    // CHECK: ttnn.to_layout
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: tensor<32x32xui8
    // CHECK-SAME: tensor<32x32xsi32
    // CHECK: ttnn.concat
    // CHECK-SAME: tensor<32x32xsi32
    // CHECK-SAME: tensor<32x64xsi32
    // CHECK: ttnn.to_layout
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<u8>
    // CHECK-SAME: tensor<32x64xsi32
    // CHECK-SAME: tensor<32x64xui8
    %0 = "ttir.concat"(%arg0, %arg1) <{dim = 1 : si32}> : (tensor<32x32xui8>, tensor<32x32xui8>) -> tensor<32x64xui8>
    return %0 : tensor<32x64xui8>
  }
}
