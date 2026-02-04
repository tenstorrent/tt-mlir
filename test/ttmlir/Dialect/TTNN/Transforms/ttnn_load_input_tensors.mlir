// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttcore-unwrap-device-module --ttnn-tuplify-tensors --ttnn-load-input-tensors -o %t.default.mlir %s
// RUN: FileCheck %s --check-prefix=DEFAULT --input-file=%t.default.mlir

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttcore-unwrap-device-module --ttnn-tuplify-tensors --ttnn-load-input-tensors="tensor-load-directory=tensors" -o %t.custom_dir.mlir %s
// RUN: FileCheck %s --check-prefix=CUSTOM-DIR --input-file=%t.custom_dir.mlir

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttcore-unwrap-device-module --ttnn-tuplify-tensors --ttnn-load-input-tensors="tensor-load-file-prefix=input" -o %t.custom_prefix.mlir %s
// RUN: FileCheck %s --check-prefix=CUSTOM-PREFIX --input-file=%t.custom_prefix.mlir

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttcore-unwrap-device-module --ttnn-tuplify-tensors --ttnn-load-input-tensors="tensor-load-directory=tensors tensor-load-file-prefix=input" -o %t.custom_full.mlir %s
// RUN: FileCheck %s --check-prefix=CUSTOM-FULL --input-file=%t.custom_full.mlir

module {
  // DEFAULT-NOT: @load_inputs_for_func_no_inputs()
  // DEFAULT: @load_inputs_for_add()
  // DEFAULT: "ttnn.load_tensor"
  // DEFAULT-SAME: file_path = "arg0.tensorbin"
  // DEFAULT-NEXT: "ttnn.load_tensor"
  // DEFAULT-SAME: file_path = "arg1.tensorbin"

  // CUSTOM-DIR-NOT: @load_inputs_for_func_no_inputs()
  // CUSTOM-DIR: @load_inputs_for_add()
  // CUSTOM-DIR: "ttnn.load_tensor"
  // CUSTOM-DIR-SAME: file_path = "tensors/arg0.tensorbin"
  // CUSTOM-DIR-NEXT: "ttnn.load_tensor"
  // CUSTOM-DIR-SAME: file_path = "tensors/arg1.tensorbin"

  // CUSTOM-PREFIX-NOT: @load_inputs_for_func_no_inputs()
  // CUSTOM-PREFIX: @load_inputs_for_add()
  // CUSTOM-PREFIX: "ttnn.load_tensor"
  // CUSTOM-PREFIX-SAME: file_path = "input0.tensorbin"
  // CUSTOM-PREFIX-NEXT: "ttnn.load_tensor"
  // CUSTOM-PREFIX-SAME: file_path = "input1.tensorbin"

  // CUSTOM-FULL-NOT: @load_inputs_for_func_no_inputs()
  // CUSTOM-FULL: @load_inputs_for_add()
  // CUSTOM-FULL: "ttnn.load_tensor"
  // CUSTOM-FULL-SAME: file_path = "tensors/input0.tensorbin"
  // CUSTOM-FULL-NEXT: "ttnn.load_tensor"
  // CUSTOM-FULL-SAME: file_path = "tensors/input1.tensorbin"
  func.func @add(%arg0 : tensor<32x32xbf16> { ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1 : tensor<32x32xbf16> { ttcore.argument_type = #ttcore.argument_type<input> }) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.subtract"(%arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }

  func.func @func_no_inputs() -> (tensor<64x64xbf16>) {
    %0 = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<64x64xbf16>}> :
        () -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }
}
