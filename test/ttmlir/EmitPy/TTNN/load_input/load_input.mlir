// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% load-input-tensors-from-disk=true" -o %t.default.mlir %s
// RUN: FileCheck %s --check-prefix=DEFAULT --input-file=%t.default.mlir

// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% load-input-tensors-from-disk=true tensor-load-directory=tensors" -o %t.custom_dir.mlir %s
// RUN: FileCheck %s --check-prefix=CUSTOM-DIR --input-file=%t.custom_dir.mlir

// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% load-input-tensors-from-disk=true tensor-load-file-prefix=input" -o %t.custom_prefix.mlir %s
// RUN: FileCheck %s --check-prefix=CUSTOM-PREFIX --input-file=%t.custom_prefix.mlir

// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% load-input-tensors-from-disk=true tensor-load-directory=tensors tensor-load-directory=tensors tensor-load-file-prefix=input" -o %t.custom_full.mlir %s
// RUN: FileCheck %s --check-prefix=CUSTOM-FULL --input-file=%t.custom_full.mlir

module {
  // DEFAULT: load_tensor{{.*}}args = [#emitpy.opaque<"\22arg0.tensorbin\22"
  // DEFAULT: load_tensor{{.*}}args = [#emitpy.opaque<"\22arg1.tensorbin\22"

  // CUSTOM-DIR: load_tensor{{.*}}args = [#emitpy.opaque<"\22tensors/arg0.tensorbin\22"
  // CUSTOM-DIR: load_tensor{{.*}}args = [#emitpy.opaque<"\22tensors/arg1.tensorbin\22"

  // CUSTOM-PREFIX: load_tensor{{.*}}args = [#emitpy.opaque<"\22input0.tensorbin\22"
  // CUSTOM-PREFIX: load_tensor{{.*}}args = [#emitpy.opaque<"\22input1.tensorbin\22"

  // CUSTOM-FULL: load_tensor{{.*}}args = [#emitpy.opaque<"\22tensors/input0.tensorbin\22"
  // CUSTOM-FULL: load_tensor{{.*}}args = [#emitpy.opaque<"\22tensors/input1.tensorbin\22"
  func.func @add(%arg0 : tensor<32x32xbf16> { ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1 : tensor<32x32xbf16> { ttcore.argument_type = #ttcore.argument_type<input> }) -> tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg0, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.subtract"(%arg1, %1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %3 : tensor<32x32xbf16>
  }
}
