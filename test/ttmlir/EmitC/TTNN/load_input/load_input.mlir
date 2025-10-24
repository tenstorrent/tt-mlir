// RUN: ttmlir-opt --ttir-to-emitc-pipeline="system-desc-path=%system_desc_path% load-input-tensors-from-disk=true" -o %t.default.mlir %s
// RUN: FileCheck %s --check-prefix=DEFAULT --input-file=%t.default.mlir

// RUN: ttmlir-opt --ttir-to-emitc-pipeline="system-desc-path=%system_desc_path% load-input-tensors-from-disk=true tensor-load-directory=tensors" -o %t.custom_dir.mlir %s
// RUN: FileCheck %s --check-prefix=CUSTOM-DIR --input-file=%t.custom_dir.mlir

// RUN: ttmlir-opt --ttir-to-emitc-pipeline="system-desc-path=%system_desc_path% load-input-tensors-from-disk=true tensor-load-file-prefix=input" -o %t.custom_prefix.mlir %s
// RUN: FileCheck %s --check-prefix=CUSTOM-PREFIX --input-file=%t.custom_prefix.mlir

// RUN: ttmlir-opt --ttir-to-emitc-pipeline="system-desc-path=%system_desc_path% load-input-tensors-from-disk=true tensor-load-directory=tensors tensor-load-directory=tensors tensor-load-file-prefix=input" -o %t.custom_full.mlir %s
// RUN: FileCheck %s --check-prefix=CUSTOM-FULL --input-file=%t.custom_full.mlir

module {
  // DEFAULT: "::tt::tt_metal::load_tensor_flatbuffer"(%0
  // DEFAULT-SAME: args = [#emitc.opaque<"\22arg0.tensorbin\22"
  // DEFAULT-NEXT: "::tt::tt_metal::load_tensor_flatbuffer"(%0
  // DEFAULT-SAME: args = [#emitc.opaque<"\22arg1.tensorbin\22"

  // CUSTOM-DIR: "::tt::tt_metal::load_tensor_flatbuffer"(%0
  // CUSTOM-DIR-SAME: args = [#emitc.opaque<"\22tensors/arg0.tensorbin\22"
  // CUSTOM-DIR-NEXT: "::tt::tt_metal::load_tensor_flatbuffer"(%0
  // CUSTOM-DIR-SAME: args = [#emitc.opaque<"\22tensors/arg1.tensorbin\22"

  // CUSTOM-PREFIX: "::tt::tt_metal::load_tensor_flatbuffer"(%0
  // CUSTOM-PREFIX-SAME: args = [#emitc.opaque<"\22input0.tensorbin\22"
  // CUSTOM-PREFIX-NEXT: "::tt::tt_metal::load_tensor_flatbuffer"(%0
  // CUSTOM-PREFIX-SAME: args = [#emitc.opaque<"\22input1.tensorbin\22"

  // CUSTOM-FULL: "::tt::tt_metal::load_tensor_flatbuffer"(%0
  // CUSTOM-FULL-SAME: args = [#emitc.opaque<"\22tensors/input0.tensorbin\22"
  // CUSTOM-FULL-NEXT: "::tt::tt_metal::load_tensor_flatbuffer"(%0
  // CUSTOM-FULL-SAME: args = [#emitc.opaque<"\22tensors/input1.tensorbin\22"
  func.func @add(%arg0 : tensor<32x32xbf16> { ttcore.argument_type = #ttcore.argument_type<constant>}, %arg1 : tensor<32x32xbf16> { ttcore.argument_type = #ttcore.argument_type<input> }) -> tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg0, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.subtract"(%arg1, %1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %3 : tensor<32x32xbf16>
  }
}
