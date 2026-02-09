// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-cpu-hoisted-const-eval=false" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @forward(%arg0: tensor<1x32x64x512xbf16>, %arg1: tensor<1x32x1x512xbf16>) -> tensor<1x32x64x512xbf16> {
    // CHECK: "ttnn.paged_update_cache"
    %update_index = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "ttir.update_cache"(%arg0, %arg1, %update_index) <{batch_offset = 0: i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x1x512xbf16>, tensor<1xi32>) -> tensor<1x32x64x512xbf16>
    %cst = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1x32x64x512xbf16>}> : () -> tensor<1x32x64x512xbf16>
    %2 = "ttir.add"(%1, %cst) : (tensor<1x32x64x512xbf16>, tensor<1x32x64x512xbf16>) -> tensor<1x32x64x512xbf16>
    return %2 : tensor<1x32x64x512xbf16>
  }
}
