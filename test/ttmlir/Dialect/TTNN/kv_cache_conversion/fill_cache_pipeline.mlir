// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="experimental-kv-cache-dtype=bfp_bf8" --mlir-print-local-scope -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module {
  // CHECK-LABEL: func.func @forward
  // CHECK-SAME: %arg0: tensor<{{.*}}bfp_bf8{{.*}} {ttcore.kv_cache}
  // CHECK-SAME: %arg1: tensor<{{.*}}bf16
  func.func @forward(%arg0: tensor<1x32x64x512xbf16>, %arg1: tensor<1x32x3x512xbf16>) -> tensor<1x32x64x512xbf16> {
    // CHECK: "ttnn.typecast"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
    // CHECK: "ttnn.fill_cache"
    %1 = "ttir.fill_cache"(%arg0, %arg1) <{batch_offset = 0 : i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x3x512xbf16>) -> tensor<1x32x64x512xbf16>
    return %1 : tensor<1x32x64x512xbf16>
  }
}
