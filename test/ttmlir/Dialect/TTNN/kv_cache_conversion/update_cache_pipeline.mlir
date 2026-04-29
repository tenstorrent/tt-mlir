// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="experimental-kv-cache-dtype=bfp_bf8 enable-cpu-hoisted-const-eval=false" --mlir-print-local-scope -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module {
  // CHECK-LABEL: func.func @forward
  // CHECK-SAME: %arg0: tensor<{{.*}}bfp_bf8{{.*}} {ttcore.kv_cache}
  // CHECK-SAME: %arg1: tensor<{{.*}}bf16
  func.func @forward(%arg0: tensor<1x32x64x512xbf16>, %arg1: tensor<1x32x1x512xbf16>) -> tensor<1x32x64x512xbf16> {
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: "ttnn.paged_update_cache"
    %update_index = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "ttir.update_cache"(%arg0, %arg1, %update_index) <{batch_offset = 0 : i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x1x512xbf16>, tensor<1xi32>) -> tensor<1x32x64x512xbf16>
    return %1 : tensor<1x32x64x512xbf16>
  }
}
