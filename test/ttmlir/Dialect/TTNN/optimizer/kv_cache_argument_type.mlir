// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=false" -o %t %s -mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// Verify that KVCache arguments stay tiled and are not forced to row-major layout.
// KVCache tensors have different layout requirements than regular inputs.

// CHECK-LABEL: func.func @kv_cache_stays_tiled
// The cache argument should remain tiled (no conversion to row-major).
// CHECK: %arg0: tensor<{{.*}}#ttnn.ttnn_layout<{{.*}}tile{{.*}}>>
// CHECK-NOT: ttnn.to_layout{{.*}}%arg0{{.*}}row_major
func.func @kv_cache_stays_tiled(
    %cache: tensor<1x32x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<kv_cache>},
    %fill_value: tensor<1x32x1x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}
) -> tensor<1x32x64x512xbf16> {
    %update_index = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = "ttir.update_cache"(%cache, %fill_value, %update_index) <{batch_offset = 0: i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x1x512xbf16>, tensor<1xi32>) -> tensor<1x32x64x512xbf16>
    return %0 : tensor<1x32x64x512xbf16>
}
