// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Test RMSNorm sharding for LLM models, without weights, with bias, Llama 3.2 example variant

func.func @rmsnorm_sharding(%arg0: tensor<1x1x32x2048xbf16>, %arg1: tensor<2048xbf16>) -> tensor<1x1x32x2048xbf16> {
    // CHECK-DAG: #[[LAYOUT_0:.*]] = #ttnn.ttnn_layout{{.*}}width_sharded{{.*}}

    %0 = ttir.empty() : tensor<1x1x32x2048xbf16>
    // First RMSNorm operation with width sharding: input [1,1,32,2048xbf16], no weights, bias [2048xbf16]
    // CHECK: = "ttnn.rms_norm"{{.*}} -> {{.*}}#[[LAYOUT_0]]{{.*}}
    %1 = "ttir.rms_norm"(%arg0, %arg1, %0) <{normalized_shape = array<i64: 2048>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 1, 1>}> : (tensor<1x1x32x2048xbf16>, tensor<2048xbf16>, tensor<1x1x32x2048xbf16>) -> tensor<1x1x32x2048xbf16>

    %2 = ttir.empty() : tensor<1x1x32x2048xbf16>
    // Additional add to create chain (required for sharding policy to apply)
    %3 = "ttir.add"(%1, %1, %2) : (tensor<1x1x32x2048xbf16>, tensor<1x1x32x2048xbf16>, tensor<1x1x32x2048xbf16>) -> tensor<1x1x32x2048xbf16>

    return %3: tensor<1x1x32x2048xbf16>
}
