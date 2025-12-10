// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true" %s -o %t.mlir --mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Test GELU sharding for LLM models, example with shape [32x1x8192] from Phi LLM

func.func @gelu_sharding(%arg0: tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16> {
    // GELU operation with potential sharding: input [32,1,8192xbf16]
    // CHECK: = "ttnn.gelu"{{.*}} -> {{.*}}#ttnn.ttnn_layout{{.*}}width_sharded{{.*}}
    %1 = "ttir.gelu"(%arg0) : (tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16>

    // Additional add to create chain (required for sharding policy to apply)
    %2 = "ttir.add"(%1, %1) : (tensor<32x1x8192xbf16>, tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16>

    return %2: tensor<32x1x8192xbf16>
}
