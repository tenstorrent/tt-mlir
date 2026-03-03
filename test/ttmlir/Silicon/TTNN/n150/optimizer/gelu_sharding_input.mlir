// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2" %s -o %t.mlir --mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Test GELU sharding for LLM models, example with shape [32x1x8192] from Phi LLM with sharded input too

func.func @gelu_sharding(%arg0: tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16> {
    // Initial add to create chain and sharded input
    %1 = "ttir.add"(%arg0, %arg0) : (tensor<32x1x8192xbf16>, tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16>

    // GELU operation with potential sharding: input [32,1,8192xbf16]
    // CHECK: = "ttnn.gelu"{{.*}}#ttnn.ttnn_layout{{.*}}_sharded{{.*}} -> {{.*}}#ttnn.ttnn_layout{{.*}}width_sharded{{.*}}
    %2 = "ttir.gelu"(%1) : (tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16>

    // Additional add to create chain (required for sharding policy to apply)
    %3 = "ttir.add"(%2, %2) : (tensor<32x1x8192xbf16>, tensor<32x1x8192xbf16>) -> tensor<32x1x8192xbf16>

    return %3: tensor<32x1x8192xbf16>
}
