// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true" %s -o %t.mlir --mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Test rotary embedding sharding for LLM models, Llama 3.2 example for key

func.func @rope_key_llama3_2_sharding(
    %input: tensor<1x1x8x64xbf16>,
    %cos_cache: tensor<1x1x1x64xbf16>,
    %sin_cache: tensor<1x1x1x64xbf16>,
    %output: tensor<1x1x8x64xbf16>) -> tensor<1x1x8x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x8x64xbf16>
    // First rotary embedding operation with height sharding: input [1,1,32,64xbf16]
    // CHECK: = "ttnn.rotary_embedding"{{.*}} -> {{.*}}#ttnn.ttnn_layout{{.*}}height_sharded{{.*}}
    %1 = "ttir.rotary_embedding"(%input, %cos_cache, %sin_cache, %0) : (tensor<1x1x8x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x1x8x64xbf16>) -> tensor<1x1x8x64xbf16>

    %2 = ttir.empty() : tensor<1x1x8x64xbf16>
    // Additional add to create chain (required for sharding policy to apply)
    %3 = "ttir.add"(%1, %1, %2) : (tensor<1x1x8x64xbf16>, tensor<1x1x8x64xbf16>, tensor<1x1x8x64xbf16>) -> tensor<1x1x8x64xbf16>

    return %3: tensor<1x1x8x64xbf16>
}
