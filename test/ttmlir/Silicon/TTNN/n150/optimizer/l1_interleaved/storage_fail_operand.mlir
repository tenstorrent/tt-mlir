// Test for L1InterleavedFallbackAnalysis: add op should NOT be upgraded to L1 due to insufficient L1 memory to hold both operand and result tensors simultaneously
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false l1-interleaved-fallback-analysis-enabled=true" -o %t_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t_ttnn.mlir

module @L1InterleavedTestLargeTensorInput attributes {} {
  func.func @forward(%arg0: tensor<5120x5120xbf16>, %arg1: tensor<5120x5120xbf16>) -> tensor<5120x5120xbf16> {
    // CHECK-DAG: #[[DRAM_LAYOUT:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<160x160x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[L1_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#l1>{{.*}}<interleaved>>

    %0 = ttir.empty() : tensor<5120x5120xbf16>
    // CHECK: "ttnn.relu"{{.*}} -> tensor<5120x5120xbf16, #[[L1_LAYOUT]]>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<5120x5120xbf16>, tensor<5120x5120xbf16>) -> tensor<5120x5120xbf16>

    // CHECK: "ttnn.add"{{.*}} -> tensor<5120x5120xbf16, #[[DRAM_LAYOUT]]>
    %2 = ttir.empty() : tensor<5120x5120xbf16>
    %3 = "ttir.add"(%1, %arg1, %2) : (tensor<5120x5120xbf16>, tensor<5120x5120xbf16>, tensor<5120x5120xbf16>) -> tensor<5120x5120xbf16>

    // As output is the return value, not beneficial to move to L1, will always stay in DRAM.
    // CHECK: "ttnn.add"{{.*}} -> tensor<5120x5120xbf16, #[[DRAM_LAYOUT]]>
    %4 = ttir.empty() : tensor<5120x5120xbf16>
    %5 = "ttir.add"(%3, %arg1, %4) : (tensor<5120x5120xbf16>, tensor<5120x5120xbf16>, tensor<5120x5120xbf16>) -> tensor<5120x5120xbf16>

    return %5 : tensor<5120x5120xbf16>
  }
}
