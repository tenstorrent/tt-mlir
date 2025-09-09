// Test for L1InterleavedFallbackAnalysis: relu op should NOT be upgraded to L1 interleaved due to insufficient L1 memory when add op is overridden to use L1 height-sharded layout.
// Tensor l1 usage cap is set to 0.6 to ensure that relu op cannot fit into L1 interleaved layout along with add op being sharded in L1.
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false l1-interleaved-fallback-analysis-enabled=true override-output-layout=add_op=l1:height_sharded tensor-l1-usage-cap=0.6" -o %t_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t_ttnn.mlir

module @L1InterleavedTestLargeTensorSharded attributes {} {
  func.func @forward(%arg0: tensor<4096x4096xbf16>, %arg1: tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16> {
    // CHECK-DAG: #[[DRAM_LAYOUT:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x128x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[SHARDED_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}, <height_sharded>>

    %0 = ttir.empty() : tensor<4096x4096xbf16>
    // CHECK: "ttnn.relu"{{.*}}-> tensor<4096x4096xbf16, #[[DRAM_LAYOUT]]>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>

    // CHECK: "ttnn.add"{{.*}}-> tensor<4096x4096xbf16, #[[SHARDED_LAYOUT]]>
    %2 = ttir.empty() : tensor<4096x4096xbf16>
    %3 = "ttir.add"(%1, %arg1, %2) : (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16> loc(#loc)

    return %3 : tensor<4096x4096xbf16>
  }
}
#loc = loc("add_op")
