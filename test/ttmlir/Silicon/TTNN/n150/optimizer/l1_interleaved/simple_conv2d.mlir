// Test for L1InterleavedFallbackAnalysis: simple conv2d pattern
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false l1-interleaved-fallback-analysis-enabled=true max-legal-layouts=32" -o %t_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t_ttnn.mlir

module @L1InterleavedTestConv2D attributes {} {
  func.func @forward(
    %arg0: tensor<4x32x32x16xbf16>,
    %arg1: tensor<32x16x3x3xbf16>,
    %arg2: tensor<64x32x1x1xbf16>,
    %arg3: tensor<64x64x1x1xbf16>
  ) -> tensor<4x32x32x64xbf16> {
    %0 = ttir.empty() : tensor<4x32x32x32xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0) {dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}: (tensor<4x32x32x16xbf16>, tensor<32x16x3x3xbf16>, tensor<4x32x32x32xbf16>) -> tensor<4x32x32x32xbf16>
    %2 = ttir.empty() : tensor<4x32x32x64xbf16>
    %3 = "ttir.conv2d"(%1, %arg2, %2) {dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}: (tensor<4x32x32x32xbf16>, tensor<64x32x1x1xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>
    %4 = ttir.empty() : tensor<4x32x32x64xbf16>
    %5 = "ttir.conv2d"(%3, %arg3, %4) {dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}: (tensor<4x32x32x64xbf16>, tensor<64x64x1x1xbf16>, tensor<4x32x32x64xbf16>) -> tensor<4x32x32x64xbf16>
    return %5 : tensor<4x32x32x64xbf16>
  }
}

// CHECK-DAG: #[[DRAM_1:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
// CHECK-DAG: #[[DRAM_2:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
// CHECK-DAG: #[[DRAM_3:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
// CHECK-DAG: #[[DRAM_4:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
// CHECK-DAG: #[[DRAM_5:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
// CHECK-DAG: #[[DRAM_6:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>

// Reshape skipped to be optimized out in metal lowering, stays in DRAM.
// CHECK: %{{.*}} = "ttnn.reshape"{{.*}} -> tensor<{{.*}}, #[[DRAM_3]]>

// Consumer is conv2d which uses matmul, stays in DRAM.
// CHECK: %{{.*}} = "ttnn.conv2d"{{.*}} -> tensor<{{.*}}, #[[DRAM_3]]>
// Conv2d which uses matmul, stays in DRAM.
// CHECK: %{{.*}} = "ttnn.conv2d"{{.*}} -> tensor<{{.*}}, #[[DRAM_6]]>
// Conv2d which uses matmul, stays in DRAM.
// CHECK: %{{.*}} = "ttnn.conv2d"{{.*}} -> tensor<{{.*}}, #[[DRAM_6]]>

// As output is the return value, not beneficial to move to L1, will always stay in DRAM.
// CHECK: %{{.*}} = "ttnn.view"{{.*}} -> tensor<{{.*}}, #[[DRAM_2]]>
// CHECK: return{{.*}} : tensor<{{.*}}, #[[DRAM_2]]>
