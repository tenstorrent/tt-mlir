// Test for L1InterleavedFallbackAnalysis: simple fork-join pattern
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false l1-interleaved-fallback-analysis-enabled=true max-legal-layouts=32" -o %t_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t_ttnn.mlir

module @L1InterleavedTestForkJoin attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %arg2: tensor<64x128xbf16>, %arg3: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK-DAG: #[[DRAM:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[L1:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#l1>{{.*}}<interleaved>>
    %0 = ttir.empty() : tensor<64x128xbf16>
    // not immediately consumed -> dram
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<{{.*}}, #[[DRAM]]>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %2 = ttir.empty() : tensor<64x128xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<{{.*}}, #[[L1]]>
    %3 = "ttir.add"(%arg2, %arg3, %2) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %4 = ttir.empty() : tensor<64x128xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<{{.*}}, #[[L1]]>
    %5 = "ttir.add"(%1, %3, %4) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %6 = ttir.empty() : tensor<64x128xbf16>
    // fork, not a single user -> dram
    // CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<{{.*}}, #[[DRAM]]>
    %7 = "ttir.relu"(%5, %6) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // Fork: %7 used in two ops
    %8 = ttir.empty() : tensor<64x128xbf16>
    // not immediately consumed -> dram
    // CHECK: %{{.*}} = "ttnn.neg"{{.*}} -> tensor<{{.*}}, #[[DRAM]]>
    %9 = "ttir.neg"(%7, %8) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %10 = ttir.empty() : tensor<64x128xbf16>
    // CHECK: %{{.*}} = "ttnn.abs"{{.*}} -> tensor<{{.*}}, #[[L1]]>
    %11 = "ttir.abs"(%7, %10) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // Join: add the results
    %12 = ttir.empty() : tensor<64x128xbf16>
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<{{.*}}, #[[L1]]>
    %13 = "ttir.add"(%9, %11, %12) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %14 = ttir.empty() : tensor<64x128xbf16>
    // As output is the return value, not beneficial to move to L1, will always stay in DRAM.
    // CHECK: %{{.*}} = "ttnn.abs"{{.*}} -> tensor<{{.*}}, #[[DRAM]]>
    %15 = "ttir.abs"(%13, %14) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %15 : tensor<64x128xbf16>
  }
}
