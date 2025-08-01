// Test for L1InterleavedFallbackAnalysis: simple fork-join pattern with some sharded inputs
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false l1-interleaved-fallback-analysis-enabled=true max-legal-layouts=32 override-output-layout=op=l1:height_sharded" -o %t_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t_ttnn.mlir

module @L1InterleavedTestForkJoin attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %arg2: tensor<64x128xbf16>, %arg3: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    %0 = ttir.empty() : tensor<64x128xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %2 = ttir.empty() : tensor<64x128xbf16>
    %3 = "ttir.add"(%arg2, %arg3, %2) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %4 = ttir.empty() : tensor<64x128xbf16>
    %5 = "ttir.add"(%1, %3, %4) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %6 = ttir.empty() : tensor<64x128xbf16>
    %7 = "ttir.relu"(%5, %6) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // Fork: %7 used in two ops
    %8 = ttir.empty() : tensor<64x128xbf16>
    %9 = "ttir.neg"(%7, %8) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16> loc(#loc)
    %10 = ttir.empty() : tensor<64x128xbf16>
    %11 = "ttir.abs"(%7, %10) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16> loc(#loc)
    // Join: add the results
    %12 = ttir.empty() : tensor<64x128xbf16>
    %13 = "ttir.add"(%9, %11, %12) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %13 : tensor<64x128xbf16>
  }
}
#loc = loc("op")
// CHECK-DAG: #[[DRAM:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
// CHECK-DAG: #[[L1:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#l1>{{.*}}<interleaved>>
// CHECK-DAG: #[[SHARDED:.*]] = #ttnn.ttnn_layout<{{.*}}, <height_sharded>>

// not immediately consumed -> dram
// CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<{{.*}}, #[[DRAM]]>

// CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<{{.*}}, #[[L1]]>
// CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<{{.*}}, #[[L1]]>

// fork, not a single user -> dram
// CHECK: %{{.*}} = "ttnn.relu"{{.*}} -> tensor<{{.*}}, #[[DRAM]]>

// not immediately consumed, not backend supported -> dram
// CHECK: %{{.*}} = "ttnn.neg"{{.*}} -> tensor<{{.*}}, #[[SHARDED]]>
// not backend supported -> dram
// CHECK: %{{.*}} = "ttnn.abs"{{.*}} -> tensor<{{.*}}, #[[SHARDED]]>
// CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<{{.*}}, #[[L1]]>
