// Test for L1InterleavedFallbackAnalysis: Ops Typecast and Slice are currently not properly supported for L1 interleaved input nor output, as in that layout they break during runtime.
// Comparison operations are fine for L1 interleaved if consumed by a different op, showcased by the Abs operation.
// Example ops inspired by the YOLO model, can be used to check these ops when fixed as a trial for YOLO.
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=false l1-interleaved-fallback-analysis-enabled=true" -o %t_ttnn.mlir %s --mlir-print-debuginfo
// RUN: FileCheck %s --input-file=%t_ttnn.mlir

module @L1InterleavedRuntimeSupportTypecastSlice attributes {} {
  func.func @forward(%arg0: tensor<1x144x8400xbf16>, %arg1: tensor<1x64x8400xbf16>) -> tensor<1x16x8400xf32> {
    // CHECK-DAG: #[[DRAM_LAYOUT1:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT2:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT3:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT4:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[DRAM_LAYOUT5:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#dram>{{.*}}<interleaved>>
    // CHECK-DAG: #[[L1_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}#l1>{{.*}}<interleaved>>


    %0 = ttir.empty() : tensor<1x64x8400xbf16>
    // slice_static operation - not supported runtime upgrade to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.slice_static"{{.*}} -> tensor<1x64x8400xbf16, #[[DRAM_LAYOUT2]]>
    %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 64 : i32, 8400 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x144x8400xbf16>, tensor<1x64x8400xbf16>) -> tensor<1x64x8400xbf16>

    %2 = ttir.empty() : tensor<1x64x8400xbf16>
    // Greater than comparison - because of consumer being Typecast (not supported for input upgrade), not upgraded to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.gt"{{.*}} -> tensor<1x64x8400xbf16, #[[DRAM_LAYOUT2]]>
    %3 = "ttir.gt"(%1, %arg1, %2) : (tensor<1x64x8400xbf16>, tensor<1x64x8400xbf16>, tensor<1x64x8400xbf16>) -> tensor<1x64x8400xbf16>

    %4 = ttir.empty() : tensor<1x64x8400xf32>
    // Typecast operation - not supported runtime upgrade to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.typecast"{{.*}} -> tensor<1x64x8400xf32, #[[DRAM_LAYOUT4]]>
    %5 = "ttir.typecast"(%3, %4) : (tensor<1x64x8400xbf16>, tensor<1x64x8400xf32>) -> tensor<1x64x8400xf32>

    %6 = ttir.empty() : tensor<1x32x8400xf32>
    // CHECK: %{{.*}} = "ttnn.slice_static"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %7 = "ttir.slice_static"(%5, %6) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 8400 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x8400xf32>, tensor<1x32x8400xf32>) -> tensor<1x32x8400xf32>

    %8 = ttir.empty() : tensor<1x32x8400xbf16>
    // CHECK: %{{.*}} = "ttnn.slice_static"{{.*}} -> tensor<1x32x8400xbf16, #[[DRAM_LAYOUT5]]>
    %9 = "ttir.slice_static"(%arg1, %8) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 8400 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x8400xbf16>, tensor<1x32x8400xbf16>) -> tensor<1x32x8400xbf16>

    %10 = ttir.empty() : tensor<1x32x8400xf32>
    // CHECK: %{{.*}} = "ttnn.typecast"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %11 = "ttir.typecast"(%9, %10) : (tensor<1x32x8400xbf16>, tensor<1x32x8400xf32>) -> tensor<1x32x8400xf32>

    %12 = ttir.empty() : tensor<1x32x8400xf32>
    // Less than comparison - not upgraded to L1 because not immediately consumed by its user (schedule)
    // CHECK: %{{.*}} = "ttnn.lt"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %13 = "ttir.lt"(%7, %11, %12) : (tensor<1x32x8400xf32>, tensor<1x32x8400xf32>, tensor<1x32x8400xf32>) -> tensor<1x32x8400xf32>

    %14 = ttir.empty() : tensor<1x32x8400xbf16>
    // CHECK: %{{.*}} = "ttnn.slice_static"{{.*}} -> tensor<1x32x8400xbf16, #[[DRAM_LAYOUT5]]>
    %15 = "ttir.slice_static"(%arg1, %14) <{begins = [0 : i32, 32 : i32, 0 : i32], ends = [1 : i32, 64 : i32, 8400 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x8400xbf16>, tensor<1x32x8400xbf16>) -> tensor<1x32x8400xbf16>

    %16 = ttir.empty() : tensor<1x32x8400xf32>
    // CHECK: %{{.*}} = "ttnn.typecast"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %17 = "ttir.typecast"(%15, %16) : (tensor<1x32x8400xbf16>, tensor<1x32x8400xf32>) -> tensor<1x32x8400xf32>

    %18 = ttir.empty() : tensor<1x32x8400xf32>
    // Greater than comparison - because of consumer being Abs (supported for input upgrade), upgraded to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.gt"{{.*}} -> tensor<1x32x8400xf32, #[[L1_LAYOUT]]>
    %19 = "ttir.gt"(%13, %17, %18) : (tensor<1x32x8400xf32>, tensor<1x32x8400xf32>, tensor<1x32x8400xf32>) -> tensor<1x32x8400xf32>

    %20 = ttir.empty() : tensor<1x32x8400xf32>
    // Abs operation - because of consumer being Typecast (not supported for input upgrade), not upgraded to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.abs"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %21 = "ttir.abs"(%19, %20) : (tensor<1x32x8400xf32>, tensor<1x32x8400xf32>) -> tensor<1x32x8400xf32>

    %22 = ttir.empty() : tensor<1x32x8400xbf16>
    // CHECK: %{{.*}} = "ttnn.slice_static"{{.*}} -> tensor<1x32x8400xbf16, #[[DRAM_LAYOUT5]]>
    %23 = "ttir.slice_static"(%arg0, %22) <{begins = [0 : i32, 96 : i32, 0 : i32], ends = [1 : i32, 128 : i32, 8400 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x144x8400xbf16>, tensor<1x32x8400xbf16>) -> tensor<1x32x8400xbf16>

    %24 = ttir.empty() : tensor<1x32x8400xbf16>
    // CHECK: %{{.*}} = "ttnn.typecast"{{.*}} -> tensor<1x32x8400xbf16, #[[DRAM_LAYOUT5]]>
    %25 = "ttir.typecast"(%21, %24) : (tensor<1x32x8400xf32>, tensor<1x32x8400xbf16>) -> tensor<1x32x8400xbf16>

    %26 = ttir.empty() : tensor<1x32x8400xbf16>
    // Greater than or equal comparison - because of consumer being Typecast (not supported for input upgrade), not upgraded to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.ge"{{.*}} -> tensor<1x32x8400xbf16, #[[DRAM_LAYOUT5]]>
    %27 = "ttir.ge"(%25, %23, %26) : (tensor<1x32x8400xbf16>, tensor<1x32x8400xbf16>, tensor<1x32x8400xbf16>) -> tensor<1x32x8400xbf16>

    %28 = ttir.empty() : tensor<1x32x8400xf32>
    // CHECK: %{{.*}} = "ttnn.typecast"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %29 = "ttir.typecast"(%27, %28) : (tensor<1x32x8400xbf16>, tensor<1x32x8400xf32>) -> tensor<1x32x8400xf32>

    %30 = ttir.empty() : tensor<1x32x8400xf32>
    // Add operation - because of consumer being slice_static (not supported for input upgrade), not upgraded to L1 interleaved
    // CHECK: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x8400xf32, #[[DRAM_LAYOUT3]]>
    %31 = "ttir.add"(%29, %7, %30) : (tensor<1x32x8400xf32>, tensor<1x32x8400xf32>, tensor<1x32x8400xf32>) -> tensor<1x32x8400xf32>

    %32 = ttir.empty() : tensor<1x16x8400xf32>
    // As output is the return value, not beneficial to move to L1, will always stay in DRAM anyway.
    // CHECK: %{{.*}} = "ttnn.slice_static"{{.*}} -> tensor<1x16x8400xf32, #[[DRAM_LAYOUT3]]>
    %33 = "ttir.slice_static"(%31, %32) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 8400 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x8400xf32>, tensor<1x16x8400xf32>) -> tensor<1x16x8400xf32>

    return %33 : tensor<1x16x8400xf32>
  }
}
