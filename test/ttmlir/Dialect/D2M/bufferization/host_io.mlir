// RUN: ttmlir-opt --mlir-print-local-scope --ttir-bufferization-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @test_bounce_buffer_insertion
  func.func @test_bounce_buffer_insertion(%arg0: tensor<3x43x7xf32>, %arg1: tensor<7x43x7xf32>) -> tensor<9x43x7xf32> {
    // CHECK: %[[ARG0_DEV:.*]] = memref.alloc
    // CHECK-NOT: #ttcore.host_layout
    %0 = d2m.empty() : tensor<6x1x32x32xf32, #ttcore.metal_layout<logical_shape = 3x43x7, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>>
    // CHECK: %[[ARG0_BOUNCE:.*]] = memref.alloc
    // CHECK-SAME: #ttcore.host_layout<logical_shape = {{.*}} host_strides = {{.*}} host_volume =
    // CHECK: memref.copy %{{.*}}, %[[ARG0_BOUNCE]]
    // CHECK: d2m.to_layout %[[ARG0_BOUNCE]], %[[ARG0_DEV]]
    %1 = d2m.to_layout %arg0, %0 : tensor<3x43x7xf32> into tensor<6x1x32x32xf32, #ttcore.metal_layout<logical_shape = 3x43x7, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>> hostInfo = <logical_shape = 3x43x7, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1> -> tensor<6x1x32x32xf32, #ttcore.metal_layout<logical_shape = 3x43x7, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>>

    // CHECK: %[[ARG1_DEV:.*]] = memref.alloc
    // CHECK-NOT: #ttcore.host_layout
    %2 = d2m.empty() : tensor<8x1x64x32xf32, #ttcore.metal_layout<logical_shape = 7x43x7, dim_alignments = 256x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>>
    // CHECK: %[[ARG1_BOUNCE:.*]] = memref.alloc
    // CHECK-SAME: #ttcore.host_layout<logical_shape = {{.*}} host_strides = {{.*}} host_volume =
    // CHECK: memref.copy %{{.*}}, %[[ARG1_BOUNCE]]
    // CHECK: d2m.to_layout %[[ARG1_BOUNCE]], %[[ARG1_DEV]]
    %3 = d2m.to_layout %arg1, %2 : tensor<7x43x7xf32> into tensor<8x1x64x32xf32, #ttcore.metal_layout<logical_shape = 7x43x7, dim_alignments = 256x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>> hostInfo = <logical_shape = 7x43x7, dim_alignments = 256x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1> -> tensor<8x1x64x32xf32, #ttcore.metal_layout<logical_shape = 7x43x7, dim_alignments = 256x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>>


    // CHECK: %[[OUT:.*]] = memref.alloc
    // CHECK-NOT: #ttcore.host_layout
    %4 = d2m.empty() : tensor<9x43x7xf32>
    // CHECK: %[[OUT_DEV:.*]] = memref.alloc
    // CHECK-NOT: #ttcore.host_layout
    %5 = d2m.empty() : tensor<8x1x96x32xf32, #ttcore.metal_layout<logical_shape = 9x43x7, dim_alignments = 256x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>>
    // CHECK: %[[OUT_BOUNCE:.*]] = memref.alloc
    // CHECK-SAME: #ttcore.host_layout<logical_shape = {{.*}} host_strides = {{.*}} host_volume =
    // CHECK: d2m.to_layout %[[OUT_DEV]], %[[OUT_BOUNCE]]
    %6 = d2m.to_layout %5, %4 : tensor<8x1x96x32xf32, #ttcore.metal_layout<logical_shape = 9x43x7, dim_alignments = 256x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>> into tensor<9x43x7xf32> hostInfo = <logical_shape = 9x43x7, dim_alignments = 256x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1> -> tensor<9x43x7xf32>
    // CHECK: memref.copy %[[OUT_BOUNCE]], %[[OUT]]

    // CHECK: return %[[OUT]]
    return %6 : tensor<9x43x7xf32>
  }

  // CHECK-LABEL: func.func @test_skipped_bounce_buffer_insertion
  func.func @test_skipped_bounce_buffer_insertion(%arg0: tensor<8x32x32xf32>) -> tensor<8x32x32xf32> {
    // CHECK: %[[ARG0_DEV:.*]] = memref.alloc
    // CHECK-NOT: #ttcore.host_layout
    %0 = d2m.empty() : tensor<8x1x32x32xf32, #ttcore.metal_layout<logical_shape = 8x32x32, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>>
    // CHECK: d2m.to_layout %{{.*}}, %[[ARG0_DEV]]
    %1 = d2m.to_layout %arg0, %0 : tensor<8x32x32xf32> into tensor<8x1x32x32xf32, #ttcore.metal_layout<logical_shape = 8x32x32, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>> hostInfo = <logical_shape = 8x32x32, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1> -> tensor<8x1x32x32xf32, #ttcore.metal_layout<logical_shape = 8x32x32, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>>

    // CHECK: %[[OUT:.*]] = memref.alloc
    // CHECK-NOT: #ttcore.host_layout
    %2 = d2m.empty() : tensor<8x32x32xf32>
    // CHECK: %[[OUT_DEV:.*]] = memref.alloc
    // CHECK-NOT: #ttcore.host_layout
    %3 = d2m.empty() : tensor<8x1x32x32xf32, #ttcore.metal_layout<logical_shape = 8x32x32, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>>
    // CHECK: d2m.to_layout %[[OUT_DEV]], %[[OUT]]
    %4 = d2m.to_layout %3, %2 : tensor<8x1x32x32xf32, #ttcore.metal_layout<logical_shape = 8x32x32, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>> into tensor<8x32x32xf32> hostInfo = <logical_shape = 8x32x32, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1> -> tensor<8x32x32xf32>

    // CHECK: return %[[OUT]]
    return %4 : tensor<8x32x32xf32>
  }
}
