// RUN: ttmlir-opt --ttmetal-generate-dim-aligned-host-io -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module {
  // CHECK-LABEL: func.func @test_bounce_buffer_insertion
  func.func @test_bounce_buffer_insertion(%arg0: memref<3x43x7xf32>, %arg1: memref<7x43x7xf32>) -> memref<9x43x7xf32> {
    // CHECK: %[[DEV_TENSOR_0:.*]] = "ttmetal.create_buffer"
    %0 = "ttmetal.create_buffer"() <{address = 1024 : i64}> : () -> memref<6x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
    // CHECK-NEXT: %[[ARG0_BOUNCE:.*]] = memref.alloc() : memref<{{.*}}, #ttcore.host_layout<shape = {{.*}}, alignments = {{.*}}>
    // CHECK-NEXT: memref.copy %{{.*}}, %[[ARG0_BOUNCE]]
    // CHECK-NEXT: "ttmetal.enqueue_write_buffer"(%[[ARG0_BOUNCE]], %[[DEV_TENSOR_0]]) : ({{.*}} #ttcore.host_layout
    "ttmetal.enqueue_write_buffer"(%arg0, %0) {ttcore.host_layout = #ttcore.metal_layout<logical_shape = 3x43x7, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>} : (memref<3x43x7xf32>, memref<6x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>) -> ()
    "ttmetal.deallocate_buffer"(%0) : (memref<6x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>) -> ()

    // CHECK: %[[DEV_TENSOR_1:.*]] = "ttmetal.create_buffer"
    %1 = "ttmetal.create_buffer"() <{address = 32768 : i64}> : () -> memref<8x1x64x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
    // CHECK-NEXT: %[[ARG1_BOUNCE:.*]] = memref.alloc() : memref<{{.*}}, #ttcore.host_layout<shape = {{.*}}, alignments = {{.*}}>
    // CHECK-NEXT: memref.copy %{{.*}}, %[[ARG1_BOUNCE]]
    // CHECK-NEXT: "ttmetal.enqueue_write_buffer"(%[[ARG1_BOUNCE]], %[[DEV_TENSOR_1]]) : ({{.*}} #ttcore.host_layout
    "ttmetal.enqueue_write_buffer"(%arg1, %1) {ttcore.host_layout = #ttcore.metal_layout<logical_shape = 7x43x7, dim_alignments = 4x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>} : (memref<7x43x7xf32>, memref<8x1x64x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>) -> ()
    "ttmetal.deallocate_buffer"(%1) : (memref<8x1x64x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>) -> ()

    // CHECK: %[[ARG_OUT:.*]] = memref.alloc() : memref
    // CHECK-NOT: #ttcore.host_layout
    %alloc = memref.alloc() : memref<9x43x7xf32>
    // CHECK: %[[DEV_TENSOR_2:.*]] = "ttmetal.create_buffer"
    %2 = "ttmetal.create_buffer"() <{address = 65536 : i64}> : () -> memref<8x1x96x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
    // CHECK-NEXT: %[[ARG_OUT_BOUNCE:.*]] = memref.alloc() : memref<{{.*}}, #ttcore.host_layout<shape = {{.*}}, alignments = {{.*}}>
    // CHECK-NEXT: "ttmetal.enqueue_read_buffer"(%[[DEV_TENSOR_2]], %[[ARG_OUT_BOUNCE]]) : ({{.*}} #ttcore.host_layout
    "ttmetal.enqueue_read_buffer"(%2, %alloc) {ttcore.host_layout = #ttcore.metal_layout<logical_shape = 9x43x7, dim_alignments = 4x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>} : (memref<8x1x96x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>, memref<9x43x7xf32>) -> ()
    // CHECK-NEXT: memref.copy %[[ARG_OUT_BOUNCE]], %[[ARG_OUT]]
    "ttmetal.finish"() : () -> ()
    "ttmetal.deallocate_buffer"(%2) : (memref<8x1x96x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>) -> ()
    // CHECK: return %[[ARG_OUT]]
    return %alloc : memref<9x43x7xf32>
  }

  // CHECK-LABEL: func.func @test_skipped_bounce_buffer_insertion
  func.func @test_skipped_bounce_buffer_insertion(%arg0: memref<8x32x32xf32>) -> memref<8x32x32xf32> {
    // CHECK: "ttmetal.create_buffer"
    %0 = "ttmetal.create_buffer"() <{address = 103872 : i64}> : () -> memref<8x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
    // CHECK-NEXT: "ttmetal.enqueue_write_buffer"
    // CHECK-NOT: ttcore.host_layout
    "ttmetal.enqueue_write_buffer"(%arg0, %0) {ttcore.host_layout = #ttcore.metal_layout<logical_shape = 8x32x32, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>} : (memref<8x32x32xf32>, memref<8x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>) -> ()
    // CHECK-NEXT: "ttmetal.deallocate_buffer"
    "ttmetal.deallocate_buffer"(%0) : (memref<8x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>) -> ()
    // CHECK-NEXT: %[[OUT_BUF:.*]] = memref.alloc
    // CHECK-NOT: #ttcore.host_layout
    %alloc = memref.alloc() : memref<8x32x32xf32>
    // CHECK-NEXT: "ttmetal.create_buffer"
    %1 = "ttmetal.create_buffer"() <{address = 99776 : i64}> : () -> memref<8x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>
    // CHECK-NEXT: "ttmetal.enqueue_read_buffer"(%{{.*}}, %[[OUT_BUF]])
    // CHECK-NOT: ttcore.host_layout
    "ttmetal.enqueue_read_buffer"(%1, %alloc) {ttcore.host_layout = #ttcore.metal_layout<logical_shape = 8x32x32, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1>} : (memref<8x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>, memref<8x32x32xf32>) -> ()
    "ttmetal.finish"() : () -> ()
    "ttmetal.deallocate_buffer"(%1) : (memref<8x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>) -> ()
    // CHECK: return %[[OUT_BUF]]
    return %alloc : memref<8x32x32xf32>
  }
}
