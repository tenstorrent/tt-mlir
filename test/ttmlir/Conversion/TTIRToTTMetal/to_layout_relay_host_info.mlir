// RUN: ttmlir-opt --mlir-print-local-scope --convert-ttir-to-ttmetal -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module {
  // CHECK-LABEL: func.func @test_to_layout_host_layout_relay
  func.func @test_to_layout_host_layout_relay(%arg0: memref<43x7xf32>) -> memref<43x7xf32> {
    // CHECK: %[[DEV_TENSOR:.*]] = "ttmetal.create_buffer"()
    %alloc_0 = memref.alloc() {address = 0x1000 : i64, alignment = 16 : i64} : memref<2x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>

    // CHECK: "ttmetal.enqueue_write_buffer"(%{{.*}}, %[[DEV_TENSOR]]) {ttcore.host_layout = #ttcore.metal_layout[[DEV_TENSOR_LAYOUT:.*]]}
    ttir.to_layout %arg0, %alloc_0 : memref<43x7xf32> into memref<2x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>> hostInfo = <logical_shape = 43x7, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>

    // CHECK: %[[HOST_TENSOR:.*]] = memref.alloc()
    %alloc_1 = memref.alloc() : memref<43x7xf32>

    // CHECK: "ttmetal.enqueue_read_buffer"(%[[DEV_TENSOR]], %[[HOST_TENSOR]]) {ttcore.host_layout = #ttcore.metal_layout[[DEV_TENSOR_LAYOUT]]}
    ttir.to_layout %alloc_0, %alloc_1 : memref<2x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>> into memref<43x7xf32> hostInfo = <logical_shape = 43x7, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>

    // CHECK-NOT: memref.dealloc
    // CHECK: "ttmetal.deallocate_buffer"(%[[DEV_TENSOR]])
    memref.dealloc %alloc_0 : memref<2x1x32x32xf32, #ttcore.shard<128x4>, #ttcore.memory_space<l1>>

    // CHECK: return %[[HOST_TENSOR]]
    return %alloc_1 : memref<43x7xf32>
  }
}
