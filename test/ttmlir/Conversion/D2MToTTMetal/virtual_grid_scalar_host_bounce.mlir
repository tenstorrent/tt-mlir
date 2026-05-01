// RUN: ttmlir-opt --ttcore-register-device="mock-system-desc-arch=blackhole" --ttir-to-ttmetal-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module {
  // CHECK-LABEL: func.func @slice_f32_cb_page_compatible
  // CHECK-NOT: memref<8x8x384x4xf32
  // CHECK: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64, virtualGridForwardMapping = #map{{[0-9]*}}, virtualGridInverseMapping = #map{{[0-9]*}}}> : () -> memref<3x16x2x1x2x4x32x32xf32, #ttcore.shard<16384x4096x128x4, 1>, #l1>
  // CHECK: "ttmetal.enqueue_read_buffer"{{.*}} : (memref<3x32x1x1x1x1x32x32xf32, #ttcore.shard<4096x4096x128x4, 1>, #l1>
  func.func @slice_f32_cb_page_compatible(%arg0: tensor<6x64x64x4xf32>) -> tensor<3x32x32x2xf32> {
    %0 = "ttir.slice_static"(%arg0) <{
      begins = [1 : i32, 15 : i32, 16 : i32, 2 : i32],
      ends = [6 : i32, 47 : i32, 48 : i32, 4 : i32],
      step = [2 : i32, 1 : i32, 1 : i32, 1 : i32]
    }> : (tensor<6x64x64x4xf32>) -> tensor<3x32x32x2xf32>
    return %0 : tensor<3x32x32x2xf32>
  }

  // CHECK-LABEL: func.func @slice_bf16_noc_aligned
  // CHECK: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64, virtualGridForwardMapping = #map{{[0-9]*}}, virtualGridInverseMapping = #map{{[0-9]*}}}> : () -> memref<4x4x2x1x1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048x2048x2048, 1>, #l1>
  // CHECK: "ttmetal.enqueue_read_buffer"{{.*}} : (memref<4x4x2x1x1x1x32x32xbf16, #ttcore.shard<2048x2048x64x2, 1>, #l1>
  func.func @slice_bf16_noc_aligned(%arg0: tensor<8x6x64x32xbf16>) -> tensor<4x4x64x32xbf16> {
    %0 = "ttir.slice_static"(%arg0) <{
      begins = [2 : i32, 1 : i32, 0 : i32, 0 : i32],
      ends = [6 : i32, 5 : i32, 64 : i32, 32 : i32],
      step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
    }> : (tensor<8x6x64x32xbf16>) -> tensor<4x4x64x32xbf16>
    return %0 : tensor<4x4x64x32xbf16>
  }

  // CHECK-LABEL: func.func @permute_f32_virtual_grid_readback
  // CHECK-NOT: memref<6x8x96x16xf32
  // CHECK: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64, virtualGridForwardMapping = #map{{[0-9]*}}, virtualGridInverseMapping = #map{{[0-9]*}}}> : () -> memref<1x18x1x4x1x1x32x32xf32, #ttcore.shard<4096x4096x128x4, 1>, #l1>
  // CHECK: "ttmetal.enqueue_read_buffer"{{.*}} : (memref<18x1x1x4x1x1x32x32xf32, #ttcore.shard<4096x4096x128x4, 1>, #l1>
  func.func @permute_f32_virtual_grid_readback(%arg0: tensor<1x18x8x128xf32>) -> tensor<18x1x8x128xf32> {
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 2, 3>}> : (tensor<1x18x8x128xf32>) -> tensor<18x1x8x128xf32>
    return %0 : tensor<18x1x8x128xf32>
  }

  // CHECK-LABEL: func.func @reduce_min_f32_keep1_block_grid
  // CHECK-NOT: virtualGridForwardMapping
  // CHECK: "ttmetal.enqueue_read_buffer"{{.*}} : (memref<10x1x64x32xf32, #ttcore.shard<128x4, 1>, #l1>
  func.func @reduce_min_f32_keep1_block_grid(%arg0: tensor<512x128xf32>) -> tensor<512x1xf32> {
    %0 = "ttir.min"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<512x128xf32>) -> tensor<512x1xf32>
    return %0 : tensor<512x1xf32>
  }

  // CHECK-LABEL: func.func @reduce_max_bf16_keep0_block_grid
  // CHECK-NOT: virtualGridForwardMapping
  // CHECK: "ttmetal.enqueue_read_buffer"{{.*}} : (memref<1x4x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
  func.func @reduce_max_bf16_keep0_block_grid(%arg0: tensor<512x128xbf16>) -> tensor<128xbf16> {
    %0 = "ttir.max"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<512x128xbf16>) -> tensor<128xbf16>
    return %0 : tensor<128xbf16>
  }
}
