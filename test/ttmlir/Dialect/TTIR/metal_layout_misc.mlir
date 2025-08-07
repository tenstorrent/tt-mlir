// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

func.func @add_unaligned(%arg0: tensor<33x128xf32>, %arg1: tensor<33x128xf32>) -> tensor<33x128xf32> {
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    %0 = ttir.empty() : tensor<33x128xf32>
    // CHECK: "ttmetal.enqueue_program"
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<33x128xf32>, tensor<33x128xf32>, tensor<33x128xf32>) -> tensor<33x128xf32>
    // CHECK: "ttmetal.enqueue_read_buffer"
    // CHECK: "ttmetal.finish"
    return %1 : tensor<33x128xf32>
}

func.func @add_3d(%arg0: tensor<2x32x128xf32>, %arg1: tensor<2x32x128xf32>) -> tensor<2x32x128xf32> {
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    // CHECK-DAG:  = "ttmetal.create_buffer"{{.*}} : () -> memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    // CHECK-DAG:  = "ttmetal.create_buffer"{{.*}} : () -> memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    %0 = ttir.empty() : tensor<2x32x128xf32>
    // CHECK: "ttmetal.enqueue_program"
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x32x128xf32>, tensor<2x32x128xf32>, tensor<2x32x128xf32>) -> tensor<2x32x128xf32>
    // CHECK: "ttmetal.enqueue_read_buffer"
    // CHECK: "ttmetal.finish"
    return %1 : tensor<2x32x128xf32>
}

// For 3D tensors with shape 2x33x128:
// - Logical shape: 2x33x128
// - After collapse [[0,2], [2,3]] and tile-alignment: dims 0,1 -> 2 * alignUp(33, 32) -> 128, dim 2 -> alignUp(128, 32) -> 128
// - Tiled shape: 4x4 tiles
// - Physical buffer shape: 4x4x1x1
func.func @add_3d_unaligned(%arg0: tensor<2x33x128xf32>, %arg1: tensor<2x33x128xf32>) -> tensor<2x33x128xf32> {
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<4x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    %0 = ttir.empty() : tensor<2x33x128xf32>
    // CHECK: "ttmetal.enqueue_program"
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x33x128xf32>, tensor<2x33x128xf32>, tensor<2x33x128xf32>) -> tensor<2x33x128xf32>
    // CHECK: "ttmetal.enqueue_read_buffer"
    // CHECK: "ttmetal.finish"
    return %1 : tensor<2x33x128xf32>
}

// Test different 3D shapes
//  For 3D tensors with shape 4x64x256:
//  - Logical shape: 4x64x256
//  - After collapse [[0,2], [2,3]]: dims 0,1 -> 256, dim 2 -> 256
//  - With alignment 1x32x32: 256 stays 256 (8*32), 256 stays 256 (8*32)
//  - Tiled shape: 8x8 tiles
//  - Physical buffer shape: 8x8x1x1
func.func @add_3d_larger(%arg0: tensor<4x64x256xf32>, %arg1: tensor<4x64x256xf32>) -> tensor<4x64x256xf32> {
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<8x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<8x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<8x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    %0 = ttir.empty() : tensor<4x64x256xf32>
    // CHECK: "ttmetal.enqueue_program"
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<4x64x256xf32>, tensor<4x64x256xf32>, tensor<4x64x256xf32>) -> tensor<4x64x256xf32>
    // CHECK: "ttmetal.enqueue_read_buffer"
    // CHECK: "ttmetal.finish"
    return %1 : tensor<4x64x256xf32>
}

// Test unaligned in different dimensions
//  For 3D tensors with shape 2x32x130:
//  - Logical shape: 2x32x130
//  - After collapse [[0,2], [2,3]]: dims 0,1 -> 64, dim 2 -> 130
//  - With alignment 1x32x32: 64 stays 64 (2*32), 130 rounds to 160 (5*32)
//  - Tiled shape: 2x5 tiles
//  - Physical buffer shape: 2x5x1x1
func.func @add_3d_unaligned_last_dim(%arg0: tensor<2x32x130xf32>, %arg1: tensor<2x32x130xf32>) -> tensor<2x32x130xf32> {
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<2x5x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<2x5x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<2x5x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    %0 = ttir.empty() : tensor<2x32x130xf32>
    // CHECK: "ttmetal.enqueue_program"
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x32x130xf32>, tensor<2x32x130xf32>, tensor<2x32x130xf32>) -> tensor<2x32x130xf32>
    // CHECK: "ttmetal.enqueue_read_buffer"
    // CHECK: "ttmetal.finish"
    return %1 : tensor<2x32x130xf32>
}
