// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

// CHECK-LABEL: func.func @add_unaligned
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

// CHECK-LABEL: func.func @add_3d
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
// CHECK-LABEL: func.func @add_3d_unaligned
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
// CHECK-LABEL: func.func @add_3d_larger
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
// CHECK-LABEL: func.func @add_3d_unaligned_last_dim
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

// Test 4D tensor that is not aligned in any of the dimensions
//  - Logical shape: 2x3x5x193
//  - After collapse [[0, 3], [3, 4]]:
//    - Alignments will be 1x1x32x256
//    - dims 0,1,2 -> alignUp(alignUp(alignUp(5, 32) * 3, 1) * 2, 32) -> 192
//    - dim 2 -> alignUp(193, 256) -> 224
//  - Tile shape: 6x7
//  - Physical buffer shape: 6x7x1x1
// CHECK-LABEL: func.func @add_4d_all_unaligned
func.func @add_4d_all_unaligned(%arg0: tensor<2x3x5x193xf32>, %arg1: tensor<2x3x5x193xf32>) -> tensor<2x3x5x193xf32> {
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<6x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<6x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    // CHECK-DAG: = "ttmetal.create_buffer"{{.*}} : () -> memref<6x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
    %0 = ttir.empty() : tensor<2x3x5x193xf32>
    // CHECK: "ttmetal.enqueue_program"
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x3x5x193xf32>, tensor<2x3x5x193xf32>, tensor<2x3x5x193xf32>) -> tensor<2x3x5x193xf32>
    // CHECK: "ttmetal.enqueue_read_buffer"
    // CHECK: "ttmetal.finish"
    return %1 : tensor<2x3x5x193xf32>
}
