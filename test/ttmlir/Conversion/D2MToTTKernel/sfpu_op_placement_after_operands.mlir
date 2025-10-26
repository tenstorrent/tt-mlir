// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

func.func private @compute_kernel8(%arg0: memref<6x1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %arg1: memref<6x1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %arg2: memref<6x1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>) attributes {d2m.thread = #d2m.thread<compute>} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c2 = arith.constant 2 : index
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2]] : memref<6x1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> into memref<6x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1, 2]] : memref<6x1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> into memref<6x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1, 2]] : memref<6x1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> into memref<6x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  scf.for %arg3 = %c0 to %c6 step %c1 {
    %dst = d2m.acquire_dst() : memref<8x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
    %collapse_shape_2 = memref.collapse_shape %dst [[0, 1, 2, 3]] : memref<8x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>> into memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
    %0 = memref.load %collapse_shape[%arg3] : memref<6x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
    memref.store %0, %collapse_shape_2[%c0] : memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
    %1 = memref.load %collapse_shape_0[%arg3] : memref<6x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
    memref.store %1, %collapse_shape_2[%c1] : memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
    %2 = memref.load %collapse_shape_2[%c0] : memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
    %3 = memref.load %collapse_shape_2[%c1] : memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
    // CHECK: ttkernel.add_binary_tile_init
    // CHECK-NEXT: ttkernel.add_binary_tile
    %4 = "d2m.tile_add"(%2, %3) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    memref.store %4, %collapse_shape_2[%c2] : memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
    %5 = memref.load %collapse_shape_2[%c2] : memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
    memref.store %5, %collapse_shape_1[%arg3] : memref<6x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  }
  return
}
