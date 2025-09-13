

func.func private @compute_kernel8_loop(%arg0: memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %arg1: memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %arg2: memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>) attributes {ttir.thread = #ttir.thread<compute>} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  ttir.await %arg0, %arg1 : (memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>)
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> into memref<16x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> into memref<16x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> into memref<16x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  scf.for %arg3 = %c0 to %c2 step %c1 {
    %0 = arith.muli %arg3, %c2 overflow<nsw> : index
    %dst = ttir.acquire_dst() : memref<1x2x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
    %collapse_shape_2 = memref.collapse_shape %dst [[0, 1, 2]] : memref<1x2x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>> into memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
    scf.for %arg4 = %c0 to %c8 step %c1 {
      %1 = arith.remsi %arg4, %c4 : index
      %2 = arith.divsi %arg4, %c4 : index
      %3 = arith.addi %0, %2 : index
      %4 = arith.muli %3, %c4 overflow<nsw> : index
      %5 = arith.addi %4, %1 : index
      %6 = memref.load %collapse_shape[%5] : memref<16x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %7 = memref.load %collapse_shape_0[%5] : memref<16x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %8 = "ttir.tile_add"(%6, %7) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      %9 = arith.muli %2, %c4 overflow<nsw> : index
      %10 = arith.addi %9, %1 : index
      memref.store %8, %collapse_shape_2[%10] : memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
    }
    scf.for %arg4 = %c0 to %c8 step %c1 {
      %1 = arith.remsi %arg4, %c4 : index
      %2 = arith.divsi %arg4, %c4 : index
      %3 = arith.muli %2, %c4 overflow<nsw> : index
      %4 = arith.addi %3, %1 : index
      %5 = memref.load %collapse_shape_2[%4] : memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
      %6 = arith.addi %0, %2 : index
      %7 = arith.muli %6, %c4 overflow<nsw> : index
      %8 = arith.addi %7, %1 : index
      memref.store %5, %collapse_shape_1[%8] : memref<16x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
    }
  }
  ttir.yield %arg2 : (memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>)
  ttir.await %arg2 : (memref<4x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>)
  return
}

func.func private @compute_kernel8_no_loop(%arg0: memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %arg1: memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, %arg2: memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>) attributes {ttir.thread = #ttir.thread<compute>} {
  %c0 = arith.constant 0 : index
  ttir.await %arg0, %arg1 : (memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>)
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> into memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> into memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> into memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  %dst = ttir.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
  %collapse_shape_2 = memref.collapse_shape %dst [[0, 1, 2]] : memref<8x1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>> into memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
  %0 = memref.load %collapse_shape[%c0] : memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  %1 = memref.load %collapse_shape_0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  %2 = "ttir.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
  memref.store %2, %collapse_shape_2[%c0] : memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
  %3 = memref.load %collapse_shape_2[%c0] : memref<8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
  memref.store %3, %collapse_shape_1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
  ttir.yield %arg2 : (memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>)
  ttir.await %arg2 : (memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>)
  return
}
