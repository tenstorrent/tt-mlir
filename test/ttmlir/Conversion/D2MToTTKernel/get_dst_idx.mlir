// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @test_get_dst_idx_2x2
  func.func @test_get_dst_idx_2x2(%arg0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    d2m.await %arg0, %arg1 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_> into memref<4x!ttcore.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_> into memref<4x!ttcore.tile<32x32, f32>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_> into memref<4x!ttcore.tile<32x32, f32>, #l1_>
    %dst = d2m.acquire_dst() : memref<2x2x2x!ttcore.tile<32x32, f32>, #dst_>
    %collapse_shape_2 = memref.collapse_shape %dst [[0, 1, 2]] : memref<2x2x2x!ttcore.tile<32x32, f32>, #dst_> into memref<8x!ttcore.tile<32x32, f32>, #dst_>
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %0 = arith.muli %arg3, %c2 overflow<nsw> : index
      scf.for %arg4 = %c0 to %c2 step %c1 {
        %1 = arith.addi %0, %arg4 : index
        %2 = memref.load %collapse_shape[%1] : memref<4x!ttcore.tile<32x32, f32>, #l1_>
        memref.store %2, %collapse_shape_2[%1] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
        %3 = memref.load %collapse_shape_0[%1] : memref<4x!ttcore.tile<32x32, f32>, #l1_>
        %4 = arith.addi %1, %c4 : index
        memref.store %3, %collapse_shape_2[%4] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
      }
    }
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %0 = arith.muli %arg3, %c2 overflow<nsw> : index
      scf.for %arg4 = %c0 to %c2 step %c1 {
        %1 = arith.addi %0, %arg4 : index
        %2 = memref.load %collapse_shape_2[%1] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
        %3 = arith.addi %1, %c4 : index
        %4 = memref.load %collapse_shape_2[%3] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
        // CHECK: ttkernel.div_binary_tile_init
        // CHECK-NEXT: %[[DST_IDX:.*]] = arith{{.*}} : index
        // CHECK-NEXT: ttkernel.div_binary_tile(%{{.*}}, %{{.*}}, %[[DST_IDX]])
        %5 = "d2m.tile_div"(%2, %4) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        %6 = arith.addi %1, %c8 : index
        memref.store %5, %collapse_shape_2[%6] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
      }
    }
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %0 = arith.muli %arg3, %c2 overflow<nsw> : index
      scf.for %arg4 = %c0 to %c2 step %c1 {
        %1 = arith.addi %0, %arg4 : index
        %2 = arith.addi %1, %c8 : index
        %3 = memref.load %collapse_shape_2[%2] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
        memref.store %3, %collapse_shape_1[%1] : memref<4x!ttcore.tile<32x32, f32>, #l1_>
      }
    }
    d2m.yield %arg2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
    d2m.await %arg2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_get_dst_idx_2x1
  func.func @test_get_dst_idx_2x1(%arg0: memref<2x1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<2x1x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<2x1x!ttcore.tile<32x32, f32>, #l1_>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    d2m.await %arg0, %arg1 : (memref<2x1x!ttcore.tile<32x32, f32>, #l1_>, memref<2x1x!ttcore.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<2x1x!ttcore.tile<32x32, f32>, #l1_> into memref<2x!ttcore.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<2x1x!ttcore.tile<32x32, f32>, #l1_> into memref<2x!ttcore.tile<32x32, f32>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<2x1x!ttcore.tile<32x32, f32>, #l1_> into memref<2x!ttcore.tile<32x32, f32>, #l1_>
    %dst = d2m.acquire_dst() : memref<4x2x1x!ttcore.tile<32x32, f32>, #dst_>
    %collapse_shape_2 = memref.collapse_shape %dst [[0, 1, 2]] : memref<4x2x1x!ttcore.tile<32x32, f32>, #dst_> into memref<8x!ttcore.tile<32x32, f32>, #dst_>
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %0 = memref.load %collapse_shape[%arg3] : memref<2x!ttcore.tile<32x32, f32>, #l1_>
      memref.store %0, %collapse_shape_2[%arg3] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
      %1 = memref.load %collapse_shape_0[%arg3] : memref<2x!ttcore.tile<32x32, f32>, #l1_>
      %2 = arith.addi %arg3, %c2 : index
      memref.store %1, %collapse_shape_2[%2] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
    }
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %0 = memref.load %collapse_shape_2[%arg3] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
      %1 = arith.addi %arg3, %c2 : index
      %2 = memref.load %collapse_shape_2[%1] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
      // CHECK: ttkernel.div_binary_tile_init
      // CHECK-NEXT: %[[DST_IDX:.*]] = arith{{.*}} : index
      // CHECK-NEXT: ttkernel.div_binary_tile(%{{.*}}, %{{.*}}, %[[DST_IDX]])
      %3 = "d2m.tile_div"(%0, %2) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      %4 = arith.addi %arg3, %c4 : index
      memref.store %3, %collapse_shape_2[%4] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
    }
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %0 = arith.addi %arg3, %c4 : index
      %1 = memref.load %collapse_shape_2[%0] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
      memref.store %1, %collapse_shape_1[%arg3] : memref<2x!ttcore.tile<32x32, f32>, #l1_>
    }
    d2m.yield %arg2 : (memref<2x1x!ttcore.tile<32x32, f32>, #l1_>)
    d2m.await %arg2 : (memref<2x1x!ttcore.tile<32x32, f32>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_get_dst_idx_1x2
  func.func @test_get_dst_idx_1x2(%arg0: memref<1x2x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x2x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x2x!ttcore.tile<32x32, f32>, #l1_>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    d2m.await %arg0, %arg1 : (memref<1x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x2x!ttcore.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x2x!ttcore.tile<32x32, f32>, #l1_> into memref<2x!ttcore.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x2x!ttcore.tile<32x32, f32>, #l1_> into memref<2x!ttcore.tile<32x32, f32>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x2x!ttcore.tile<32x32, f32>, #l1_> into memref<2x!ttcore.tile<32x32, f32>, #l1_>
    %dst = d2m.acquire_dst() : memref<4x1x2x!ttcore.tile<32x32, f32>, #dst_>
    %collapse_shape_2 = memref.collapse_shape %dst [[0, 1, 2]] : memref<4x1x2x!ttcore.tile<32x32, f32>, #dst_> into memref<8x!ttcore.tile<32x32, f32>, #dst_>
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %0 = memref.load %collapse_shape[%arg3] : memref<2x!ttcore.tile<32x32, f32>, #l1_>
      memref.store %0, %collapse_shape_2[%arg3] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
      %1 = memref.load %collapse_shape_0[%arg3] : memref<2x!ttcore.tile<32x32, f32>, #l1_>
      %2 = arith.addi %arg3, %c2 : index
      memref.store %1, %collapse_shape_2[%2] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
    }
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %0 = memref.load %collapse_shape_2[%arg3] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
      %1 = arith.addi %arg3, %c2 : index
      %2 = memref.load %collapse_shape_2[%1] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
      // CHECK: ttkernel.div_binary_tile_init
      // CHECK-NEXT: %[[DST_IDX:.*]] = arith{{.*}} : index
      // CHECK-NEXT: ttkernel.div_binary_tile(%{{.*}}, %{{.*}}, %[[DST_IDX]])
      %3 = "d2m.tile_div"(%0, %2) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      %4 = arith.addi %arg3, %c4 : index
      memref.store %3, %collapse_shape_2[%4] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
    }
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %0 = arith.addi %arg3, %c4 : index
      %1 = memref.load %collapse_shape_2[%0] : memref<8x!ttcore.tile<32x32, f32>, #dst_>
      memref.store %1, %collapse_shape_1[%arg3] : memref<2x!ttcore.tile<32x32, f32>, #l1_>
    }
    d2m.yield %arg2 : (memref<1x2x!ttcore.tile<32x32, f32>, #l1_>)
    d2m.await %arg2 : (memref<1x2x!ttcore.tile<32x32, f32>, #l1_>)
    return
  }
}
