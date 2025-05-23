// RUN: ttmlir-opt --tt-register-device --convert-ttir-to-ttkernel --canonicalize %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #tt.memory_space<l1>
module {
  func.func private @no_loops(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0, %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>, memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    %1 = memref.load %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    %2 = memref.load %collapse_shape_1[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: scf.if
    // CHECK-NOT: ttkernel.copy_tile_init
    // CHECK-NOT: ttkernel.copy_tile
    %3 = "ttir.tile_matmul"(%0, %1, %2) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    memref.store %3, %collapse_shape_1[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg2 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg2 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }

  func.func private @one_inner(%arg0: memref<1x8x!tt.tile<32x32, f16>, #l1_>, %arg1: memref<8x1x!tt.tile<32x32, f16>, #l1_>, %arg2: memref<1x1x!tt.tile<32x32, f16>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    ttir.await %arg0, %arg1 : (memref<1x8x!tt.tile<32x32, f16>, #l1_>, memref<8x1x!tt.tile<32x32, f16>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x8x!tt.tile<32x32, f16>, #l1_> into memref<8x!tt.tile<32x32, f16>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<8x1x!tt.tile<32x32, f16>, #l1_> into memref<8x!tt.tile<32x32, f16>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x1x!tt.tile<32x32, f16>, #l1_> into memref<1x!tt.tile<32x32, f16>, #l1_>
    // CHECK: scf.for [[INNER:[%a-zA-Z0-9_]+]]
    scf.for %arg3 = %c0 to %c8 step %c1 {
      %0 = memref.load %collapse_shape[%arg3] : memref<8x!tt.tile<32x32, f16>, #l1_>
      %1 = memref.load %collapse_shape_0[%arg3] : memref<8x!tt.tile<32x32, f16>, #l1_>
      %2 = memref.load %collapse_shape_1[%c0] : memref<1x!tt.tile<32x32, f16>, #l1_>
      // CHECK: [[COND:[%0-9]+]] = arith.cmpi ne, [[INNER]], %c0
      // CHECK: scf.if [[COND]] {
      // CHECK: "ttkernel.copy_tile_init"
      // CHECK: "ttkernel.copy_tile"
      // CHECK: }
      %3 = "ttir.tile_matmul"(%0, %1, %2) : (!tt.tile<32x32, f16>, !tt.tile<32x32, f16>, !tt.tile<32x32, f16>) -> !tt.tile<32x32, f16>
      memref.store %3, %collapse_shape_1[%c0] : memref<1x!tt.tile<32x32, f16>, #l1_>
    }
    ttir.yield %arg2 : (memref<1x1x!tt.tile<32x32, f16>, #l1_>)
    ttir.await %arg2 : (memref<1x1x!tt.tile<32x32, f16>, #l1_>)
    return
  }

  func.func private @one_outer_one_inner(%arg0: memref<1x8x!tt.tile<32x32, f16>, #l1_>, %arg1: memref<8x16x!tt.tile<32x32, f16>, #l1_>, %arg2: memref<1x16x!tt.tile<32x32, f16>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    ttir.await %arg0, %arg1 : (memref<1x8x!tt.tile<32x32, f16>, #l1_>, memref<8x16x!tt.tile<32x32, f16>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x8x!tt.tile<32x32, f16>, #l1_> into memref<8x!tt.tile<32x32, f16>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<8x16x!tt.tile<32x32, f16>, #l1_> into memref<128x!tt.tile<32x32, f16>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x16x!tt.tile<32x32, f16>, #l1_> into memref<16x!tt.tile<32x32, f16>, #l1_>
    // CHECK: scf.for
    scf.for %arg3 = %c0 to %c16 step %c1 {
      // CHECK: scf.for [[INNER:[%a-zA-Z0-9_]+]]
      scf.for %arg4 = %c0 to %c8 step %c1 {
        %0 = memref.load %collapse_shape[%arg4] : memref<8x!tt.tile<32x32, f16>, #l1_>
        %1 = arith.muli %arg4, %c16 overflow<nsw> : index
        %2 = arith.addi %1, %arg3 : index
        %3 = memref.load %collapse_shape_0[%2] : memref<128x!tt.tile<32x32, f16>, #l1_>
        %4 = memref.load %collapse_shape_1[%arg3] : memref<16x!tt.tile<32x32, f16>, #l1_>
        // CHECK: [[COND:[%0-9]+]] = arith.cmpi ne, [[INNER]], %c0
        // CHECK: scf.if [[COND]] {
        // CHECK: "ttkernel.copy_tile_init"
        // CHECK: "ttkernel.copy_tile"
        // CHECK: }
        %5 = "ttir.tile_matmul"(%0, %3, %4) : (!tt.tile<32x32, f16>, !tt.tile<32x32, f16>, !tt.tile<32x32, f16>) -> !tt.tile<32x32, f16>
        memref.store %5, %collapse_shape_1[%arg3] : memref<16x!tt.tile<32x32, f16>, #l1_>
      }
    }
    ttir.yield %arg2 : (memref<1x16x!tt.tile<32x32, f16>, #l1_>)
    ttir.await %arg2 : (memref<1x16x!tt.tile<32x32, f16>, #l1_>)
    return
  }

  func.func private @two_outer_one_inner(%arg0: memref<4x8x!tt.tile<32x32, f16>, #l1_>, %arg1: memref<8x16x!tt.tile<32x32, f16>, #l1_>, %arg2: memref<4x16x!tt.tile<32x32, f16>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    ttir.await %arg0, %arg1 : (memref<4x8x!tt.tile<32x32, f16>, #l1_>, memref<8x16x!tt.tile<32x32, f16>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<4x8x!tt.tile<32x32, f16>, #l1_> into memref<32x!tt.tile<32x32, f16>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<8x16x!tt.tile<32x32, f16>, #l1_> into memref<128x!tt.tile<32x32, f16>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<4x16x!tt.tile<32x32, f16>, #l1_> into memref<64x!tt.tile<32x32, f16>, #l1_>
    // CHECK: scf.for
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %0 = arith.muli %arg3, %c8 overflow<nsw> : index
      %1 = arith.muli %arg3, %c16 overflow<nsw> : index
      // CHECK: scf.for
      scf.for %arg4 = %c0 to %c16 step %c1 {
        %2 = arith.addi %1, %arg4 : index
        // CHECK: scf.for [[INNER:[%a-zA-Z0-9_]+]]
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %3 = arith.addi %0, %arg5 : index
          %4 = memref.load %collapse_shape[%3] : memref<32x!tt.tile<32x32, f16>, #l1_>
          %5 = arith.muli %arg5, %c16 overflow<nsw> : index
          %6 = arith.addi %5, %arg4 : index
          %7 = memref.load %collapse_shape_0[%6] : memref<128x!tt.tile<32x32, f16>, #l1_>
          %8 = memref.load %collapse_shape_1[%2] : memref<64x!tt.tile<32x32, f16>, #l1_>
          // CHECK: [[COND:[%0-9]+]] = arith.cmpi ne, [[INNER]], %c0
          // CHECK: scf.if [[COND]] {
          // CHECK: "ttkernel.copy_tile_init"
          // CHECK: "ttkernel.copy_tile"
          // CHECK: }
          %9 = "ttir.tile_matmul"(%4, %7, %8) : (!tt.tile<32x32, f16>, !tt.tile<32x32, f16>, !tt.tile<32x32, f16>) -> !tt.tile<32x32, f16>
          memref.store %9, %collapse_shape_1[%2] : memref<64x!tt.tile<32x32, f16>, #l1_>
        }
      }
    }
    ttir.yield %arg2 : (memref<4x16x!tt.tile<32x32, f16>, #l1_>)
    ttir.await %arg2 : (memref<4x16x!tt.tile<32x32, f16>, #l1_>)
    return
  }

  // The following is not a valid MM example, but it exercises the posibility where an inner dim loop is not the innermost loop in the IR.
  func.func private @outer_inner_outer(%arg0: memref<4x8x!tt.tile<32x32, f16>, #l1_>, %arg1: memref<8x16x!tt.tile<32x32, f16>, #l1_>, %arg2: memref<4x16x!tt.tile<32x32, f16>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    ttir.await %arg0, %arg1 : (memref<4x8x!tt.tile<32x32, f16>, #l1_>, memref<8x16x!tt.tile<32x32, f16>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<4x8x!tt.tile<32x32, f16>, #l1_> into memref<32x!tt.tile<32x32, f16>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<8x16x!tt.tile<32x32, f16>, #l1_> into memref<128x!tt.tile<32x32, f16>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<4x16x!tt.tile<32x32, f16>, #l1_> into memref<64x!tt.tile<32x32, f16>, #l1_>
    // CHECK: scf.for
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %0 = arith.muli %arg3, %c8 overflow<nsw> : index
      %1 = arith.muli %arg3, %c16 overflow<nsw> : index
      // CHECK: scf.for [[INNER:[%a-zA-Z0-9_]+]]
      scf.for %arg4 = %c0 to %c16 step %c1 {
        // CHECK: scf.for
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %3 = arith.addi %0, %arg5 : index
          %4 = memref.load %collapse_shape[%3] : memref<32x!tt.tile<32x32, f16>, #l1_>
          %5 = arith.muli %arg5, %c16 overflow<nsw> : index
          %6 = arith.addi %5, %arg4 : index
          %10 = arith.muli %arg3, %arg5 overflow<nsw> : index
          %7 = memref.load %collapse_shape_0[%6] : memref<128x!tt.tile<32x32, f16>, #l1_>
          %8 = memref.load %collapse_shape_1[%10] : memref<64x!tt.tile<32x32, f16>, #l1_>
          // CHECK: [[COND:[%0-9]+]] = arith.cmpi ne, [[INNER]], %c0
          // CHECK: scf.if [[COND]] {
          // CHECK: "ttkernel.copy_tile_init"
          // CHECK: "ttkernel.copy_tile"
          // CHECK: }
          %9 = "ttir.tile_matmul"(%4, %7, %8) : (!tt.tile<32x32, f16>, !tt.tile<32x32, f16>, !tt.tile<32x32, f16>) -> !tt.tile<32x32, f16>
          memref.store %9, %collapse_shape_1[%10] : memref<64x!tt.tile<32x32, f16>, #l1_>
        }
      }
    }
    ttir.yield %arg2 : (memref<4x16x!tt.tile<32x32, f16>, #l1_>)
    ttir.await %arg2 : (memref<4x16x!tt.tile<32x32, f16>, #l1_>)
    return
  }

  // The following is not a valid MM example, but it exercises the posibility where there are multiple inner loops.
  // i.e. Multiple loop variables that do not contribute to the load index.
  func.func private @two_inner(%arg0: memref<4x8x!tt.tile<32x32, f16>, #l1_>, %arg1: memref<8x16x!tt.tile<32x32, f16>, #l1_>, %arg2: memref<4x16x!tt.tile<32x32, f16>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    ttir.await %arg0, %arg1 : (memref<4x8x!tt.tile<32x32, f16>, #l1_>, memref<8x16x!tt.tile<32x32, f16>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<4x8x!tt.tile<32x32, f16>, #l1_> into memref<32x!tt.tile<32x32, f16>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<8x16x!tt.tile<32x32, f16>, #l1_> into memref<128x!tt.tile<32x32, f16>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<4x16x!tt.tile<32x32, f16>, #l1_> into memref<64x!tt.tile<32x32, f16>, #l1_>
    // CHECK: scf.for [[INNER1:[%a-zA-Z0-9_]+]]
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %0 = arith.muli %arg3, %c8 overflow<nsw> : index
      %1 = arith.muli %arg3, %c16 overflow<nsw> : index
        // CHECK: scf.for [[INNER2:[%a-zA-Z0-9_]+]]
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %3 = arith.addi %0, %arg5 : index
          %4 = memref.load %collapse_shape[%3] : memref<32x!tt.tile<32x32, f16>, #l1_>
          %5 = arith.muli %arg3, %c16 overflow<nsw> : index
          %6 = arith.addi %5, %arg5 : index
          %7 = memref.load %collapse_shape_0[%6] : memref<128x!tt.tile<32x32, f16>, #l1_>
          %8 = memref.load %collapse_shape_1[%c0] : memref<64x!tt.tile<32x32, f16>, #l1_>
          // CHECK: [[COND1:[%0-9]+]] = arith.cmpi ne, [[INNER1]], %c0
          // CHECK: [[COND2:[%0-9]+]] = arith.cmpi ne, [[INNER2]], %c0
          // CHECK: [[AND:[%0-9]+]] = arith.ori [[COND1]], [[COND2]]
          // CHECK: scf.if [[AND]] {
          // CHECK: "ttkernel.copy_tile_init"
          // CHECK: "ttkernel.copy_tile"
          // CHECK: }
          %9 = "ttir.tile_matmul"(%4, %7, %8) : (!tt.tile<32x32, f16>, !tt.tile<32x32, f16>, !tt.tile<32x32, f16>) -> !tt.tile<32x32, f16>
          memref.store %9, %collapse_shape_1[%c0] : memref<64x!tt.tile<32x32, f16>, #l1_>
        }
    }
    ttir.yield %arg2 : (memref<4x16x!tt.tile<32x32, f16>, #l1_>)
    ttir.await %arg2 : (memref<4x16x!tt.tile<32x32, f16>, #l1_>)
    return
  }
}
