// RUN: ttmlir-opt --tt-register-device --convert-ttir-to-ttkernel %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #tt.memory_space<l1>

module {
  //===----------------------------------------------------------------------===//
  // TTIR FPU operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: func.func @test_add_lowering
  func.func @test_add_lowering(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0, %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>, memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    %1 = memref.load %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_add
    // CHECK: ttkernel.binary_op_init_common
    // CHECK: ttkernel.add_tiles_init
    // CHECK: ttkernel.add_tiles
    %2 = "ttir.tile_add"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    memref.store %2, %collapse_shape_1[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg2 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg2 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_mul_lowering
  func.func @test_mul_lowering(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0, %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>, memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    %1 = memref.load %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_mul
    // CHECK: ttkernel.binary_op_init_common
    // CHECK: ttkernel.mul_tiles_init
    // CHECK: ttkernel.mul_tiles
    %2 = "ttir.tile_mul"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    memref.store %2, %collapse_shape_1[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg2 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg2 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }

  //===----------------------------------------------------------------------===//
  // TTIR SFPU operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: func.func @test_max_lowering
  func.func @test_max_lowering(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0, %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>, memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    %1 = memref.load %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_max
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init
    // CHECK: ttkernel.copy_tile
    // CHECK: ttkernel.copy_tile_init
    // CHECK: ttkernel.copy_tile
    // CHECK: ttkernel.max_tile_init
    // CHECK: ttkernel.max_tile
    %2 = "ttir.tile_maximum"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    memref.store %2, %collapse_shape_1[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg2 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg2 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_div_lowering
  func.func @test_div_lowering(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0, %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>, memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    %1 = memref.load %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_div
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init
    // CHECK: ttkernel.copy_tile
    // CHECK: ttkernel.copy_tile_init
    // CHECK: ttkernel.copy_tile
    // CHECK: ttkernel.div_binary_tile_init
    // CHECK: ttkernel.div_binary_tile
    %2 = "ttir.tile_div"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    memref.store %2, %collapse_shape_1[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg2 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg2 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_exp_lowering
  func.func @test_exp_lowering(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_exp
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init
    // CHECK: ttkernel.copy_tile
    // CHECK: ttkernel.exp_tile_init
    // CHECK: ttkernel.exp_tile
    %1 = "ttir.tile_exp"(%0) : (!tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    memref.store %1, %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_sin_lowering
  func.func @test_sin_lowering(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_sin
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init
    // CHECK: ttkernel.copy_tile
    // CHECK: ttkernel.sin_tile_init
    // CHECK: ttkernel.sin_tile
    %1 = "ttir.tile_sin"(%0) : (!tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    memref.store %1, %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_sigmoid_lowering
  func.func @test_sigmoid_lowering(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_sigmoid
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init
    // CHECK: ttkernel.copy_tile
    // CHECK: ttkernel.sigmoid_tile_init
    // CHECK: ttkernel.sigmoid_tile
    %1 = "ttir.tile_sigmoid"(%0) : (!tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    memref.store %1, %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }

}
