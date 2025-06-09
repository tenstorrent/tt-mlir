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

  // CHECK-LABEL: func.func @test_sub_lowering
  func.func @test_sub_lowering(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0, %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>, memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    %1 = memref.load %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_sub
    // CHECK: ttkernel.binary_op_init_common
    // CHECK: ttkernel.sub_tiles_init
    // CHECK: ttkernel.sub_tiles
    %2 = "ttir.tile_sub"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
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
    // CHECK: "ttkernel.copy_tile_init"(%[[CB0:.+]]) :
    // CHECK-NEXT: "ttkernel.copy_tile"(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]]) :
    // CHECK: "ttkernel.copy_tile_init"(%[[CB1:.+]]) :
    // CHECK-NOT: "ttkernel.copy_tile"(%{{.+}}, %{{.+}}, %[[DST_IDX0]]) :
    // CHECK-NEXT: "ttkernel.copy_tile"(%[[CB1]], %{{.+}}, %{{.+}}) :
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
    // CHECK: "ttkernel.copy_tile_init"(%[[CB0:.+]]) :
    // CHECK-NEXT: "ttkernel.copy_tile"(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]]) :
    // CHECK: "ttkernel.copy_tile_init"(%[[CB1:.+]]) :
    // CHECK-NOT: "ttkernel.copy_tile"(%{{.+}}, %{{.+}}, %[[DST_IDX0]]) :
    // CHECK-NEXT: "ttkernel.copy_tile"(%[[CB1]], %{{.+}}, %{{.+}}) :
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
    // CHECK: "ttkernel.copy_tile_init"(%[[CB0:.+]]) :
    // CHECK-NEXT: "ttkernel.copy_tile"(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.exp_tile_init
    // CHECK: ttkernel.exp_tile
    %1 = "ttir.tile_exp"(%0) : (!tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    memref.store %1, %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_cos_lowering
  func.func @test_cos_lowering(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_cos
    // CHECK: ttkernel.init_sfpu
    // CHECK: "ttkernel.copy_tile_init"(%[[CB0:.+]]) :
    // CHECK-NEXT: "ttkernel.copy_tile"(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.cos_tile_init
    // CHECK: ttkernel.cos_tile
    %1 = "ttir.tile_cos"(%0) : (!tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    memref.store %1, %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_negative_lowering
  func.func @test_negative_lowering(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_neg
    // CHECK: ttkernel.init_sfpu
    // CHECK: "ttkernel.copy_tile_init"(%[[CB0:.+]]) :
    // CHECK-NEXT: "ttkernel.copy_tile"(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.negative_tile_init
    // CHECK: ttkernel.negative_tile
    %1 = "ttir.tile_negative"(%0) : (!tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    memref.store %1, %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_rsqrt_lowering
  func.func @test_rsqrt_lowering(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_rsqrt
    // CHECK: ttkernel.init_sfpu
    // CHECK: "ttkernel.copy_tile_init"(%[[CB0:.+]]) :
    // CHECK-NEXT: "ttkernel.copy_tile"(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.rsqrt_tile_init
    // CHECK: ttkernel.rsqrt_tile
    %1 = "ttir.tile_rsqrt"(%0) : (!tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
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
    // CHECK: "ttkernel.copy_tile_init"(%[[CB0:.+]]) :
    // CHECK-NEXT: "ttkernel.copy_tile"(%[[CB0]], %{{.+}}, %{{.+}}) :
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
    // CHECK: "ttkernel.copy_tile_init"(%[[CB0:.+]]) :
    // CHECK-NEXT: "ttkernel.copy_tile"(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.sigmoid_tile_init
    // CHECK: ttkernel.sigmoid_tile
    %1 = "ttir.tile_sigmoid"(%0) : (!tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    memref.store %1, %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_ceil_lowering
  func.func @test_ceil_lowering(%arg0: memref<1x1x!tt.tile<32x32, bf16>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, bf16>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0 : (memref<1x1x!tt.tile<32x32, bf16>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, bf16>, #l1_> into memref<1x!tt.tile<32x32, bf16>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, bf16>, #l1_> into memref<1x!tt.tile<32x32, bf16>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, bf16>, #l1_>
    // CHECK-NOT: ttir.tile_ceil
    // CHECK: ttkernel.init_sfpu
    // CHECK: "ttkernel.copy_tile_init"(%[[CB0:.+]]) :
    // CHECK-NEXT: "ttkernel.copy_tile"(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.rounding_op_tile_init
    // CHECK: ttkernel.ceil_tile
    %1 = "ttir.tile_ceil"(%0) : (!tt.tile<32x32, bf16>) -> !tt.tile<32x32, bf16>
    // CHECK: ttkernel.pack_tile
    memref.store %1, %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, bf16>, #l1_>
    ttir.yield %arg1 : (memref<1x1x!tt.tile<32x32, bf16>, #l1_>)
    ttir.await %arg1 : (memref<1x1x!tt.tile<32x32, bf16>, #l1_>)
    return
  }

  // CHECK-LABEL: func.func @test_ceil_lowering_f32
  func.func @test_ceil_lowering_f32(%arg0: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x!tt.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    ttir.await %arg0 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<1x1x!tt.tile<32x32, f32>, #l1_> into memref<1x!tt.tile<32x32, f32>, #l1_>
    %0 = memref.load %collapse_shape[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_ceil
    // CHECK: ttkernel.init_sfpu
    // CHECK: "ttkernel.copy_tile_init"(%[[CB0:.+]]) :
    // CHECK-NEXT: "ttkernel.copy_tile"(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.rounding_op_tile_init
    // CHECK: ttkernel.ceil_tile_float32
    %1 = "ttir.tile_ceil"(%0) : (!tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    memref.store %1, %collapse_shape_0[%c0] : memref<1x!tt.tile<32x32, f32>, #l1_>
    ttir.yield %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg1 : (memref<1x1x!tt.tile<32x32, f32>, #l1_>)
    return
  }
}
