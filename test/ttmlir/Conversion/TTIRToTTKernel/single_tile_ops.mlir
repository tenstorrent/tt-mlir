// RUN: ttmlir-opt --ttcore-register-device --ttir-insert-dst-register-access --lower-affine --ttir-generic-linearize-memref --lower-affine --convert-ttir-to-ttkernel --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>

module {
  //===----------------------------------------------------------------------===//
  // TTIR FPU operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: func.func @test_matmul_lowering
  func.func @test_matmul_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    %2 = affine.load %arg2[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_matmul
    // CHECK: ttkernel.mm_init
    // CHECK: ttkernel.mm_init_short
    // CHECK: ttkernel.matmul_tiles
    %3 = "ttir.tile_matmul"(%0, %1, %2) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %3, %arg2[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_add_lowering
  func.func @test_add_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_add
    // CHECK: ttkernel.binary_op_init_common
    // CHECK: ttkernel.add_tiles_init
    // CHECK: ttkernel.add_tiles
    %2 = "ttir.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %2, %arg2[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_sub_lowering
  func.func @test_sub_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_sub
    // CHECK: ttkernel.binary_op_init_common
    // CHECK: ttkernel.sub_tiles_init
    // CHECK: ttkernel.sub_tiles
    %2 = "ttir.tile_sub"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %2, %arg2[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_mul_lowering
  func.func @test_mul_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_mul
    // CHECK: ttkernel.binary_op_init_common
    // CHECK: ttkernel.mul_tiles_init
    // CHECK: ttkernel.mul_tiles
    %2 = "ttir.tile_mul"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %2, %arg2[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  //===----------------------------------------------------------------------===//
  // TTIR SFPU operations
  //===----------------------------------------------------------------------===//

  // CHECK-LABEL: func.func @test_max_lowering
  func.func @test_max_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_max
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]]) :
    // CHECK: ttkernel.copy_tile_init(%[[CB1:.+]]) :
    // CHECK-NOT: ttkernel.copy_tile(%{{.+}}, %{{.+}}, %[[DST_IDX0]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB1]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.max_tile_init
    // CHECK: ttkernel.max_tile(%{{.+}}, %{{.+}})
    %2 = "ttir.tile_maximum"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %2, %arg2[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_div_lowering
  func.func @test_div_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_div
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]]) :
    // CHECK: ttkernel.copy_tile_init(%[[CB1:.+]]) :
    // CHECK-NOT: ttkernel.copy_tile(%{{.+}}, %{{.+}}, %[[DST_IDX0]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB1]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.div_binary_tile_init
    // CHECK: ttkernel.div_binary_tile(%{{.+}}, %{{.+}}, %{{.+}})
    %2 = "ttir.tile_div"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %2, %arg2[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_recip_lowering
  func.func @test_recip_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_recip
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.recip_tile_init
    // CHECK: ttkernel.recip_tile
    %1 = "ttir.tile_recip"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_pow_lowering
  func.func @test_pow_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    %1 = affine.load %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_pow
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]]) :
    // CHECK: ttkernel.copy_tile_init(%[[CB1:.+]]) :
    // CHECK-NOT: ttkernel.copy_tile(%{{.+}}, %{{.+}}, %[[DST_IDX0]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB1]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.power_binary_tile_init
    // CHECK: ttkernel.power_binary_tile(%{{.+}}, %{{.+}}, %{{.+}})
    %2 = "ttir.tile_pow"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %2, %arg2[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_exp_lowering
  func.func @test_exp_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_exp
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.exp_tile_init
    // CHECK: ttkernel.exp_tile
    %1 = "ttir.tile_exp"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_log_lowering
  func.func @test_log_lowering(%arg0 : memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_log
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.log_tile_init
    // CHECK: ttkernel.log_tile
    %1 = "ttir.tile_log"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_cos_lowering
  func.func @test_cos_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_cos
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.cos_tile_init
    // CHECK: ttkernel.cos_tile
    %1 = "ttir.tile_cos"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_tan_lowering
  func.func @test_tan_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_tan
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.tan_tile_init
    // CHECK: ttkernel.tan_tile
    %1 = "ttir.tile_tan"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_negative_lowering
  func.func @test_negative_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_neg
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.negative_tile_init
    // CHECK: ttkernel.negative_tile
    %1 = "ttir.tile_negative"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_sqrt_lowering
  func.func @test_sqrt_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_sqrt
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.sqrt_tile_init
    // CHECK: ttkernel.sqrt_tile
    %1 = "ttir.tile_sqrt"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_rsqrt_lowering
  func.func @test_rsqrt_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_rsqrt
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.rsqrt_tile_init
    // CHECK: ttkernel.rsqrt_tile
    %1 = "ttir.tile_rsqrt"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_sin_lowering
  func.func @test_sin_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_sin
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.sin_tile_init
    // CHECK: ttkernel.sin_tile
    %1 = "ttir.tile_sin"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_sigmoid_lowering
  func.func @test_sigmoid_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_sigmoid
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.sigmoid_tile_init
    // CHECK: ttkernel.sigmoid_tile
    %1 = "ttir.tile_sigmoid"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_gelu_lowering
  func.func @test_gelu_lowering(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_gelu
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.gelu_tile_init
    // CHECK: ttkernel.gelu_tile
    %1 = "ttir.tile_gelu"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_ceil_lowering
  func.func @test_ceil_lowering(%arg0: memref<1x!ttcore.tile<32x32, bf16>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, bf16>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1_>
    // CHECK-NOT: ttir.tile_ceil
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.rounding_op_tile_init
    // CHECK: ttkernel.ceil_tile
    %1 = "ttir.tile_ceil"(%0) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_ceil_lowering_f32
  func.func @test_ceil_lowering_f32(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_ceil
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.rounding_op_tile_init
    // CHECK: ttkernel.ceil_tile_float32
    %1 = "ttir.tile_ceil"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_floor_lowering
  func.func @test_floor_lowering(%arg0: memref<1x!ttcore.tile<32x32, bf16>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, bf16>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1_>
    // CHECK-NOT: ttir.tile_floor
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.rounding_op_tile_init
    // CHECK: ttkernel.floor_tile
    %1 = "ttir.tile_floor"(%0) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_floor_lowering_f32
  func.func @test_floor_lowering_f32(%arg0: memref<1x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, f32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    // CHECK-NOT: ttir.tile_floor
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.rounding_op_tile_init
    // CHECK: ttkernel.floor_tile_float32
    %1 = "ttir.tile_floor"(%0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_abs_lowering
  func.func @test_abs_lowering(%arg0: memref<1x!ttcore.tile<32x32, bf16>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, bf16>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1_>
    // CHECK-NOT: ttir.tile_abs
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.abs_tile_init
    // CHECK: ttkernel.abs_tile
    %1 = "ttir.tile_abs"(%0) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_abs_i32_lowering
  func.func @test_abs_i32_lowering(%arg0: memref<1x!ttcore.tile<32x32, si32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, si32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, si32>, #l1_>
    // CHECK-NOT: ttir.tile_abs
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.abs_tile_init
    // CHECK: ttkernel.abs_tile_int32
    %1 = "ttir.tile_abs"(%0) : (!ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, si32>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_logical_not_lowering
  func.func @test_logical_not_lowering(%arg0: memref<1x!ttcore.tile<32x32, bf16>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, bf16>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1_>
    // CHECK-NOT: ttir.tile_logical_not
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.logical_not_unary_tile_init
    // CHECK: ttkernel.logical_not_unary_tile
    %1 = "ttir.tile_logical_not"(%0) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, bf16>, #l1_>
    return
  }

  // CHECK-LABEL: func.func @test_logical_not_i32_lowering
  func.func @test_logical_not_i32_lowering(%arg0: memref<1x!ttcore.tile<32x32, si32>, #l1_>, %arg1: memref<1x!ttcore.tile<32x32, si32>, #l1_>) attributes {ttir.thread = #ttir.thread<compute>} {
    %c0 = arith.constant 0 : index
    %0 = affine.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, si32>, #l1_>
    // CHECK-NOT: ttir.tile_logical_not
    // CHECK: ttkernel.init_sfpu
    // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
    // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
    // CHECK: ttkernel.logical_not_unary_tile_init
    // CHECK: ttkernel.logical_not_unary_tile_int32
    %1 = "ttir.tile_logical_not"(%0) : (!ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
    // CHECK: ttkernel.pack_tile
    affine.store %1, %arg1[%c0] : memref<1x!ttcore.tile<32x32, si32>, #l1_>
    return
  }
}
