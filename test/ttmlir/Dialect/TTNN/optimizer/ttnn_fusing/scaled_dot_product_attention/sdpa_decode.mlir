// SDPA Decode tests with two softmax variants:
//
// 1. Simple softmax: max -> subtract -> exp -> sum -> div
//    - Standard numerically stable softmax
//    - Produces NaN if entire row is masked (-inf)
//
// 2. NaN-safe softmax: adds "where" op to handle degenerate case
//    - Detects rows where all positions are -inf (fully masked)
//    - Outputs zeros instead of NaN for those rows
//    - Pattern: eq(-inf) -> logical_not -> reduce_or -> where(condition, 0, softmax)
//

// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true" %s | FileCheck %s
module {

  // CHECK: "ttnn.scaled_dot_product_attention_decode"
  func.func @sdpa_decode_simple_softmax_mha(%arg0: tensor<1x32x1x64xbf16>, %arg1: tensor<1x32x128x64xbf16>, %arg2: tensor<1x32x128x64xbf16>, %arg3: tensor<1x1x1x128xbf16>) -> tensor<1x32x1x64xbf16> {
    %0 = "ttir.transpose"(%arg1) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%arg0, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x1x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x1x128xbf16>
    %2 = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %3 = "ttir.multiply"(%1, %2) : (tensor<1x32x1x128xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x32x1x128xbf16>
    %4 = "ttir.add"(%3, %arg3) : (tensor<1x32x1x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x32x1x128xbf16>
    %5 = "ttir.typecast"(%4) <{conservative_folding = false}> : (tensor<1x32x1x128xbf16>) -> tensor<1x32x1x128xf32>
    %6 = "ttir.max"(%5) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<1x32x1x128xf32>) -> tensor<1x32x1xf32>
    %7 = "ttir.reshape"(%6) <{shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %8 = "ttir.broadcast"(%7) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x128xf32>
    %9 = "ttir.subtract"(%5, %8) : (tensor<1x32x1x128xf32>, tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xf32>
    %10 = "ttir.exp"(%9) : (tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xf32>
    %11 = "ttir.sum"(%10) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<1x32x1x128xf32>) -> tensor<1x32x1xf32>
    %12 = "ttir.reshape"(%11) <{shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %13 = "ttir.broadcast"(%12) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x128xf32>
    %14 = "ttir.div"(%10, %13) : (tensor<1x32x1x128xf32>, tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xf32>
    %15 = "ttir.typecast"(%14) <{conservative_folding = false}> : (tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xbf16>
    %16 = "ttir.matmul"(%15, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x1x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x1x64xbf16>
    return %16 : tensor<1x32x1x64xbf16>
  }

  // CHECK: "ttnn.scaled_dot_product_attention_decode"
  func.func @sdpa_decode_simple_softmax_mqa(%arg0: tensor<8x32x1x64xbf16>, %arg1: tensor<8x1x128x64xbf16>, %arg2: tensor<8x1x128x64xbf16>, %arg3: tensor<8x1x1x128xbf16>) -> tensor<8x32x1x64xbf16> {
    %0 = "ttir.reshape"(%arg1) <{shape = [8 : i32, 1 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<8x1x128x64xbf16>) -> tensor<8x1x1x128x64xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 1, 32, 1, 1>}> : (tensor<8x1x1x128x64xbf16>) -> tensor<8x1x32x128x64xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [8 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<8x1x32x128x64xbf16>) -> tensor<8x32x128x64xbf16>
    %3 = "ttir.reshape"(%arg2) <{shape = [8 : i32, 1 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<8x1x128x64xbf16>) -> tensor<8x1x1x128x64xbf16>
    %4 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 1, 32, 1, 1>}> : (tensor<8x1x1x128x64xbf16>) -> tensor<8x1x32x128x64xbf16>
    %5 = "ttir.reshape"(%4) <{shape = [8 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<8x1x32x128x64xbf16>) -> tensor<8x32x128x64xbf16>
    %6 = "ttir.transpose"(%2) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x32x128x64xbf16>) -> tensor<8x32x64x128xbf16>
    %7 = "ttir.matmul"(%arg0, %6) <{transpose_a = false, transpose_b = false}> : (tensor<8x32x1x64xbf16>, tensor<8x32x64x128xbf16>) -> tensor<8x32x1x128xbf16>
    %8 = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %9 = "ttir.multiply"(%7, %8) : (tensor<8x32x1x128xbf16>, tensor<1x1x1x1xbf16>) -> tensor<8x32x1x128xbf16>
    %10 = "ttir.add"(%9, %arg3) : (tensor<8x32x1x128xbf16>, tensor<8x1x1x128xbf16>) -> tensor<8x32x1x128xbf16>
    %11 = "ttir.typecast"(%10) <{conservative_folding = false}> : (tensor<8x32x1x128xbf16>) -> tensor<8x32x1x128xf32>
    %12 = "ttir.max"(%11) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<8x32x1x128xf32>) -> tensor<8x32x1xf32>
    %13 = "ttir.reshape"(%12) <{shape = [8 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<8x32x1xf32>) -> tensor<8x32x1x1xf32>
    %14 = "ttir.broadcast"(%13) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<8x32x1x1xf32>) -> tensor<8x32x1x128xf32>
    %15 = "ttir.subtract"(%11, %14) : (tensor<8x32x1x128xf32>, tensor<8x32x1x128xf32>) -> tensor<8x32x1x128xf32>
    %16 = "ttir.exp"(%15) : (tensor<8x32x1x128xf32>) -> tensor<8x32x1x128xf32>
    %17 = "ttir.sum"(%16) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<8x32x1x128xf32>) -> tensor<8x32x1xf32>
    %18 = "ttir.reshape"(%17) <{shape = [8 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<8x32x1xf32>) -> tensor<8x32x1x1xf32>
    %19 = "ttir.broadcast"(%18) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<8x32x1x1xf32>) -> tensor<8x32x1x128xf32>
    %20 = "ttir.div"(%16, %19) : (tensor<8x32x1x128xf32>, tensor<8x32x1x128xf32>) -> tensor<8x32x1x128xf32>
    %21 = "ttir.typecast"(%20) <{conservative_folding = false}> : (tensor<8x32x1x128xf32>) -> tensor<8x32x1x128xbf16>
    %22 = "ttir.matmul"(%21, %5) <{transpose_a = false, transpose_b = false}> : (tensor<8x32x1x128xbf16>, tensor<8x32x128x64xbf16>) -> tensor<8x32x1x64xbf16>
    return %22 : tensor<8x32x1x64xbf16>
  }

  // CHECK: "ttnn.scaled_dot_product_attention_decode"
  func.func @sdpa_decode_simple_softmax_gqa(%arg0: tensor<32x32x1x64xbf16>, %arg1: tensor<32x8x128x64xbf16>, %arg2: tensor<32x8x128x64xbf16>, %arg3: tensor<32x1x1x128xbf16>) -> tensor<32x32x1x64xbf16> {
    %0 = "ttir.reshape"(%arg1) <{shape = [32 : i32, 8 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x128x64xbf16>) -> tensor<32x8x1x128x64xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x8x1x128x64xbf16>) -> tensor<32x8x4x128x64xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x4x128x64xbf16>) -> tensor<32x32x128x64xbf16>
    %3 = "ttir.reshape"(%arg2) <{shape = [32 : i32, 8 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x128x64xbf16>) -> tensor<32x8x1x128x64xbf16>
    %4 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x8x1x128x64xbf16>) -> tensor<32x8x4x128x64xbf16>
    %5 = "ttir.reshape"(%4) <{shape = [32 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x4x128x64xbf16>) -> tensor<32x32x128x64xbf16>
    %6 = "ttir.transpose"(%2) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x32x128x64xbf16>) -> tensor<32x32x64x128xbf16>
    %7 = "ttir.matmul"(%arg0, %6) <{transpose_a = false, transpose_b = false}> : (tensor<32x32x1x64xbf16>, tensor<32x32x64x128xbf16>) -> tensor<32x32x1x128xbf16>
    %8 = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %9 = "ttir.multiply"(%7, %8) : (tensor<32x32x1x128xbf16>, tensor<1x1x1x1xbf16>) -> tensor<32x32x1x128xbf16>
    %10 = "ttir.add"(%9, %arg3) : (tensor<32x32x1x128xbf16>, tensor<32x1x1x128xbf16>) -> tensor<32x32x1x128xbf16>
    %11 = "ttir.typecast"(%10) <{conservative_folding = false}> : (tensor<32x32x1x128xbf16>) -> tensor<32x32x1x128xf32>
    %12 = "ttir.max"(%11) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<32x32x1x128xf32>) -> tensor<32x32x1xf32>
    %13 = "ttir.reshape"(%12) <{shape = [32 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1xf32>) -> tensor<32x32x1x1xf32>
    %14 = "ttir.broadcast"(%13) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x32x1x1xf32>) -> tensor<32x32x1x128xf32>
    %15 = "ttir.subtract"(%11, %14) : (tensor<32x32x1x128xf32>, tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %16 = "ttir.exp"(%15) : (tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %17 = "ttir.sum"(%16) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<32x32x1x128xf32>) -> tensor<32x32x1xf32>
    %18 = "ttir.reshape"(%17) <{shape = [32 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1xf32>) -> tensor<32x32x1x1xf32>
    %19 = "ttir.broadcast"(%18) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x32x1x1xf32>) -> tensor<32x32x1x128xf32>
    %20 = "ttir.div"(%16, %19) : (tensor<32x32x1x128xf32>, tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %21 = "ttir.typecast"(%20) <{conservative_folding = false}> : (tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xbf16>
    %22 = "ttir.matmul"(%21, %5) <{transpose_a = false, transpose_b = false}> : (tensor<32x32x1x128xbf16>, tensor<32x32x128x64xbf16>) -> tensor<32x32x1x64xbf16>
    return %22 : tensor<32x32x1x64xbf16>
  }

  // CHECK: "ttnn.scaled_dot_product_attention_decode"
  func.func @sdpa_decode_NAN_safe_softmax_mha(%arg0: tensor<1x32x1x64xbf16>, %arg1: tensor<1x32x128x64xbf16>, %arg2: tensor<1x32x128x64xbf16>, %arg3: tensor<1x1x1x128xbf16>) -> tensor<1x32x1x64xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x1x64xbf16>) -> tensor<1x32x1x64xf32>
    %1 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    %2 = "ttir.multiply"(%0, %1) : (tensor<1x32x1x64xf32>, tensor<1x1x1x1xf32>) -> tensor<1x32x1x64xf32>
    %3 = "ttir.typecast"(%arg1) <{conservative_folding = false}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xf32>
    %4 = "ttir.transpose"(%3) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xf32>) -> tensor<1x32x64x128xf32>
    %5 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    %6 = "ttir.multiply"(%4, %5) : (tensor<1x32x64x128xf32>, tensor<1x1x1x1xf32>) -> tensor<1x32x64x128xf32>
    %7 = "ttir.matmul"(%2, %6) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x1x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x1x128xf32>
    %8 = "ttir.typecast"(%arg3) <{conservative_folding = false}> : (tensor<1x1x1x128xbf16>) -> tensor<1x1x1x128xf32>
    %9 = "ttir.add"(%7, %8) : (tensor<1x32x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x32x1x128xf32>
    %10 = "ttir.constant"() <{value = dense<0xFF800000> : tensor<1x32x1x128xf32>}> : () -> tensor<1x32x1x128xf32>
    %11 = "ttir.eq"(%9, %10) : (tensor<1x32x1x128xf32>, tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xf32>
    %12 = "ttir.logical_not"(%11) : (tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xf32>
    %13 = "ttir.reduce_or"(%12) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<1x32x1x128xf32>) -> tensor<1x32x1xf32>
    %14 = "ttir.reshape"(%13) <{shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %15 = "ttir.logical_not"(%14) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %16 = "ttir.broadcast"(%15) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x128xf32>
    %17 = "ttir.max"(%9) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<1x32x1x128xf32>) -> tensor<1x32x1xf32>
    %18 = "ttir.reshape"(%17) <{shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %19 = "ttir.broadcast"(%18) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x128xf32>
    %20 = "ttir.subtract"(%9, %19) : (tensor<1x32x1x128xf32>, tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xf32>
    %21 = "ttir.exp"(%20) : (tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xf32>
    %22 = "ttir.sum"(%21) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<1x32x1x128xf32>) -> tensor<1x32x1xf32>
    %23 = "ttir.reshape"(%22) <{shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1xf32>) -> tensor<1x32x1x1xf32>
    %24 = "ttir.broadcast"(%23) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x128xf32>
    %25 = "ttir.div"(%21, %24) : (tensor<1x32x1x128xf32>, tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xf32>
    %26 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x32x1x128xf32>}> : () -> tensor<1x32x1x128xf32>
    %27 = "ttir.where"(%16, %26, %25) : (tensor<1x32x1x128xf32>, tensor<1x32x1x128xf32>, tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xf32>
    %28 = "ttir.typecast"(%arg2) <{conservative_folding = false}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xf32>
    %29 = "ttir.matmul"(%27, %28) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x1x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x1x64xf32>
    %30 = "ttir.typecast"(%29) <{conservative_folding = false}> : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x64xbf16>
    return %30 : tensor<1x32x1x64xbf16>
  }

  // CHECK: "ttnn.scaled_dot_product_attention_decode"
  func.func @sdpa_decode_NAN_safe_softmax_mqa(%arg0: tensor<8x32x1x64xbf16>, %arg1: tensor<8x1x128x64xbf16>, %arg2: tensor<8x1x128x64xbf16>, %arg3: tensor<8x1x1x128xbf16>) -> tensor<8x32x1x64xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<8x32x1x64xbf16>) -> tensor<8x32x1x64xf32>
    %1 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    %2 = "ttir.multiply"(%0, %1) : (tensor<8x32x1x64xf32>, tensor<1x1x1x1xf32>) -> tensor<8x32x1x64xf32>
    %3 = "ttir.reshape"(%arg1) <{shape = [8 : i32, 1 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<8x1x128x64xbf16>) -> tensor<8x1x1x128x64xbf16>
    %4 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 1, 32, 1, 1>}> : (tensor<8x1x1x128x64xbf16>) -> tensor<8x1x32x128x64xbf16>
    %5 = "ttir.reshape"(%4) <{shape = [8 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<8x1x32x128x64xbf16>) -> tensor<8x32x128x64xbf16>
    %6 = "ttir.reshape"(%arg2) <{shape = [8 : i32, 1 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<8x1x128x64xbf16>) -> tensor<8x1x1x128x64xbf16>
    %7 = "ttir.broadcast"(%6) <{broadcast_dimensions = array<i64: 1, 1, 32, 1, 1>}> : (tensor<8x1x1x128x64xbf16>) -> tensor<8x1x32x128x64xbf16>
    %8 = "ttir.reshape"(%7) <{shape = [8 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<8x1x32x128x64xbf16>) -> tensor<8x32x128x64xbf16>
    %9 = "ttir.typecast"(%5) <{conservative_folding = false}> : (tensor<8x32x128x64xbf16>) -> tensor<8x32x128x64xf32>
    %10 = "ttir.transpose"(%9) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<8x32x128x64xf32>) -> tensor<8x32x64x128xf32>
    %11 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    %12 = "ttir.multiply"(%10, %11) : (tensor<8x32x64x128xf32>, tensor<1x1x1x1xf32>) -> tensor<8x32x64x128xf32>
    %13 = "ttir.matmul"(%2, %12) <{transpose_a = false, transpose_b = false}> : (tensor<8x32x1x64xf32>, tensor<8x32x64x128xf32>) -> tensor<8x32x1x128xf32>
    %14 = "ttir.typecast"(%arg3) <{conservative_folding = false}> : (tensor<8x1x1x128xbf16>) -> tensor<8x1x1x128xf32>
    %15 = "ttir.add"(%13, %14) : (tensor<8x32x1x128xf32>, tensor<8x1x1x128xf32>) -> tensor<8x32x1x128xf32>
    %16 = "ttir.constant"() <{value = dense<0xFF800000> : tensor<8x32x1x128xf32>}> : () -> tensor<8x32x1x128xf32>
    %17 = "ttir.eq"(%15, %16) : (tensor<8x32x1x128xf32>, tensor<8x32x1x128xf32>) -> tensor<8x32x1x128xf32>
    %18 = "ttir.logical_not"(%17) : (tensor<8x32x1x128xf32>) -> tensor<8x32x1x128xf32>
    %19 = "ttir.reduce_or"(%18) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<8x32x1x128xf32>) -> tensor<8x32x1xf32>
    %20 = "ttir.reshape"(%19) <{shape = [8 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<8x32x1xf32>) -> tensor<8x32x1x1xf32>
    %21 = "ttir.logical_not"(%20) : (tensor<8x32x1x1xf32>) -> tensor<8x32x1x1xf32>
    %22 = "ttir.broadcast"(%21) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<8x32x1x1xf32>) -> tensor<8x32x1x128xf32>
    %23 = "ttir.max"(%15) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<8x32x1x128xf32>) -> tensor<8x32x1xf32>
    %24 = "ttir.reshape"(%23) <{shape = [8 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<8x32x1xf32>) -> tensor<8x32x1x1xf32>
    %25 = "ttir.broadcast"(%24) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<8x32x1x1xf32>) -> tensor<8x32x1x128xf32>
    %26 = "ttir.subtract"(%15, %25) : (tensor<8x32x1x128xf32>, tensor<8x32x1x128xf32>) -> tensor<8x32x1x128xf32>
    %27 = "ttir.exp"(%26) : (tensor<8x32x1x128xf32>) -> tensor<8x32x1x128xf32>
    %28 = "ttir.sum"(%27) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<8x32x1x128xf32>) -> tensor<8x32x1xf32>
    %29 = "ttir.reshape"(%28) <{shape = [8 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<8x32x1xf32>) -> tensor<8x32x1x1xf32>
    %30 = "ttir.broadcast"(%29) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<8x32x1x1xf32>) -> tensor<8x32x1x128xf32>
    %31 = "ttir.div"(%27, %30) : (tensor<8x32x1x128xf32>, tensor<8x32x1x128xf32>) -> tensor<8x32x1x128xf32>
    %32 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<8x32x1x128xf32>}> : () -> tensor<8x32x1x128xf32>
    %33 = "ttir.where"(%22, %32, %31) : (tensor<8x32x1x128xf32>, tensor<8x32x1x128xf32>, tensor<8x32x1x128xf32>) -> tensor<8x32x1x128xf32>
    %34 = "ttir.typecast"(%8) <{conservative_folding = false}> : (tensor<8x32x128x64xbf16>) -> tensor<8x32x128x64xf32>
    %35 = "ttir.matmul"(%33, %34) <{transpose_a = false, transpose_b = false}> : (tensor<8x32x1x128xf32>, tensor<8x32x128x64xf32>) -> tensor<8x32x1x64xf32>
    %36 = "ttir.typecast"(%35) <{conservative_folding = false}> : (tensor<8x32x1x64xf32>) -> tensor<8x32x1x64xbf16>
    return %36 : tensor<8x32x1x64xbf16>
  }

  // CHECK: "ttnn.scaled_dot_product_attention_decode"
  func.func @sdpa_decode_NAN_safe_softmax_gqa(%arg0: tensor<32x32x1x64xbf16>, %arg1: tensor<32x8x128x64xbf16>, %arg2: tensor<32x8x128x64xbf16>, %arg3: tensor<32x1x1x128xbf16>) -> tensor<32x32x1x64xbf16> {
    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x32x1x64xbf16>) -> tensor<32x32x1x64xf32>
    %1 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    %2 = "ttir.multiply"(%0, %1) : (tensor<32x32x1x64xf32>, tensor<1x1x1x1xf32>) -> tensor<32x32x1x64xf32>
    %3 = "ttir.reshape"(%arg1) <{shape = [32 : i32, 8 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x128x64xbf16>) -> tensor<32x8x1x128x64xbf16>
    %4 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x8x1x128x64xbf16>) -> tensor<32x8x4x128x64xbf16>
    %5 = "ttir.reshape"(%4) <{shape = [32 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x4x128x64xbf16>) -> tensor<32x32x128x64xbf16>
    %6 = "ttir.reshape"(%arg2) <{shape = [32 : i32, 8 : i32, 1 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x128x64xbf16>) -> tensor<32x8x1x128x64xbf16>
    %7 = "ttir.broadcast"(%6) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x8x1x128x64xbf16>) -> tensor<32x8x4x128x64xbf16>
    %8 = "ttir.reshape"(%7) <{shape = [32 : i32, 32 : i32, 128 : i32, 64 : i32]}> : (tensor<32x8x4x128x64xbf16>) -> tensor<32x32x128x64xbf16>
    %9 = "ttir.typecast"(%5) <{conservative_folding = false}> : (tensor<32x32x128x64xbf16>) -> tensor<32x32x128x64xf32>
    %10 = "ttir.transpose"(%9) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x32x128x64xf32>) -> tensor<32x32x64x128xf32>
    %11 = "ttir.constant"() <{value = dense<0.353553385> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
    %12 = "ttir.multiply"(%10, %11) : (tensor<32x32x64x128xf32>, tensor<1x1x1x1xf32>) -> tensor<32x32x64x128xf32>
    %13 = "ttir.matmul"(%2, %12) <{transpose_a = false, transpose_b = false}> : (tensor<32x32x1x64xf32>, tensor<32x32x64x128xf32>) -> tensor<32x32x1x128xf32>
    %14 = "ttir.typecast"(%arg3) <{conservative_folding = false}> : (tensor<32x1x1x128xbf16>) -> tensor<32x1x1x128xf32>
    %15 = "ttir.add"(%13, %14) : (tensor<32x32x1x128xf32>, tensor<32x1x1x128xf32>) -> tensor<32x32x1x128xf32>
    %16 = "ttir.constant"() <{value = dense<0xFF800000> : tensor<32x32x1x128xf32>}> : () -> tensor<32x32x1x128xf32>
    %17 = "ttir.eq"(%15, %16) : (tensor<32x32x1x128xf32>, tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %18 = "ttir.logical_not"(%17) : (tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %19 = "ttir.reduce_or"(%18) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<32x32x1x128xf32>) -> tensor<32x32x1xf32>
    %20 = "ttir.reshape"(%19) <{shape = [32 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1xf32>) -> tensor<32x32x1x1xf32>
    %21 = "ttir.logical_not"(%20) : (tensor<32x32x1x1xf32>) -> tensor<32x32x1x1xf32>
    %22 = "ttir.broadcast"(%21) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x32x1x1xf32>) -> tensor<32x32x1x128xf32>
    %23 = "ttir.max"(%15) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<32x32x1x128xf32>) -> tensor<32x32x1xf32>
    %24 = "ttir.reshape"(%23) <{shape = [32 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1xf32>) -> tensor<32x32x1x1xf32>
    %25 = "ttir.broadcast"(%24) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x32x1x1xf32>) -> tensor<32x32x1x128xf32>
    %26 = "ttir.subtract"(%15, %25) : (tensor<32x32x1x128xf32>, tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %27 = "ttir.exp"(%26) : (tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %28 = "ttir.sum"(%27) <{dim_arg = [-1 : i32], keep_dim = false}> : (tensor<32x32x1x128xf32>) -> tensor<32x32x1xf32>
    %29 = "ttir.reshape"(%28) <{shape = [32 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x1xf32>) -> tensor<32x32x1x1xf32>
    %30 = "ttir.broadcast"(%29) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x32x1x1xf32>) -> tensor<32x32x1x128xf32>
    %31 = "ttir.div"(%27, %30) : (tensor<32x32x1x128xf32>, tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %32 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<32x32x1x128xf32>}> : () -> tensor<32x32x1x128xf32>
    %33 = "ttir.where"(%22, %32, %31) : (tensor<32x32x1x128xf32>, tensor<32x32x1x128xf32>, tensor<32x32x1x128xf32>) -> tensor<32x32x1x128xf32>
    %34 = "ttir.typecast"(%8) <{conservative_folding = false}> : (tensor<32x32x128x64xbf16>) -> tensor<32x32x128x64xf32>
    %35 = "ttir.matmul"(%33, %34) <{transpose_a = false, transpose_b = false}> : (tensor<32x32x1x128xf32>, tensor<32x32x128x64xf32>) -> tensor<32x32x1x64xf32>
    %36 = "ttir.typecast"(%35) <{conservative_folding = false}> : (tensor<32x32x1x64xf32>) -> tensor<32x32x1x64xbf16>
    return %36 : tensor<32x32x1x64xbf16>
  }

}
