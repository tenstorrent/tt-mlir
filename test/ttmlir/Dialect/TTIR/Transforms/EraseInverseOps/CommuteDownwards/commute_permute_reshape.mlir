// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-upwards=false" -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
  // Positive examples: reshape has permute as user

  // [N, H, W, C]  -> [N, C, H, W]
  // [N, C, H, W]  -> [N, 1, C, HW]
  // [N, 1, C, HW] -> [N, 1, HW, C]
  // func.func @test_commute_permute_n_1_c_hw(%arg0: tensor<16x7x7x2048xbf16>) -> tensor<16x1x49x2048xbf16> {
  //   // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [16 : i32, 2048 : i32, 49 : i32, 1 : i32]}>
  //   // CHECK: %[[PERMUTE1:[0-9]+]] = "ttir.permute"(%[[RESHAPE]], %{{[0-9]+}}) <{permutation = array<i64: 0, 3, 1, 2>}>
  //   // CHECK: %[[PERMUTE2:[0-9]+]] = "ttir.permute"(%[[PERMUTE1]], %{{[0-9]+}}) <{permutation = array<i64: 0, 1, 3, 2>}>
  //   // CHECK: return %[[PERMUTE2]]
  //   %0 = tensor.empty() : tensor<16x2048x7x7xbf16>
  //   %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<16x7x7x2048xbf16>, tensor<16x2048x7x7xbf16>) -> tensor<16x2048x7x7xbf16>
  //   %2 = tensor.empty() : tensor<16x1x2048x49xbf16>
  //   %3 = "ttir.reshape"(%1, %2) <{shape = [16:i32, 1: i32, 2048: i32, 49: i32]}> : (tensor<16x2048x7x7xbf16>, tensor<16x1x2048x49xbf16>) -> tensor<16x1x2048x49xbf16>
  //   %4 = tensor.empty() : tensor<16x1x49x2048xbf16>
  //   %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<16x1x2048x49xbf16>, tensor<16x1x49x2048xbf16>) -> tensor<16x1x49x2048xbf16>
  //   return %5: tensor<16x1x49x2048xbf16>
  // }

  // func.func @test_commute_permute_with_ones_in_reshape(%arg0: tensor<16x7x7x2048xbf16>) -> tensor<16x1x49x2048x1xbf16> {
  //   // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [16 : i32, 2048 : i32, 49 : i32, 1 : i32]}>
  //   // CHECK: %[[PERMUTE1:[0-9]+]] = "ttir.permute"(%[[RESHAPE]], %{{[0-9]+}}) <{permutation = array<i64: 0, 3, 1, 2>}>
  //   // CHECK: %[[PERMUTE2:[0-9]+]] = "ttir.permute"(%[[PERMUTE1]], %{{[0-9]+}}) <{permutation = array<i64: 0, 1, 3, 2>}>
  //   // CHECK: return %[[PERMUTE2]]
  //   %0 = tensor.empty() : tensor<16x2048x7x7xbf16>
  //   %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<16x7x7x2048xbf16>, tensor<16x2048x7x7xbf16>) -> tensor<16x2048x7x7xbf16>
  //   %2 = tensor.empty() : tensor<16x1x2048x49x1xbf16>
  //   %3 = "ttir.reshape"(%1, %2) <{shape = [16:i32, 1: i32, 2048: i32, 49: i32, 1: i32]}> : (tensor<16x2048x7x7xbf16>, tensor<16x1x2048x49x1xbf16>) -> tensor<16x1x2048x49x1xbf16>
  //   %4 = tensor.empty() : tensor<16x1x49x2048x1xbf16>
  //   %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 1, 3, 2, 4>}> : (tensor<16x1x2048x49x1xbf16>, tensor<16x1x49x2048x1xbf16>) -> tensor<16x1x49x2048x1xbf16>
  //   return %5: tensor<16x1x49x2048x1xbf16>
  // }


  // // [N, H, W, C]  -> [N, C, H, W]
  // // [N, C, H, W]  -> [N, C, 1, HW]
  // // [N, C, 1, HW] -> [N, 1, HW, C]
  // func.func @test_commute_permute_reshape_n_c_1_hw(%arg0: tensor<12x7x7x1152xbf16>) -> tensor<12x1x49x1152xbf16> {
  //   // CHECK-LABEL: func.func @test_commute_permute_reshape_n_c_1_hw
  //   // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [12 : i32, 1 : i32, 49 : i32, 1152 : i32]}>
  //   // CHECK: %[[PERMUTE1:[0-9]+]] = "ttir.permute"(%[[RESHAPE]], %{{[0-9]+}}) <{permutation = array<i64: 0, 3, 1, 2>}>
  //   // CHECK: %[[PERMUTE2:[0-9]+]] = "ttir.permute"(%[[PERMUTE1]], %{{[0-9]+}}) <{permutation = array<i64: 0, 2, 3, 1>}>
  //   // CHECK: return %[[PERMUTE2]]
  //   %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
  //   %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
  //   %2 = ttir.empty() : tensor<12x1152x1x49xbf16>
  //   %3 = "ttir.reshape"(%1, %2) <{shape = [12 : i32, 1152 : i32, 1 : i32, 49 : i32]}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x49xbf16>) -> tensor<12x1152x1x49xbf16>
  //   %4 = ttir.empty() : tensor<12x1x49x1152xbf16>
  //   %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<12x1152x1x49xbf16>, tensor<12x1x49x1152xbf16>) -> tensor<12x1x49x1152xbf16>
  //   return %5 : tensor<12x1x49x1152xbf16>
  // }

  // [8, 1, 2048, 1] -> [8, 2048, 1, 1] (permute [0, 2, 3, 1])
  // [8, 2048, 1, 1] -> [8, 2048] (reshape)
  func.func @test_commute_permute_reshape_8_2048(%arg0: tensor<8x1x2048x1xbf16>) -> tensor<8x2048xbf16> {
    // CHECK-LABEL: func.func @test_commute_permute_reshape_8_2048
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [8 : i32, 2048 : i32]}>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[RESHAPE]], %{{[0-9]+}}) <{permutation = array<i64: 0, 1>}>
    // CHECK: return %[[PERMUTE]]
    %0 = ttir.empty() : tensor<8x2048x1x1xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<8x1x2048x1xbf16>, tensor<8x2048x1x1xbf16>) -> tensor<8x2048x1x1xbf16>
    %2 = ttir.empty() : tensor<8x2048xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [8 : i32, 2048 : i32]}> : (tensor<8x2048x1x1xbf16>, tensor<8x2048xbf16>) -> tensor<8x2048xbf16>
    return %3 : tensor<8x2048xbf16>
  }
  

  // // [A, B, C, D]   -> [B, C, D, A]
  // // [B, C, D, A]   -> [BCD, A, 1, 1]
  // // [BCD, A, 1, 1] -> [1, 1, A, BCD]
  // func.func @test_commute_permute_reshape_1_1_a_bcd(%arg0: tensor<2x2x2x2xbf16>) -> tensor<1x1x2x8xbf16> {
  //   // CHECK-LABEL: func.func @test_commute_permute_reshape_1_1_a_bcd
  //   // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [1 : i32, 8 : i32, 2 : i32, 1 : i32]}>
  //   // CHECK: %[[PERMUTE1:[0-9]+]] = "ttir.permute"(%[[RESHAPE]], %{{[0-9]+}}) <{permutation = array<i64: 1, 2, 3, 0>}>
  //   // CHECK: %[[PERMUTE2:[0-9]+]] = "ttir.permute"(%[[PERMUTE1]], %{{[0-9]+}}) <{permutation = array<i64: 3, 2, 1, 0>}>
  //   // CHECK: return %[[PERMUTE2]]
  //   %0 = ttir.empty() : tensor<2x2x2x2xbf16>
  //   %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 1, 2, 3, 0>}> : (tensor<2x2x2x2xbf16>, tensor<2x2x2x2xbf16>) -> tensor<2x2x2x2xbf16>
  //   %2 = ttir.empty() : tensor<8x2x1x1xbf16>
  //   %3 = "ttir.reshape"(%1, %2) <{shape = [8 : i32, 2 : i32, 1 : i32, 1 : i32]}> : (tensor<2x2x2x2xbf16>, tensor<8x2x1x1xbf16>) -> tensor<8x2x1x1xbf16>
  //   %4 = ttir.empty() : tensor<1x1x2x8xbf16>
  //   %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 3, 2, 1, 0>}> : (tensor<8x2x1x1xbf16>, tensor<1x1x2x8xbf16>) -> tensor<1x1x2x8xbf16>
  //   return %5 : tensor<1x1x2x8xbf16>
  // }

  // // [A, B, C, D, E]   -> [A, D, E, B, C]
  // // [A, D, E, B, C]   -> [A, DE, 1, 1, BC]
  // // [A, DE, 1, 1, BC] -> [A, 1, BC, 1, DE]
  // func.func @test_commute_permute_reshape_a_1_bc_1_de(%arg0: tensor<2x3x5x7x11xbf16>) -> tensor<2x1x15x1x77xbf16> {
  //   // CHECK-LABEL: func.func @test_commute_permute_reshape_a_1_bc_1_de
  //   // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [2 : i32, 1 : i32, 15 : i32, 77 : i32, 1 : i32]}>
  //   // CHECK: %[[PERMUTE1:[0-9]+]] = "ttir.permute"(%[[RESHAPE]], %{{[0-9]+}}) <{permutation = array<i64: 0, 3, 4, 1, 2>}>
  //   // CHECK: %[[PERMUTE2:[0-9]+]] = "ttir.permute"(%[[PERMUTE1]], %{{[0-9]+}}) <{permutation = array<i64: 0, 3, 4, 2, 1>}>
  //   // CHECK: return %[[PERMUTE2]]
  //   %0 = ttir.empty() : tensor<2x7x11x3x5xbf16>
  //   %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 4, 1, 2>}> : (tensor<2x3x5x7x11xbf16>, tensor<2x7x11x3x5xbf16>) -> tensor<2x7x11x3x5xbf16>
  //   %2 = ttir.empty() : tensor<2x77x1x1x15xbf16>
  //   %3 = "ttir.reshape"(%1, %2) <{shape = [2 : i32, 77 : i32, 1 : i32, 1 : i32, 15 : i32]}> : (tensor<2x7x11x3x5xbf16>, tensor<2x77x1x1x15xbf16>) -> tensor<2x77x1x1x15xbf16>
  //   %4 = ttir.empty() : tensor<2x1x15x1x77xbf16>
  //   %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 3, 4, 2, 1>}> : (tensor<2x77x1x1x15xbf16>, tensor<2x1x15x1x77xbf16>) -> tensor<2x1x15x1x77xbf16>
  //   return %5 : tensor<2x1x15x1x77xbf16>
  // }

  // // [A, B, C, D]   -> [C, D, A, B]
  // // [C, D, A, B]   -> [CD, 1, 1, AB]
  // // [CD, 1, 1, AB] -> [AB, 1, CD, 1]
  // func.func @test_commute_permute_reshape_ab_1_1_cd(%arg0: tensor<2x2x2x2xbf16>) -> tensor<4x1x1x4xbf16> {
  //   // CHECK-LABEL: func.func @test_commute_permute_reshape_ab_1_1_cd
  //   // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [1 : i32, 4 : i32, 1 : i32, 4 : i32]}>
  //   // CHECK: %[[PERMUTE1:[0-9]+]] = "ttir.permute"(%[[RESHAPE]], %{{[0-9]+}}) <{permutation = array<i64: 2, 3, 0, 1>}>
  //   // CHECK: %[[PERMUTE2:[0-9]+]] = "ttir.permute"(%[[PERMUTE1]], %{{[0-9]+}}) <{permutation = array<i64: 3, 0, 2, 1>}>
  //   // CHECK: return %[[PERMUTE2]]
  //   %0 = ttir.empty() : tensor<2x2x2x2xbf16>
  //   %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 2, 3, 0, 1>}> : (tensor<2x2x2x2xbf16>, tensor<2x2x2x2xbf16>) -> tensor<2x2x2x2xbf16>
  //   %2 = ttir.empty() : tensor<1x4x1x4xbf16>
  //   %3 = "ttir.reshape"(%1, %2) <{shape = [1 : i32, 4 : i32, 1 : i32, 4 : i32]}> : (tensor<2x2x2x2xbf16>, tensor<1x4x1x4xbf16>) -> tensor<1x4x1x4xbf16>
  //   %4 = ttir.empty() : tensor<4x1x1x4xbf16>
  //   %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 3, 0, 2, 1>}> : (tensor<1x4x1x4xbf16>, tensor<4x1x1x4xbf16>) -> tensor<4x1x1x4xbf16>
  //   return %5 : tensor<4x1x1x4xbf16>
  // }

  // // Negative examples:

  // // Reshape does not have permute as user - should not commute
  // func.func @test_no_commute_reshape_no_permute_user(%arg0: tensor<16x7x7x2048xbf16>) -> tensor<16x1x2048x49xbf16> {
  //   // CHECK-LABEL: func.func @test_no_commute_reshape_no_permute_user
  //   // CHECK: "ttir.permute"
  //   // CHECK: "ttir.reshape"
  //   // CHECK-NOT: "ttir.permute"
  //   %0 = tensor.empty() : tensor<16x2048x7x7xbf16>
  //   %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<16x7x7x2048xbf16>, tensor<16x2048x7x7xbf16>) -> tensor<16x2048x7x7xbf16>
  //   %2 = tensor.empty() : tensor<16x1x2048x49xbf16>
  //   %3 = "ttir.reshape"(%1, %2) <{shape = [16:i32, 1: i32, 2048: i32, 49: i32]}> : (tensor<16x2048x7x7xbf16>, tensor<16x1x2048x49xbf16>) -> tensor<16x1x2048x49xbf16>
  //   return %3: tensor<16x1x2048x49xbf16>
  // }

  // // Reshape has non-permute user - should not commute
  // func.func @test_no_commute_reshape_non_permute_user(%arg0: tensor<16x7x7x2048xbf16>) -> tensor<16x1x2048x49xbf16> {
  //   // CHECK-LABEL: func.func @test_no_commute_reshape_non_permute_user
  //   // CHECK: "ttir.permute"
  //   // CHECK: "ttir.reshape"
  //   // CHECK: "ttir.add"
  //   %0 = tensor.empty() : tensor<16x2048x7x7xbf16>
  //   %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<16x7x7x2048xbf16>, tensor<16x2048x7x7xbf16>) -> tensor<16x2048x7x7xbf16>
  //   %2 = tensor.empty() : tensor<16x1x2048x49xbf16>
  //   %3 = "ttir.reshape"(%1, %2) <{shape = [16:i32, 1: i32, 2048: i32, 49: i32]}> : (tensor<16x2048x7x7xbf16>, tensor<16x1x2048x49xbf16>) -> tensor<16x1x2048x49xbf16>
  //   %4 = tensor.empty() : tensor<16x1x2048x49xbf16>
  //   %5 = "ttir.add"(%3, %3, %4) : (tensor<16x1x2048x49xbf16>, tensor<16x1x2048x49xbf16>, tensor<16x1x2048x49xbf16>) -> tensor<16x1x2048x49xbf16>
  //   return %5: tensor<16x1x2048x49xbf16>
  // }

  // // [A, B, C, D]   -> [C, D, B, A]
  // // [C, D, B, A]   -> [CD, 1, 1, BA]
  // // [CD, 1, 1, BA] -> [BA, 1, CD, 1]
  // func.func @test_commute_permute_reshape_ba_1_1_cd(%arg0: tensor<2x2x2x2xbf16>) -> tensor<4x1x1x4xbf16> {
  //   // CHECK-LABEL: func.func @test_commute_permute_reshape_ba_1_1_cd
  //   // CHECK: ttir.permute
  //   // CHECK: ttir.reshape
  //   // CHECK: ttir.permute
  //   %0 = ttir.empty() : tensor<2x2x2x2xbf16>
  //   %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 2, 3, 1, 0>}> : (tensor<2x2x2x2xbf16>, tensor<2x2x2x2xbf16>) -> tensor<2x2x2x2xbf16>
  //   %2 = ttir.empty() : tensor<1x4x1x4xbf16>
  //   %3 = "ttir.reshape"(%1, %2) <{shape = [1 : i32, 4 : i32, 1 : i32, 4 : i32]}> : (tensor<2x2x2x2xbf16>, tensor<1x4x1x4xbf16>) -> tensor<1x4x1x4xbf16>
  //   %4 = ttir.empty() : tensor<4x1x1x4xbf16>
  //   %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 3, 0, 2, 1>}> : (tensor<1x4x1x4xbf16>, tensor<4x1x1x4xbf16>) -> tensor<4x1x1x4xbf16>
  //   return %5 : tensor<4x1x1x4xbf16>
  // }

  // // [A, B, C]  -> [B, C, A]
  // // [B, C, A]  -> [BC, A, 1]
  // // [BC, A, 1] -> [1, BC, A]
  // func.func @test_commute_permute_reshape_1_bc_a(%arg0: tensor<2x2x2xbf16>) -> tensor<1x4x2xbf16> {
  //   // CHECK-LABEL: func.func @test_commute_permute_reshape_1_bc_a
  //   // CHECK: ttir.permute
  //   // CHECK: ttir.reshape
  //   // CHECK: ttir.permute
  //   %0 = ttir.empty() : tensor<2x2x2xbf16>
  //   %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 1, 2, 0>}> : (tensor<2x2x2xbf16>, tensor<2x2x2xbf16>) -> tensor<2x2x2xbf16>
  //   %2 = ttir.empty() : tensor<4x1x2xbf16>
  //   %3 = "ttir.reshape"(%1, %2) <{shape = [4 : i32, 1 : i32, 2 : i32]}> : (tensor<2x2x2xbf16>, tensor<4x1x2xbf16>) -> tensor<4x1x2xbf16>
  //   %4 = ttir.empty() : tensor<1x4x2xbf16>
  //   %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 1, 0, 2>}> : (tensor<4x1x2xbf16>, tensor<1x4x2xbf16>) -> tensor<1x4x2xbf16>
  //   return %5 : tensor<1x4x2xbf16>
  // }

  // func.func @test_commute_permute_reshape_non_divisible(%arg0: tensor<4x4x4xbf16>) -> tensor<2x32x1xbf16> {
  //   // CHECK-LABEL: func.func @test_commute_permute_reshape_non_divisible
  //   // CHECK: ttir.permute
  //   // CHECK: ttir.reshape
  //   // CHECK: ttir.permute
  //   %0 = ttir.empty() : tensor<4x4x4xbf16>
  //   %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 2, 1>}> : (tensor<4x4x4xbf16>, tensor<4x4x4xbf16>) -> tensor<4x4x4xbf16>
  //   %2 = ttir.empty() : tensor<2x1x32xbf16>
  //   %3 = "ttir.reshape"(%1, %2) <{shape = [2 : i32, 1 : i32, 32 : i32]}> : (tensor<4x4x4xbf16>, tensor<2x1x32xbf16>) -> tensor<2x1x32xbf16>
  //   %4 = ttir.empty() : tensor<2x32x1xbf16>
  //   %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 2, 1>}> : (tensor<2x1x32xbf16>, tensor<2x32x1xbf16>) -> tensor<2x32x1xbf16>
  //   return %5 : tensor<2x32x1xbf16>
  // }

}
