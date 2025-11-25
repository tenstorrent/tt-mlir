// RUN: ttmlir-opt --ttir-erase-inverse-ops="enable-commute-upwards=false" -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
  // Positive examples:

  // [N, H, W, C]  -> [N, C, H, W]
  // [N, C, H, W]  -> [N, 1, C, HW]
  func.func @test_commute_permute_n_1_c_hw(%arg0: tensor<16x7x7x2048xbf16>) -> tensor<16x1x2048x49xbf16> {
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [16 : i32, 2048 : i32, 49 : i32, 1 : i32]}>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[RESHAPE]], %{{[0-9]+}}) <{permutation = array<i64: 0, 3, 1, 2>}>
    // CHECK: return %[[PERMUTE]]
    %0 = tensor.empty() : tensor<16x2048x7x7xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<16x7x7x2048xbf16>, tensor<16x2048x7x7xbf16>) -> tensor<16x2048x7x7xbf16>
    %2 = tensor.empty() : tensor<16x1x2048x49xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [16:i32, 1: i32, 2048: i32, 49: i32]}> : (tensor<16x2048x7x7xbf16>, tensor<16x1x2048x49xbf16>) -> tensor<16x1x2048x49xbf16>
    return %3: tensor<16x1x2048x49xbf16>
  }

  // [N, H, W, C]  -> [N, C, H, W]
  // [N, C, H, W]  -> [N, C, 1, HW]
  func.func @test_commute_permute_reshape_n_c_1_hw(%arg0: tensor<12x7x7x1152xbf16>) -> tensor<12x1152x1x49xbf16> {
    // CHECK-LABEL: func.func @test_commute_permute_reshape_n_c_1_hw
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [12 : i32, 1 : i32, 49 : i32, 1152 : i32]}>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[RESHAPE]], %{{[0-9]+}}) <{permutation = array<i64: 0, 3, 1, 2>}>
    // CHECK: return %[[PERMUTE]]
    %0 = ttir.empty() : tensor<12x1152x7x7xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<12x7x7x1152xbf16>, tensor<12x1152x7x7xbf16>) -> tensor<12x1152x7x7xbf16>
    %2 = ttir.empty() : tensor<12x1152x1x49xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [12 : i32, 1152 : i32, 1 : i32, 49 : i32]}> : (tensor<12x1152x7x7xbf16>, tensor<12x1152x1x49xbf16>) -> tensor<12x1152x1x49xbf16>
    return %3 : tensor<12x1152x1x49xbf16>
  }

  // [A, B, C, D]   -> [B, C, D, A]
  // [B, C, D, A]   -> [BCD, A, 1, 1]
  func.func @test_commute_permute_reshape_1_1_a_bcd(%arg0: tensor<2x2x2x2xbf16>) -> tensor<8x2x1x1xbf16> {
    // CHECK-LABEL: func.func @test_commute_permute_reshape_1_1_a_bcd
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [1 : i32, 8 : i32, 2 : i32, 1 : i32]}>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[RESHAPE]], %{{[0-9]+}}) <{permutation = array<i64: 1, 2, 3, 0>}>
    // CHECK: return %[[PERMUTE]]
    %0 = ttir.empty() : tensor<2x2x2x2xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 1, 2, 3, 0>}> : (tensor<2x2x2x2xbf16>, tensor<2x2x2x2xbf16>) -> tensor<2x2x2x2xbf16>
    %2 = ttir.empty() : tensor<8x2x1x1xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [8 : i32, 2 : i32, 1 : i32, 1 : i32]}> : (tensor<2x2x2x2xbf16>, tensor<8x2x1x1xbf16>) -> tensor<8x2x1x1xbf16>
    return %3 : tensor<8x2x1x1xbf16>
  }

  // [A, B, C, D, E]   -> [A, D, E, B, C]
  // [A, D, E, B, C]   -> [A, DE, 1, 1, BC]
  func.func @test_commute_permute_reshape_a_1_bc_1_de(%arg0: tensor<2x3x5x7x11xbf16>) -> tensor<2x77x1x1x15xbf16> {
    // CHECK-LABEL: func.func @test_commute_permute_reshape_a_1_bc_1_de
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [2 : i32, 1 : i32, 15 : i32, 77 : i32, 1 : i32]}>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[RESHAPE]], %{{[0-9]+}}) <{permutation = array<i64: 0, 3, 4, 1, 2>}>
    // CHECK: return %[[PERMUTE]]
    %0 = ttir.empty() : tensor<2x7x11x3x5xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 4, 1, 2>}> : (tensor<2x3x5x7x11xbf16>, tensor<2x7x11x3x5xbf16>) -> tensor<2x7x11x3x5xbf16>
    %2 = ttir.empty() : tensor<2x77x1x1x15xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [2 : i32, 77 : i32, 1 : i32, 1 : i32, 15 : i32]}> : (tensor<2x7x11x3x5xbf16>, tensor<2x77x1x1x15xbf16>) -> tensor<2x77x1x1x15xbf16>
    return %3 : tensor<2x77x1x1x15xbf16>
  }

  // [A, B, C, D]   -> [C, D, A, B]
  // [C, D, A, B]   -> [CD, 1, 1, AB]
  func.func @test_commute_permute_reshape_ab_1_1_cd(%arg0: tensor<2x2x2x2xbf16>) -> tensor<1x4x1x4xbf16> {
    // CHECK-LABEL: func.func @test_commute_permute_reshape_ab_1_1_cd
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0, %{{[0-9]+}}) <{shape = [1 : i32, 4 : i32, 1 : i32, 4 : i32]}>
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%[[RESHAPE]], %{{[0-9]+}}) <{permutation = array<i64: 2, 3, 0, 1>}>
    // CHECK: return %[[PERMUTE]]
    %0 = ttir.empty() : tensor<2x2x2x2xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 2, 3, 0, 1>}> : (tensor<2x2x2x2xbf16>, tensor<2x2x2x2xbf16>) -> tensor<2x2x2x2xbf16>
    %2 = ttir.empty() : tensor<1x4x1x4xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [1 : i32, 4 : i32, 1 : i32, 4 : i32]}> : (tensor<2x2x2x2xbf16>, tensor<1x4x1x4xbf16>) -> tensor<1x4x1x4xbf16>
    return %3 : tensor<1x4x1x4xbf16>
  }

}
