// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround --canonicalize %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1x2x3x12x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1x6x2x6x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1x6x1x12x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module @test_complex_reshape_workaround attributes {} {
  // Test Type 1 decomposition: Bi == Bo && Ci * Hi == Co && Ho * Wo == Wi && Ho > 1 && Hi > 1
  // Input:  [1, 2, 3, 12] -> Output: [1, 6, 2, 6]
  // Expected decomposition: [1, 2, 3, 12] -> [1, 6, 1, 12] -> [1, 6, 2, 6]
  func.func public @test_complex_reshape_type1(%arg0: tensor<1x2x3x12xf32, #ttnn_layout1>) -> tensor<1x6x2x6xf32, #ttnn_layout2> {
    // CHECK-LABEL: func.func public @test_complex_reshape_type1
    // CHECK: %[[FIRST_RESHAPE:[0-9]+]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: {complex_reshape_decomposed, shape = [1 : i32, 6 : i32, 1 : i32, 12 : i32]}
    // CHECK-SAME: (tensor<1x2x3x12xf32, #{{.*}}>) -> tensor<1x6x1x12xf32, #{{.*}}>
    // CHECK-SAME: loc("reshape_0_intermediate_reshape")
    // CHECK: %[[SECOND_RESHAPE:[0-9]+]] = "ttnn.reshape"(%[[FIRST_RESHAPE]])
    // CHECK-SAME: {complex_reshape_decomposed, shape = [1 : i32, 6 : i32, 2 : i32, 6 : i32]}
    // CHECK-SAME: (tensor<1x6x1x12xf32, #{{.*}}>) -> tensor<1x6x2x6xf32, #{{.*}}>
    // CHECK: return %[[SECOND_RESHAPE]]
    %0 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 6 : i32, 2 : i32, 6 : i32]}> : (tensor<1x2x3x12xf32, #ttnn_layout1>) -> tensor<1x6x2x6xf32, #ttnn_layout2> loc("reshape_0")
    return %0 : tensor<1x6x2x6xf32, #ttnn_layout2>
  }

  // Test Type 2 decomposition: Bi == Bo && Co * Ho == Ci && Hi * Wi == Wo && Ho > 1 && Wi > 1
  // Input:  [1, 6, 2, 6] -> Output: [1, 2, 3, 12]
  // Expected decomposition: [1, 6, 2, 6] -> [1, 6, 1, 12] -> [1, 2, 3, 12]
  func.func public @test_complex_reshape_type2(%arg0: tensor<1x6x2x6xf32, #ttnn_layout2>) -> tensor<1x2x3x12xf32, #ttnn_layout1> {
    // CHECK-LABEL: func.func public @test_complex_reshape_type2
    // CHECK: %[[FIRST_RESHAPE:[0-9]+]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: {complex_reshape_decomposed, shape = [1 : i32, 6 : i32, 1 : i32, 12 : i32]}
    // CHECK-SAME: (tensor<1x6x2x6xf32, #{{.*}}>) -> tensor<1x6x1x12xf32, #{{.*}}>
    // CHECK-SAME: loc("reshape_1_intermediate_reshape")
    // CHECK: %[[SECOND_RESHAPE:[0-9]+]] = "ttnn.reshape"(%[[FIRST_RESHAPE]])
    // CHECK-SAME: {complex_reshape_decomposed, shape = [1 : i32, 2 : i32, 3 : i32, 12 : i32]}
    // CHECK-SAME: (tensor<1x6x1x12xf32, #{{.*}}>) -> tensor<1x2x3x12xf32, #{{.*}}>
    // CHECK: return %[[SECOND_RESHAPE]]
    %0 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 2 : i32, 3 : i32, 12 : i32]}> : (tensor<1x6x2x6xf32, #ttnn_layout2>) -> tensor<1x2x3x12xf32, #ttnn_layout1> loc("reshape_1")
    return %0 : tensor<1x2x3x12xf32, #ttnn_layout1>
  }

  // Negative test: Ho = 1, should NOT decompose (fails Ho > 1 constraint)
  // Input:  [1, 2, 3, 12] -> Output: [1, 6, 1, 12]
  func.func public @test_complex_reshape_no_match_ho_equals_1(%arg0: tensor<1x2x3x12xf32, #ttnn_layout1>) -> tensor<1x6x1x12xf32, #ttnn_layout3> {
    // CHECK-LABEL: func.func public @test_complex_reshape_no_match_ho_equals_1
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: {shape = [1 : i32, 6 : i32, 1 : i32, 12 : i32]}
    // CHECK-NOT: complex_reshape_decomposed
    // CHECK-NOT: _intermediate_reshape
    // CHECK: return %[[RESHAPE]]
    %0 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 6 : i32, 1 : i32, 12 : i32]}> : (tensor<1x2x3x12xf32, #ttnn_layout1>) -> tensor<1x6x1x12xf32, #ttnn_layout3> loc("reshape_2")
    return %0 : tensor<1x6x1x12xf32, #ttnn_layout3>
  }

  // Negative test: Hi = 1, should NOT decompose (fails Hi > 1 constraint for Type 1)
  // Input:  [1, 6, 1, 12] -> Output: [1, 6, 2, 6]
  func.func public @test_complex_reshape_no_match_hi_equals_1(%arg0: tensor<1x6x1x12xf32, #ttnn_layout3>) -> tensor<1x6x2x6xf32, #ttnn_layout2> {
    // CHECK-LABEL: func.func public @test_complex_reshape_no_match_hi_equals_1
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: {shape = [1 : i32, 6 : i32, 2 : i32, 6 : i32]}
    // CHECK-NOT: complex_reshape_decomposed
    // CHECK-NOT: _intermediate_reshape
    // CHECK: return %[[RESHAPE]]
    %0 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 6 : i32, 2 : i32, 6 : i32]}> : (tensor<1x6x1x12xf32, #ttnn_layout3>) -> tensor<1x6x2x6xf32, #ttnn_layout2> loc("reshape_3")
    return %0 : tensor<1x6x2x6xf32, #ttnn_layout2>
  }

  // Negative test: Wi = 1, should NOT decompose (fails Wi > 1 constraint for Type 2)
  // Input:  [1, 6, 2, 1] -> Output: [1, 2, 3, 2]
  func.func public @test_complex_reshape_no_match_wi_equals_1(%arg0: tensor<1x6x2x1xf32>) -> tensor<1x2x3x2xf32> {
    // CHECK-LABEL: func.func public @test_complex_reshape_no_match_wi_equals_1
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: {shape = [1 : i32, 2 : i32, 3 : i32, 2 : i32]}
    // CHECK-NOT: complex_reshape_decomposed
    // CHECK-NOT: _intermediate_reshape
    // CHECK: return %[[RESHAPE]]
    %0 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 2 : i32, 3 : i32, 2 : i32]}> : (tensor<1x6x2x1xf32>) -> tensor<1x2x3x2xf32> loc("reshape_4")
    return %0 : tensor<1x2x3x2xf32>
  }
}
