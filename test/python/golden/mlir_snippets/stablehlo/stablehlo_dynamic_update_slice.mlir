module @jit_dynamic_update_slice attributes {} {
  func.func public @test_dynamic_update_slice(%operand: tensor<8x8xf32>, %update: tensor<2x4xf32>, %start0: tensor<i32>, %start1: tensor<i32>) -> tensor<8x8xf32> {
    %0 = stablehlo.dynamic_update_slice %operand, %update, %start0, %start1 : (tensor<8x8xf32>, tensor<2x4xf32>, tensor<i32>, tensor<i32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}
