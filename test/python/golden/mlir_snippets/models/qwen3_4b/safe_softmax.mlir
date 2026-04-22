// Numerically stable softmax with masking from Qwen3 4B attention
// subtract -> exp -> sum -> div -> where(mask, zeros, softmax)

module {
  func.func @safe_softmax(%arg0: tensor<32x32x18x18xf32>, %arg1: tensor<32x32x18x18xi1>) -> tensor<32x32x18x18xf32> {
    %zeros = "ttir.full"() <{fill_value = 0.000000e+00 : f32, shape = array<i32: 32, 32, 18, 18>}> : () -> tensor<32x32x18x18xf32>
    %0 = "ttir.max"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x32x18x18xf32>) -> tensor<32x32x18xf32>
    %1 = "ttir.reshape"(%0) <{shape = [32 : i32, 32 : i32, 18 : i32, 1 : i32]}> : (tensor<32x32x18xf32>) -> tensor<32x32x18x1xf32>
    %2 = "ttir.broadcast"(%1) <{broadcast_dimensions = array<i64: 1, 1, 1, 18>}> : (tensor<32x32x18x1xf32>) -> tensor<32x32x18x18xf32>
    %3 = "ttir.subtract"(%arg0, %2) : (tensor<32x32x18x18xf32>, tensor<32x32x18x18xf32>) -> tensor<32x32x18x18xf32>
    %4 = "ttir.exp"(%3) : (tensor<32x32x18x18xf32>) -> tensor<32x32x18x18xf32>
    %5 = "ttir.sum"(%4) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x32x18x18xf32>) -> tensor<32x32x18xf32>
    %6 = "ttir.reshape"(%5) <{shape = [32 : i32, 32 : i32, 18 : i32, 1 : i32]}> : (tensor<32x32x18xf32>) -> tensor<32x32x18x1xf32>
    %7 = "ttir.broadcast"(%6) <{broadcast_dimensions = array<i64: 1, 1, 1, 18>}> : (tensor<32x32x18x1xf32>) -> tensor<32x32x18x18xf32>
    %8 = "ttir.div"(%4, %7) : (tensor<32x32x18x18xf32>, tensor<32x32x18x18xf32>) -> tensor<32x32x18x18xf32>
    %9 = "ttir.where"(%arg1, %zeros, %8) : (tensor<32x32x18x18xi1>, tensor<32x32x18x18xf32>, tensor<32x32x18x18xf32>) -> tensor<32x32x18x18xf32>
    return %9 : tensor<32x32x18x18xf32>
  }
}
