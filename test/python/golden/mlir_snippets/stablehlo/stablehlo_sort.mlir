module @jit_sort attributes {} {
  func.func public @test_sort(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    %0 = "stablehlo.sort"(%arg0) <{dimension = 1 : i64, is_stable = true}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %1 = stablehlo.compare  LT, %arg1, %arg2,  FLOAT : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %0 : tensor<64x128xbf16>
  }
}