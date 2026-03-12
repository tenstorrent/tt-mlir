module @jit_sort_key_value attributes {} {
  func.func public @test_sort_key_value(%arg0: tensor<4x4x4xbf16>, %arg1: tensor<4x4x4xbf16>) -> (tensor<4x4x4xbf16>, tensor<4x4x4xbf16>) {
    %0:2 = "stablehlo.sort"(%arg0, %arg1) <{dimension = 2 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>, %arg4: tensor<bf16>, %arg5: tensor<bf16>):
      %1 = stablehlo.compare  LT, %arg2, %arg3,  TOTALORDER : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<4x4x4xbf16>, tensor<4x4x4xbf16>) -> (tensor<4x4x4xbf16>, tensor<4x4x4xbf16>)
    return %0#0, %0#1 : tensor<4x4x4xbf16>, tensor<4x4x4xbf16>
  }
}
