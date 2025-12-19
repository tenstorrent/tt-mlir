module {
  func.func @my_modela(%arg0: tensor<256x2048xi1>, %arg1: tensor<256x2048xbf16>, %arg2: tensor<256x2048xbf16>, %arg3: tensor<256x1xi64>) -> tensor<51200x2048xbf16> {
    %0 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<51200x2048xbf16>}> : () -> tensor<51200x2048xbf16>
    %1 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<256x2048xi1>, tensor<256x2048xbf16>, tensor<256x2048xbf16>) -> tensor<256x2048xbf16>
    %2 = "ttir.repeat"(%arg3) <{repeat_dimensions = array<i64: 1, 2048>}> : (tensor<256x1xi64>) -> tensor<256x2048xi64>
    %3 = "ttir.scatter"(%0, %2, %1) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<sum>}> : (tensor<51200x2048xbf16>, tensor<256x2048xi64>, tensor<256x2048xbf16>) -> tensor<51200x2048xbf16>
    return %3 : tensor<51200x2048xbf16>
  }
}
