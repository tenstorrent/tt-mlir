module {
  func.func @pairwise_mul_add_collapse_512x2_i64(%arg0: tensor<512x2xi64>, %arg1: tensor<512x1xi64>) -> tensor<512xi64> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [512 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<512x2xi64>) -> tensor<512x1xi64>
    %1 = "ttir.slice_static"(%arg0) <{begins = [0 : i32, 1 : i32], ends = [512 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<512x2xi64>) -> tensor<512x1xi64>
    %2 = "ttir.multiply"(%0, %arg1) : (tensor<512x1xi64>, tensor<512x1xi64>) -> tensor<512x1xi64>
    %3 = "ttir.add"(%2, %1) : (tensor<512x1xi64>, tensor<512x1xi64>) -> tensor<512x1xi64>
    %4 = "ttir.reshape"(%3) <{shape = [512 : i32]}> : (tensor<512x1xi64>) -> tensor<512xi64>
    return %4 : tensor<512xi64>
  }
}
