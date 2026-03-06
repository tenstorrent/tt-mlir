func.func @test(%arg0: tensor<32x5120xf32>, %arg1: tensor<5120x30720xbf16>, %arg2: tensor<30720xbf16>, %arg3: tensor<1x49920x6x5120xf32>, %arg4: tensor<1x49920x5120xf32>) -> tensor<1x49920x5120xf32> {
  %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x5120xf32>, tensor<5120x30720xbf16>) -> tensor<32x30720xf32>
  %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 32 : i32, 30720 : i32]}> : (tensor<32x30720xf32>) -> tensor<1x32x30720xf32>
  %2 = "ttir.typecast"(%arg2) <{conservative_folding = false}> : (tensor<30720xbf16>) -> tensor<30720xf32>
  %3 = "ttir.reshape"(%2) <{shape = [1 : i32, 1 : i32, 30720 : i32]}> : (tensor<30720xf32>) -> tensor<1x1x30720xf32>
  %4 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 32, 1>}> : (tensor<1x1x30720xf32>) -> tensor<1x32x30720xf32>
  %5 = "ttir.add"(%1, %4) : (tensor<1x32x30720xf32>, tensor<1x32x30720xf32>) -> tensor<1x32x30720xf32>
  %6 = "ttir.reshape"(%5) <{shape = [1 : i32, 32 : i32, 6 : i32, 5120 : i32]}> : (tensor<1x32x30720xf32>) -> tensor<1x32x6x5120xf32>
  %7 = "ttir.reshape"(%6) <{shape = [1 : i32, 32 : i32, 1 : i32, 6 : i32, 5120 : i32]}> : (tensor<1x32x6x5120xf32>) -> tensor<1x32x1x6x5120xf32>
  %8 = "ttir.broadcast"(%7) <{broadcast_dimensions = array<i64: 1, 1, 1560, 1, 1>}> : (tensor<1x32x1x6x5120xf32>) -> tensor<1x32x1560x6x5120xf32>
  %9 = "ttir.reshape"(%8) <{shape = [1 : i32, 49920 : i32, 6 : i32, 5120 : i32]}> : (tensor<1x32x1560x6x5120xf32>) -> tensor<1x49920x6x5120xf32>
  %10 = "ttir.add"(%arg3, %9) : (tensor<1x49920x6x5120xf32>, tensor<1x49920x6x5120xf32>) -> tensor<1x49920x6x5120xf32>
  %11 = "ttir.slice_static"(%10) <{begins = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ends = [1 : i32, 49920 : i32, 2 : i32, 5120 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x49920x6x5120xf32>) -> tensor<1x49920x1x5120xf32>
  %12 = "ttir.reshape"(%11) <{shape = [1 : i32, 49920 : i32, 5120 : i32]}> : (tensor<1x49920x1x5120xf32>) -> tensor<1x49920x5120xf32>
  %13 = "ttir.add"(%12, %arg4) : (tensor<1x49920x5120xf32>, tensor<1x49920x5120xf32>) -> tensor<1x49920x5120xf32>
  return %13 : tensor<1x49920x5120xf32>
}
