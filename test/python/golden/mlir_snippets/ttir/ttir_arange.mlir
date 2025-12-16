module {
  func.func @arange(%arg0: tensor<5xf32>) -> tensor<5xf32> {
    %0 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 5 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<5xf32>
    return %0 : tensor<5xf32>
  }
}
