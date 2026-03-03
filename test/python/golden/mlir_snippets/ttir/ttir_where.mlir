module {
  func.func public @where_predicate_different_than_input(%arg0: tensor<13x37xf32>, %arg1: tensor<13x37xbf16>, %arg2: tensor<13x37xbf16>) -> tensor<13x37xbf16> {
    %1 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<13x37xf32>, tensor<13x37xbf16>, tensor<13x37xbf16>) -> tensor<13x37xbf16>
    return %1 : tensor<13x37xbf16>
  }
}
