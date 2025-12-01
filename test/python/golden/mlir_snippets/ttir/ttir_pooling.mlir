module {
  func.func @pooling_non_identity_window_no_fold(%arg0: tensor<1x32x32x16xf32>) -> tensor<1x16x16x16xf32> {
    %1 = "ttir.pooling"(%arg0) <{
        pooling_method = #ttir<pooling_method Max>,
        window_dimensions = array<i64: 1, 2, 2, 1>,
        window_strides = array<i64: 1, 2, 2, 1>,
        base_dilations = array<i64: 1, 1, 1, 1>,
        window_dilations = array<i64: 1, 1, 1, 1>,
        padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>,
        operandSegmentSizes = array<i32: 1, 1>
    }> : (tensor<1x32x32x16xf32>) -> tensor<1x16x16x16xf32>
    return %1 : tensor<1x16x16x16xf32>
  }
}
