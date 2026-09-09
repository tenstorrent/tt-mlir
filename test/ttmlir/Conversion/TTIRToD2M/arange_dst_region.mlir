// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttmetal-pipeline %s -o /dev/null

// Regression test for d2m-split-unified-thread wrapping DST-dependent pure ops.
// Arange lowers to a DST sequence where pure tile_add depends on a non-pure
// memref.load from dst; the wrapper must include the tile_add result store.

module {
  func.func @arange_1x128_f32(%arg0: tensor<1x128xf32>) -> tensor<1x128xf32> {
    %0 = "ttir.arange"() <{arange_dimension = 1 : i64, end = 128 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<1x128xf32>
    return %0 : tensor<1x128xf32>
  }
}
