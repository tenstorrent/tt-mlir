module @jit_apply attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @alexNet(%arg0: tensor<32x12x12x192xbf16> {mhlo.sharding = "{replicated}"}, %arg1: tensor<384x192x3x3xbf16> {mhlo.sharding = "{replicated}"}) -> (tensor<32x12x12x384xbf16> {jax.result_info = ""}) {
    %59 = ttir.empty() : tensor<32x12x12x384xbf16>
    %60 = "ttir.conv2d"(%arg0, %arg1, %59) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> : (tensor<32x12x12x192xbf16>, tensor<384x192x3x3xbf16>, tensor<32x12x12x384xbf16>) -> tensor<32x12x12x384xbf16>
    return %60 : tensor<32x12x12x384xbf16>
  }
}

