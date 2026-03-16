// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @SyncTensorsGraph.81 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1x480x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "args_0"}) -> tensor<1x480x1x64xbf16> {
    %0 = stablehlo.composite "tenstorrent.group_norm" %arg0 {composite_attributes = {channel_dim = 1 : i64, epsilon = 9.99999974E-6 : f32, num_groups = 8 : i64}, decomposition = @tenstorrent.group_norm.impl} : (tensor<1x480x1x64xbf16>) -> tensor<1x480x1x64xbf16>
    // CHECK: "ttir.group_norm"(%{{.*}})
    return %0 : tensor<1x480x1x64xbf16>
  }
  func.func private @tenstorrent.group_norm.impl(%arg0: tensor<1x480x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<1x480x1x64xbf16> {
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<1x8x1x1xf32>
    %cst_0 = stablehlo.constant dense<2.6041668E-4> : tensor<1x8xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg0 : (tensor<1x480x1x64xbf16>) -> tensor<1x480x1x64xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x480x1x64xf32>) -> tensor<1x8x60x64xf32>
    %2 = stablehlo.reduce(%1 init: %cst_1) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x8x60x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<1x8xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x60x64xf32>
    %5 = stablehlo.subtract %1, %4 : tensor<1x8x60x64xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x8x60x64xf32>
    %7 = stablehlo.reduce(%6 init: %cst_1) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x8x60x64xf32>, tensor<f32>) -> tensor<1x8xf32>
    %8 = stablehlo.multiply %7, %cst_0 : tensor<1x8xf32>
    %9 = stablehlo.reshape %8 : (tensor<1x8xf32>) -> tensor<1x8x1x1xf32>
    %10 = stablehlo.add %9, %cst : tensor<1x8x1x1xf32>
    %11 = stablehlo.rsqrt %10 : tensor<1x8x1x1xf32>
    %12 = stablehlo.reshape %11 : (tensor<1x8x1x1xf32>) -> tensor<1x8xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<1x8x60x64xf32>
    %14 = stablehlo.multiply %5, %13 : tensor<1x8x60x64xf32>
    %15 = stablehlo.reshape %14 : (tensor<1x8x60x64xf32>) -> tensor<1x480x1x64xf32>
    %16 = stablehlo.convert %15 : (tensor<1x480x1x64xf32>) -> tensor<1x480x1x64xbf16>
    return %16 : tensor<1x480x1x64xbf16>
  }
}
