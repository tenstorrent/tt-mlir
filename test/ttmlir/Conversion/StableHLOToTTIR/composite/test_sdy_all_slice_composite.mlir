// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  sdy.mesh @mesh = <["model"=2, "batch"=4]>
  func.func @test_sdy_all_slice_composite(%arg0: tensor<4x32xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<4x32xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"batch"}, {"model"}]>] manual_axes={"model", "batch"} (%arg1: tensor<4x32xbf16>) {
      %1 = stablehlo.composite "sdy.all_slice" %arg1 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}, decomposition = @sdy.all_slice1} : (tensor<4x32xbf16>) -> tensor<1x16xbf16>
      sdy.return %1 : tensor<1x16xbf16>
    } : (tensor<4x32xbf16>) -> tensor<4x32xbf16>
    return %0 : tensor<4x32xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<4x32xbf16>) -> tensor<1x16xbf16> {
    %0 = stablehlo.reshape %arg0 : (tensor<4x32xbf16>) -> tensor<4x1x32xbf16>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, split_count = 4 : i64, split_dimension = 0 : i64}> : (tensor<4x1x32xbf16>) -> tensor<4x1x32xbf16>
    %2 = stablehlo.slice %1 [0:1, 0:1, 0:32] : (tensor<4x1x32xbf16>) -> tensor<1x1x32xbf16>
    %3 = stablehlo.reshape %2 : (tensor<1x1x32xbf16>) -> tensor<1x32xbf16>
    %4 = stablehlo.reshape %3 : (tensor<1x32xbf16>) -> tensor<1x2x16xbf16>
    %5 = "stablehlo.all_to_all"(%4) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<1x2x16xbf16>) -> tensor<1x2x16xbf16>
    %6 = stablehlo.slice %5 [0:1, 0:1, 0:16] : (tensor<1x2x16xbf16>) -> tensor<1x1x16xbf16>
    %7 = stablehlo.reshape %6 : (tensor<1x1x16xbf16>) -> tensor<1x16xbf16>
    return %7 : tensor<1x16xbf16>
  }
}

// CHECK-LABEL: func.func @test_sdy_all_slice_composite
// CHECK: "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 0 : ui32, dim = 1 : si32}> : (tensor<4x32xbf16>) -> tensor<4x16xbf16>
// CHECK: "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<4x16xbf16>) -> tensor<1x16xbf16>
// CHECK-NOT: func.func private @sdy.all_slice1
