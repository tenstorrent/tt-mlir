// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Shardy emits stablehlo.composite "sdy.all_slice" as a DELTA from the
// operand's existing sharding. Here dim0 is sharded over both mesh axes, but a
// reduce_scatter has already scattered "_axis_0" (256 -> 128), so only
// "_axis_1" is still outstanding (128 -> 32). Legalizing every axis named in
// out_sharding would divide dim0 twice (128 -> 64 -> 16) and leave a value
// whose type no longer matches the composite's declared 32x2112 result, which
// used to fail the whole pipeline with:
//   error: failed to legalize unresolved materialization from
//     ('tensor<16x2112xbf16>') to ('tensor<32x2112xbf16>')

module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @all_slice_after_reduce_scatter(%arg0: tensor<256x2112xbf16>) -> tensor<256x2112xbf16> {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"_axis_0", "_axis_1"}, {}]>] manual_axes={"_axis_0", "_axis_1"} (%arg1: tensor<256x2112xbf16>) {
      %1 = "stablehlo.reduce_scatter"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
      ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
        %3 = stablehlo.add %arg2, %arg3 : tensor<bf16>
        stablehlo.return %3 : tensor<bf16>
      }) : (tensor<256x2112xbf16>) -> tensor<128x2112xbf16>
      %2 = stablehlo.composite "sdy.all_slice" %1 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0", "_axis_1"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<128x2112xbf16>) -> tensor<32x2112xbf16>
      sdy.return %2 : tensor<32x2112xbf16>
    } : (tensor<256x2112xbf16>) -> tensor<256x2112xbf16>
    return %0 : tensor<256x2112xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<128x2112xbf16>) -> tensor<32x2112xbf16> {
    %0 = stablehlo.reshape %arg0 : (tensor<128x2112xbf16>) -> tensor<4x32x2112xbf16>
    %1 = stablehlo.slice %0 [0:1, 0:32, 0:2112] : (tensor<4x32x2112xbf16>) -> tensor<1x32x2112xbf16>
    %2 = stablehlo.reshape %1 : (tensor<1x32x2112xbf16>) -> tensor<32x2112xbf16>
    return %2 : tensor<32x2112xbf16>
  }
}

// CHECK-LABEL: func.func @all_slice_after_reduce_scatter
// CHECK: "ttir.reduce_scatter"
// CHECK-SAME: -> tensor<128x2112xbf16>
// Only "_axis_1" (cluster_axis 1) is left to apply.
// CHECK: "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<128x2112xbf16>) -> tensor<32x2112xbf16>
// CHECK-NOT: "ttir.mesh_partition"
// CHECK-NOT: func.func private @sdy.all_slice1
