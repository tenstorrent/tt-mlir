// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline --split-input-file  -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module {
  sdy.mesh @mesh = <["model"=1, "batch"=2]>
  func.func @all_slice_replicated_input(%arg0: tensor<32xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> tensor<32xbf16> {
    %0 = sdy.all_slice [{"batch"}] %arg0 out_sharding=<@mesh, [{"batch"}]> : tensor<32xbf16>
    return %0 : tensor<32xbf16>
  }
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}]>] manual_axes={"model", "batch"}
// CHECK: stablehlo.reshape
// CHECK-SAME: -> tensor<2x16xbf16>
// CHECK: stablehlo.all_to_all
// CHECK-SAME: (tensor<2x16xbf16>) -> tensor<2x16xbf16>
// CHECK: stablehlo.slice
// CHECK-SAME: [0:1, 0:16] : (tensor<2x16xbf16>) -> tensor<1x16xbf16>
// CHECK: stablehlo.reshape
// CHECK-SAME: -> tensor<16xbf16>
// CHECK: sdy.return

// -----

module {
  sdy.mesh @mesh = <["model"=2, "batch"=4]>
  func.func @all_slice_2d_replicated_input(%arg0: tensor<4x32xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> tensor<4x32xbf16> {
    %0 = sdy.all_slice [{"batch"}, {"model"}] %arg0 out_sharding=<@mesh, [{"batch"}, {"model"}]> : tensor<4x32xbf16>
    return %0 : tensor<4x32xbf16>
  }
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"batch"}, {"model"}]>] manual_axes={"model", "batch"}
// CHECK: stablehlo.reshape
// CHECK-SAME: -> tensor<4x1x32xbf16>
// CHECK: stablehlo.all_to_all
// CHECK-SAME: (tensor<4x1x32xbf16>) -> tensor<4x1x32xbf16>
// CHECK: stablehlo.slice
// CHECK-SAME: [0:1, 0:1, 0:32] : (tensor<4x1x32xbf16>) -> tensor<1x1x32xbf16>
// CHECK: stablehlo.reshape
// CHECK-SAME: -> tensor<1x32xbf16>
// CHECK: stablehlo.reshape
// CHECK-SAME: -> tensor<1x2x16xbf16>
// CHECK: stablehlo.all_to_all
// CHECK-SAME: (tensor<1x2x16xbf16>) -> tensor<1x2x16xbf16>
// CHECK: stablehlo.slice
// CHECK-SAME: [0:1, 0:1, 0:16] : (tensor<1x2x16xbf16>) -> tensor<1x1x16xbf16>
// CHECK: stablehlo.reshape
// CHECK-SAME: -> tensor<1x16xbf16>
// CHECK: sdy.return

// -----

module {
  sdy.mesh @mesh = <["model"=2, "batch"=4]>
  func.func @all_slice_2d_partial_sharded_input(%arg0: tensor<4x32xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}) -> tensor<4x32xbf16> {
    %0 = sdy.all_slice [{}, {"model"}] %arg0 out_sharding=<@mesh, [{"batch"}, {"model"}]> : tensor<4x32xbf16>
    return %0 : tensor<4x32xbf16>
  }
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}]>] out_shardings=[<@mesh, [{"batch"}, {"model"}]>] manual_axes={"model", "batch"} (%arg1: tensor<1x32xbf16>)
// CHECK: stablehlo.reshape
// CHECK-SAME: -> tensor<1x2x16xbf16>
// CHECK: stablehlo.all_to_all
// CHECK-SAME: (tensor<1x2x16xbf16>) -> tensor<1x2x16xbf16>
// CHECK: stablehlo.slice
// CHECK-SAME: [0:1, 0:1, 0:16] : (tensor<1x2x16xbf16>) -> tensor<1x1x16xbf16>
// CHECK: stablehlo.reshape
// CHECK-SAME: -> tensor<1x16xbf16>
// CHECK: sdy.return
