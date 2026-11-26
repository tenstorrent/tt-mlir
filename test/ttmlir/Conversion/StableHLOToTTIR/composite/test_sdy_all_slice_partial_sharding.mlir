// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --legalize-stablehlo-composite-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t

// Shardy emits stablehlo.composite "sdy.all_slice" as a DELTA from the
// operand's existing sharding: out_sharding is the sharding of the result, so
// the axes the operand already carries (a major prefix of each dim's axis list)
// must not be applied again. The composite's declared result type is what says
// how much is still outstanding.
//
// These tests exercise the pattern in isolation. Because this pass leaves
// StableHLO neither legal nor illegal, a composite the pattern cannot handle is
// left untouched for later passes to inline, so the negative cases check that
// the composite survives instead of being rewritten to a wrong type.

// Nothing of out_sharding has been applied yet: the single axis is outstanding.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @replicated_operand_single_axis(%arg0: tensor<64x32xbf16>) -> tensor<16x32xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<64x32xbf16>) -> tensor<16x32xbf16>
    return %0 : tensor<16x32xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<64x32xbf16>) -> tensor<16x32xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<16x32xbf16>
    return %0 : tensor<16x32xbf16>
  }
}

// CHECK-LABEL: func.func @replicated_operand_single_axis
// CHECK: "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<64x32xbf16>) -> tensor<16x32xbf16>
// CHECK-NOT: "ttir.mesh_partition"
// CHECK-NOT: stablehlo.composite

// -----

// dim0 is sharded over both axes but the operand already carries "_axis_0"
// (e.g. it comes out of a reduce_scatter over "_axis_0"), so only "_axis_1"
// is outstanding: 128 / 4 = 32, not 128 / 2 / 4 = 16.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @partly_sharded_operand(%arg0: tensor<128x2112xbf16>) -> tensor<32x2112xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0", "_axis_1"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<128x2112xbf16>) -> tensor<32x2112xbf16>
    return %0 : tensor<32x2112xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<128x2112xbf16>) -> tensor<32x2112xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x2112xbf16>
    return %0 : tensor<32x2112xbf16>
  }
}

// CHECK-LABEL: func.func @partly_sharded_operand
// CHECK: "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<128x2112xbf16>) -> tensor<32x2112xbf16>
// CHECK-NOT: "ttir.mesh_partition"
// CHECK-NOT: stablehlo.composite

// -----

// Same out_sharding, but this time the operand is fully replicated, so both
// axes are outstanding and are applied major -> minor.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @replicated_operand_multi_axis(%arg0: tensor<256x2112xbf16>) -> tensor<32x2112xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0", "_axis_1"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<256x2112xbf16>) -> tensor<32x2112xbf16>
    return %0 : tensor<32x2112xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<256x2112xbf16>) -> tensor<32x2112xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x2112xbf16>
    return %0 : tensor<32x2112xbf16>
  }
}

// CHECK-LABEL: func.func @replicated_operand_multi_axis
// CHECK: %[[MAJOR:[0-9]+]] = "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 0 : ui32, dim = 0 : si32}> : (tensor<256x2112xbf16>) -> tensor<128x2112xbf16>
// CHECK: "ttir.mesh_partition"(%[[MAJOR]])
// CHECK-SAME: <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<128x2112xbf16>) -> tensor<32x2112xbf16>
// CHECK-NOT: stablehlo.composite

// -----

// The operand already carries all of out_sharding, so the all_slice is a no-op
// and is folded away entirely.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @fully_sharded_operand(%arg0: tensor<32x2112xbf16>) -> tensor<32x2112xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0", "_axis_1"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<32x2112xbf16>) -> tensor<32x2112xbf16>
    return %0 : tensor<32x2112xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<32x2112xbf16>) -> tensor<32x2112xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x2112xbf16>
    return %0 : tensor<32x2112xbf16>
  }
}

// CHECK-LABEL: func.func @fully_sharded_operand(
// CHECK-SAME: %[[ARG:[a-z0-9]+]]: tensor<32x2112xbf16>
// CHECK-NEXT: return %[[ARG]]
// CHECK-NOT: "ttir.mesh_partition"
// CHECK-NOT: stablehlo.composite

// -----

// Two dims sharded over one axis each, neither applied yet.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @two_dims_outstanding(%arg0: tensor<4x32xbf16>) -> tensor<1x16xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {"_axis_0"}]>}, decomposition = @sdy.all_slice1} : (tensor<4x32xbf16>) -> tensor<1x16xbf16>
    return %0 : tensor<1x16xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<4x32xbf16>) -> tensor<1x16xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x16xbf16>
    return %0 : tensor<1x16xbf16>
  }
}

// CHECK-LABEL: func.func @two_dims_outstanding
// CHECK: %[[DIM0:[0-9]+]] = "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<4x32xbf16>) -> tensor<1x32xbf16>
// CHECK: "ttir.mesh_partition"(%[[DIM0]])
// CHECK-SAME: <{cluster_axis = 0 : ui32, dim = 1 : si32}> : (tensor<1x32xbf16>) -> tensor<1x16xbf16>
// CHECK-NOT: stablehlo.composite

// -----

// Same sharding as above, but dim0 is already sliced: only dim1 is left. A
// per-tensor-dim delta is needed here, not a single global one.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @one_of_two_dims_outstanding(%arg0: tensor<1x32xbf16>) -> tensor<1x16xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {"_axis_0"}]>}, decomposition = @sdy.all_slice1} : (tensor<1x32xbf16>) -> tensor<1x16xbf16>
    return %0 : tensor<1x16xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<1x32xbf16>) -> tensor<1x16xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x16xbf16>
    return %0 : tensor<1x16xbf16>
  }
}

// CHECK-LABEL: func.func @one_of_two_dims_outstanding
// CHECK: "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 0 : ui32, dim = 1 : si32}> : (tensor<1x32xbf16>) -> tensor<1x16xbf16>
// CHECK-NOT: "ttir.mesh_partition"
// CHECK-NOT: stablehlo.composite

// -----

// A dim's axes are listed major -> minor, which is not necessarily mesh order.
// "_axis_1" is the major axis here, so it has to be sliced first, otherwise
// each device ends up with a different shard than out_sharding asks for.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @axes_in_reverse_mesh_order(%arg0: tensor<256x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_1", "_axis_0"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<256x8xbf16>) -> tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<256x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
}

// CHECK-LABEL: func.func @axes_in_reverse_mesh_order
// CHECK: %[[MAJOR:[0-9]+]] = "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<256x8xbf16>) -> tensor<64x8xbf16>
// CHECK: "ttir.mesh_partition"(%[[MAJOR]])
// CHECK-SAME: <{cluster_axis = 0 : ui32, dim = 0 : si32}> : (tensor<64x8xbf16>) -> tensor<32x8xbf16>
// CHECK-NOT: stablehlo.composite

// -----

// The same dim is sharded over a unit-sized axis too; slicing by 1 is a no-op
// so no mesh_partition is emitted for it.
module {
  sdy.mesh @mesh = <["x"=1, "y"=2]>
  func.func @unit_sized_axis(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<64x8xbf16>) -> tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
}

// CHECK-LABEL: func.func @unit_sized_axis
// CHECK: "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<64x8xbf16>) -> tensor<32x8xbf16>
// CHECK-NOT: "ttir.mesh_partition"
// CHECK-NOT: stablehlo.composite

// -----

// Back-to-back all_slices: the second one sees the type the first one produced,
// so its own delta is just the axis it adds.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @chained_all_slice(%arg0: tensor<256x2112xbf16>) -> tensor<32x2112xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<256x2112xbf16>) -> tensor<128x2112xbf16>
    %1 = stablehlo.composite "sdy.all_slice" %0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0", "_axis_1"}, {}]>}, decomposition = @sdy.all_slice2} : (tensor<128x2112xbf16>) -> tensor<32x2112xbf16>
    return %1 : tensor<32x2112xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<256x2112xbf16>) -> tensor<128x2112xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<128x2112xbf16>
    return %0 : tensor<128x2112xbf16>
  }
  func.func private @sdy.all_slice2(%arg0: tensor<128x2112xbf16>) -> tensor<32x2112xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x2112xbf16>
    return %0 : tensor<32x2112xbf16>
  }
}

// CHECK-LABEL: func.func @chained_all_slice
// CHECK: %[[FIRST:[0-9]+]] = "ttir.mesh_partition"
// CHECK-SAME: <{cluster_axis = 0 : ui32, dim = 0 : si32}> : (tensor<256x2112xbf16>) -> tensor<128x2112xbf16>
// CHECK: "ttir.mesh_partition"(%[[FIRST]])
// CHECK-SAME: <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<128x2112xbf16>) -> tensor<32x2112xbf16>
// CHECK-NOT: stablehlo.composite

// -----

// The result is not an integral shard of the operand, so there is no chain of
// mesh_partitions that lands on the declared result type.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @non_integral_shard(%arg0: tensor<100x8xbf16>) -> tensor<33x8xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<100x8xbf16>) -> tensor<33x8xbf16>
    return %0 : tensor<33x8xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<100x8xbf16>) -> tensor<33x8xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<33x8xbf16>
    return %0 : tensor<33x8xbf16>
  }
}

// CHECK-LABEL: func.func @non_integral_shard
// CHECK: stablehlo.composite "sdy.all_slice"
// CHECK-NOT: "ttir.mesh_partition"

// -----

// out_sharding cannot explain the operand -> result divisor: dim0 shrinks by 2,
// but the only outstanding-axis suffixes of {"_axis_0", "_axis_1"} divide it by
// 1, 4 or 8. Bail out instead of guessing a subset of the axes, which would
// give a right-shaped result holding the wrong shard.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @irreconcilable_divisor(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0", "_axis_1"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<64x8xbf16>) -> tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
}

// CHECK-LABEL: func.func @irreconcilable_divisor
// CHECK: stablehlo.composite "sdy.all_slice"
// CHECK-NOT: "ttir.mesh_partition"

// -----

// A dim shrinks without out_sharding naming any axis for it.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @unsharded_dim_shrinks(%arg0: tensor<64x8xbf16>) -> tensor<32x4xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<64x8xbf16>) -> tensor<32x4xbf16>
    return %0 : tensor<32x4xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<64x8xbf16>) -> tensor<32x4xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x4xbf16>
    return %0 : tensor<32x4xbf16>
  }
}

// CHECK-LABEL: func.func @unsharded_dim_shrinks
// CHECK: stablehlo.composite "sdy.all_slice"
// CHECK-NOT: "ttir.mesh_partition"

// -----

// Ranks have to line up before any of the per-dim reasoning is meaningful.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @rank_mismatch(%arg0: tensor<64x8xbf16>) -> tensor<32xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<64x8xbf16>) -> tensor<32xbf16>
    return %0 : tensor<32xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<64x8xbf16>) -> tensor<32xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32xbf16>
    return %0 : tensor<32xbf16>
  }
}

// CHECK-LABEL: func.func @rank_mismatch
// CHECK: stablehlo.composite "sdy.all_slice"
// CHECK-NOT: "ttir.mesh_partition"

// -----

// out_sharding's rank has to match the tensor's.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @sharding_rank_mismatch(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0"}]>}, decomposition = @sdy.all_slice1} : (tensor<64x8xbf16>) -> tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
}

// CHECK-LABEL: func.func @sharding_rank_mismatch
// CHECK: stablehlo.composite "sdy.all_slice"
// CHECK-NOT: "ttir.mesh_partition"

// -----

// An open dim sharding may still be sharded further by Shardy, so the axis list
// is not the whole story yet.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @open_dim_sharding(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_0", ?}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<64x8xbf16>) -> tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
}

// CHECK-LABEL: func.func @open_dim_sharding
// CHECK: stablehlo.composite "sdy.all_slice"
// CHECK-NOT: "ttir.mesh_partition"

// -----

// Sub-axis partitioning has no single cluster axis to slice along.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @sub_axis_sharding(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@mesh, [{"_axis_1":(1)2}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<64x8xbf16>) -> tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
}

// CHECK-LABEL: func.func @sub_axis_sharding
// CHECK: stablehlo.composite "sdy.all_slice"
// CHECK-NOT: "ttir.mesh_partition"

// -----

// The mesh the sharding names is not in the module.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @unknown_mesh(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = #sdy.sharding<@other_mesh, [{"_axis_0"}, {}]>}, decomposition = @sdy.all_slice1} : (tensor<64x8xbf16>) -> tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
}

// CHECK-LABEL: func.func @unknown_mesh
// CHECK: stablehlo.composite "sdy.all_slice"
// CHECK-NOT: "ttir.mesh_partition"

// -----

// out_sharding is missing altogether.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @missing_out_sharding(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {decomposition = @sdy.all_slice1} : (tensor<64x8xbf16>) -> tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
}

// CHECK-LABEL: func.func @missing_out_sharding
// CHECK: stablehlo.composite "sdy.all_slice"
// CHECK-NOT: "ttir.mesh_partition"

// -----

// out_sharding is present but is not a shardy tensor sharding.
module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @out_sharding_wrong_type(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.composite "sdy.all_slice" %arg0 {composite_attributes = {out_sharding = 1 : i64}, decomposition = @sdy.all_slice1} : (tensor<64x8xbf16>) -> tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
  func.func private @sdy.all_slice1(%arg0: tensor<64x8xbf16>) -> tensor<32x8xbf16> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<32x8xbf16>
    return %0 : tensor<32x8xbf16>
  }
}

// CHECK-LABEL: func.func @out_sharding_wrong_type
// CHECK: stablehlo.composite "sdy.all_slice"
// CHECK-NOT: "ttir.mesh_partition"
