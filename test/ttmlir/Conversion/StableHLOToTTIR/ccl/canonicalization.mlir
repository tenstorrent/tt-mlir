// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --shardy-ccl-canonicalization %s -o %t
// RUN: FileCheck %s --input-file=%t

// Test that sdy.all_reduce + sdy.all_slice with matching axes fuses into sdy.reduce_scatter
// CHECK-LABEL: @FusingTest
// CHECK: sdy.reduce_scatter
// CHECK-NOT: sdy.all_slice
// CHECK-NOT: sdy.all_reduce
module @FusingTest {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<32x512xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> tensor<32x512xbf16> {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {"_axis_0"}]>] manual_axes={} (%arg1: tensor<32x512xbf16>) {
      // all_reduce on _axis_0 followed by all_slice on _axis_0 -> should fuse to reduce_scatter
      %1 = sdy.all_reduce {"_axis_0"} %arg1 out_sharding=<@mesh, [{}, {}]> : tensor<32x512xbf16>
      %2 = sdy.all_slice [{}, {"_axis_0"}] %1 out_sharding=<@mesh, [{}, {"_axis_0"}]> : tensor<32x512xbf16>
      sdy.return %2 : tensor<32x512xbf16>
    } : (tensor<32x512xbf16>) -> tensor<32x512xbf16>
    return %0 : tensor<32x512xbf16>
  }
}

// -----

// Test that sdy.all_reduce + sdy.all_slice with NON-matching axes does NOT fuse
// CHECK-LABEL: @NonFusingTest
// CHECK: sdy.all_reduce {"_axis_0"}
// CHECK: sdy.all_slice [{}, {"_axis_1"}]
// CHECK-NOT: sdy.reduce_scatter
module @NonFusingTest attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<32x512xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> tensor<32x512xbf16> {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {"_axis_1"}]>] manual_axes={} (%arg1: tensor<32x512xbf16>) {
      // all_reduce on _axis_0 followed by all_slice on _axis_1 -> should NOT fuse (different axes)
      %1 = sdy.all_reduce {"_axis_0"} %arg1 out_sharding=<@mesh, [{}, {}]> : tensor<32x512xbf16>
      %2 = sdy.all_slice [{}, {"_axis_1"}] %1 out_sharding=<@mesh, [{}, {"_axis_1"}]> : tensor<32x512xbf16>
      sdy.return %2 : tensor<32x512xbf16>
    } : (tensor<32x512xbf16>) -> tensor<32x512xbf16>
    return %0 : tensor<32x512xbf16>
  }
}

// -----

// Test that sdy.all_reduce with multiple all_slice users (all matching axes) fuses into multiple reduce_scatters
// CHECK-LABEL: @MultiUseFusingTest
// CHECK: sdy.reduce_scatter
// CHECK: sdy.reduce_scatter
// CHECK-NOT: sdy.all_slice
// CHECK-NOT: sdy.all_reduce
module @MultiUseFusingTest {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @main(%arg0: tensor<32x512xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<32x512xbf16>, tensor<32x512xbf16>) {
    %0:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {"_axis_0"}]>, <@mesh, [{}, {"_axis_0"}]>] manual_axes={} (%arg1: tensor<32x512xbf16>) {
      // all_reduce on _axis_0 followed by two all_slice ops on _axis_0 -> both should fuse to reduce_scatter
      %1 = sdy.all_reduce {"_axis_0"} %arg1 out_sharding=<@mesh, [{}, {}]> : tensor<32x512xbf16>
      %2 = sdy.all_slice [{}, {"_axis_0"}] %1 out_sharding=<@mesh, [{}, {"_axis_0"}]> : tensor<32x512xbf16>
      %3 = sdy.all_slice [{}, {"_axis_0"}] %1 out_sharding=<@mesh, [{}, {"_axis_0"}]> : tensor<32x512xbf16>
      sdy.return %2, %3 : tensor<32x512xbf16>, tensor<32x512xbf16>
    } : (tensor<32x512xbf16>) -> (tensor<32x512xbf16>, tensor<32x512xbf16>)
    return %0#0, %0#1 : tensor<32x512xbf16>, tensor<32x512xbf16>
  }
}
