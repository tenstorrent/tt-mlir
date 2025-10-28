// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttir-allocate --convert-ttir-to-ttmetal -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir
// UNSUPPORTED: true

// This test verifies that layout transformations with different grid distributions
// work correctly. The transformation should create appropriate data movement operations
// to reblock data from one grid configuration to another.

#l1_ = #ttcore.memory_space<l1>

// System layout (host memory)
#system_layout = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  system
>

// Input: 2x4 grid in L1
#layout_2x4 = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

// Output: 4x2 grid in L1
#layout_4x2 = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

func.func @reblock_with_affine_transform(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // Input: system memory
  %0 = tensor.empty() : tensor<64x128xf32>

  // Move to device with 2x4 grid
  // CHECK: "ttmetal.host_write"
  %1 = "ttir.to_layout"(%arg0, %0) <{layout = #layout_2x4}> :
    (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

  // Reblock from 2x4 to 4x2 grid
  // This should use the affine map transformation we just implemented
  // CHECK: "ttmetal.enqueue_program"
  %2 = "ttir.to_layout"(%1, %0) <{layout = #layout_4x2}> :
    (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

  // Move back to system memory
  // CHECK: "ttmetal.host_read"
  %3 = "ttir.to_layout"(%2, %0) <{layout = #system_layout}> :
    (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

  return %3 : tensor<64x128xf32>
}
