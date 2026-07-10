// RUN: ttmlir-opt --split-input-file "--d2m-fe-pipeline=override-device-shape=1,1" --d2m-be-pipeline --convert-d2m-to-ttkernel %s | FileCheck %s

// Verify that the D2MToTTKernel conversion emits the correct TTKernel ops for
// topk. Using override-device-shape=1,1 forces all tiles onto a single core so
// the full sort-merge-rebuild sequence is visible in one compute kernel.

// ---- dim=1, k=16, 2-tile input ----

// 32x64 input: 1 merge iteration (logWt=1), small k, rebuild is emitted.
// CHECK-LABEL: func.func @topk_dim1_k16
func.func @topk_dim1_k16(%arg0: tensor<32x64xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>) {
  // The arange kernel initializes the index CB with sequential tile indices.
  // CHECK: ttkernel.experimental.fill_arange_tile

  // The sort-merge-rebuild group processes tile pair (0, 1).
  // CHECK: ttkernel.topk_tile_init
  // CHECK: ttkernel.topk_local_sort
  // CHECK: ttkernel.topk_merge
  // CHECK: ttkernel.topk_rebuild

  %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x64xf32>) -> (tensor<32x16xf32>, tensor<32x16xsi32>)
  return %values, %indices : tensor<32x16xf32>, tensor<32x16xsi32>
}

// -----

// ---- dim=1, k=32, 2-tile input: no rebuild ----

// 32x64 with k=32: the merge output is exactly k elements, so no rebuild is emitted.
// CHECK-LABEL: func.func @topk_k32_no_rebuild
func.func @topk_k32_no_rebuild(%arg0: tensor<32x64xf32>) -> (tensor<32x32xf32>, tensor<32x32xsi32>) {
  // CHECK: ttkernel.experimental.fill_arange_tile
  // CHECK: ttkernel.topk_local_sort
  // CHECK: ttkernel.topk_merge
  // CHECK-NOT: ttkernel.topk_rebuild
  %values, %indices = "ttir.topk"(%arg0) <{k = 32 : i32, dim = -1 : i32, largest = true, sorted = false}> : (tensor<32x64xf32>) -> (tensor<32x32xf32>, tensor<32x32xsi32>)
  return %values, %indices : tensor<32x32xf32>, tensor<32x32xsi32>
}

// -----

// ---- dim=0, k=16, 2-tile input ----

// 64x32 with dim=0: no pre-transpose is needed; the TTKernel ops are the same as dim=1.
// CHECK-LABEL: func.func @topk_dim0_k16
func.func @topk_dim0_k16(%arg0: tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>) {
  // CHECK: ttkernel.experimental.fill_arange_tile
  // CHECK: ttkernel.topk_local_sort
  // CHECK: ttkernel.topk_merge
  // CHECK: ttkernel.topk_rebuild
  %values, %indices = "ttir.topk"(%arg0) <{k = 16 : i32, dim = 0 : i32, largest = true, sorted = false}> : (tensor<64x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xsi32>)
  return %values, %indices : tensor<16x32xf32>, tensor<16x32xsi32>
}
