// RUN: ttmlir-opt --split-input-file "--d2m-fe-pipeline=override-device-shape=1,1" --d2m-be-pipeline --convert-d2m-to-ttkernel %s | FileCheck %s

// Verify that the D2MToTTKernel conversion emits the correct TTKernel ops for
// sort. Using override-device-shape=1,1 forces all tiles onto a single core so
// the whole bitonic network is visible in one compute kernel.

// ---- dim=1, 2-tile input ----

// 32x64: Wt=2, so the pairwise local sort alone sorts the row and the merge
// network is empty.
// CHECK-LABEL: func.func @sort_dim1_2tiles
func.func @sort_dim1_2tiles(%arg0: tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>) {
  // The arange kernel initializes the index CB with sequential tile indices.
  // CHECK: ttkernel.experimental.fill_arange_tile

  // CHECK: ttkernel.topk_tile_init
  // CHECK: ttkernel.topk_local_sort

  // A sort never rebuilds: topk_merge preserves both halves.
  // CHECK-NOT: ttkernel.topk_rebuild

  %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = false, stable = false}> : (tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x64xsi32>)
  return %values, %indices : tensor<32x64xf32>, tensor<32x64xsi32>
}

// -----

// ---- dim=1, 8-tile input: the merge network runs ----

// CHECK-LABEL: func.func @sort_dim1_8tiles
func.func @sort_dim1_8tiles(%arg0: tensor<32x256xf32>) -> (tensor<32x256xf32>, tensor<32x256xsi32>) {
  // CHECK: ttkernel.experimental.fill_arange_tile
  // CHECK: ttkernel.topk_tile_init
  // CHECK: ttkernel.topk_local_sort
  // CHECK: ttkernel.topk_merge
  // CHECK-NOT: ttkernel.topk_rebuild
  %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = false, stable = false}> : (tensor<32x256xf32>) -> (tensor<32x256xf32>, tensor<32x256xsi32>)
  return %values, %indices : tensor<32x256xf32>, tensor<32x256xsi32>
}

// -----

// ---- descending ----

// CHECK-LABEL: func.func @sort_dim1_descending
func.func @sort_dim1_descending(%arg0: tensor<32x256xf32>) -> (tensor<32x256xf32>, tensor<32x256xsi32>) {
  // CHECK: ttkernel.topk_local_sort
  // CHECK: ttkernel.topk_merge
  // CHECK-NOT: ttkernel.topk_rebuild
  %values, %indices = "ttir.sort"(%arg0) <{dim = -1 : si32, descending = true, stable = false}> : (tensor<32x256xf32>) -> (tensor<32x256xf32>, tensor<32x256xsi32>)
  return %values, %indices : tensor<32x256xf32>, tensor<32x256xsi32>
}

// -----

// ---- dim=0: no value pre-transpose, same TTKernel ops ----

// CHECK-LABEL: func.func @sort_dim0_8tiles
func.func @sort_dim0_8tiles(%arg0: tensor<256x32xf32>) -> (tensor<256x32xf32>, tensor<256x32xsi32>) {
  // CHECK: ttkernel.experimental.fill_arange_tile
  // CHECK: ttkernel.topk_local_sort
  // CHECK: ttkernel.topk_merge
  // CHECK-NOT: ttkernel.topk_rebuild
  %values, %indices = "ttir.sort"(%arg0) <{dim = 0 : si32, descending = false, stable = false}> : (tensor<256x32xf32>) -> (tensor<256x32xf32>, tensor<256x32xsi32>)
  return %values, %indices : tensor<256x32xf32>, tensor<256x32xsi32>
}
