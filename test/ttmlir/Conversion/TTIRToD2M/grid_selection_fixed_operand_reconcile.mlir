// RUN: ttmlir-opt --ttcore-register-device="mock-system-desc-arch=blackhole" --ttir-to-d2m --d2m-materialize-view-returns --d2m-grid-selection --canonicalize %s | FileCheck %s

module {
  // A fixed masked input can have a smaller physical shape than a rewritable
  // output would choose independently. Grid selection must reconcile the final
  // materialized shapes in loop space using the grids chosen up front; the
  // fixed mask remains at its current grid while the consumer uses the selected
  // virtual grid without an allocator-driven retry.
  // CHECK-LABEL: func.func @reduce_i32_unaligned_fixed_mask_reconcile
  // CHECK: d2m.mask
  // CHECK-SAME: tensor<1x1x1x16x!ttcore.tile<32x32, si32>
  // CHECK: d2m.empty() {virtualGridForwardMapping = {{.*}}, virtualGridInverseMapping = {{.*}}} : tensor<1x16x1x1x!ttcore.tile<32x32, si32>
  // CHECK: d2m.view_layout
  // CHECK-SAME: -> tensor<1x16x1x1x!ttcore.tile<32x32, si32>
  // CHECK: d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<1x16, virt_to_physical_map = (d0, d1) ->
  func.func @reduce_i32_unaligned_fixed_mask_reconcile(%arg0: tensor<1x501xi32>) -> tensor<1x501xi32> {
    %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<1x501xi32>) -> tensor<1x501xi32>
    return %0 : tensor<1x501xi32>
  }
}
