// RUN: ttmlir-opt --mlir-disable-threading --mlir-print-ir-after=d2m-split-unified-thread --ttcore-register-device --ttir-to-ttmetal-pipeline="default-input-memspace=dram default-output-memspace=dram" %s -o /dev/null 2>&1 | FileCheck %s

// Regression test for keeping non-compute scaler loads in the synchronized
// range. Without the wait/pop for get_cb(4), the DMA thread repeatedly pushes
// the mean scaler tile and can block on a full CB at runtime.

module {
  func.func @main(%arg0: tensor<2x128x160xf32>) -> tensor<2x128x1xf32> {
    %0 = "ttir.mean"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<2x128x160xf32>) -> tensor<2x128x1xf32>
    return %0 : tensor<2x128x1xf32>
  }
}

// CHECK: d2m.generic {{.*}}grid = #ttcore.grid<2x4x1
// CHECK: }, {
// CHECK: %[[SCALER_CB:.*]] = d2m.get_cb(4)
// CHECK: scf.for
// CHECK: %[[SCALER_WAIT:.*]] = d2m.wait %[[SCALER_CB]]
// CHECK: %[[SCALER_VIEW:.*]] = memref.collapse_shape %[[SCALER_WAIT]]
// CHECK: %[[SCALER:.*]] = memref.load %[[SCALER_VIEW]]
// CHECK: "d2m.tile_reduce_mean"({{.*}}, %[[SCALER]], {{.*}})
// CHECK: d2m.pop %[[SCALER_CB]]
