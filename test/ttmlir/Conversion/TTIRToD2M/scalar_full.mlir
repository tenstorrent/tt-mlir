// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline -o %t.ttmetal %s

module {
  // CHECK-LABEL: func.func @scalar_full
  func.func @scalar_full() -> tensor<f32> {
    // CHECK: d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>
    // CHECK: d2m.generic
    // CHECK: d2m.tile_fill
    %0 = "ttir.full"() <{shape = array<i32>, fill_value = 1.000000e+00 : f32}> : () -> tensor<f32>
    return %0 : tensor<f32>
  }
}
