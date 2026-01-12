// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true d2m-fallback-enabled=true tensor-l1-usage-cap=0.001" %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir


// with tensor-l1-usage-cap=0.001, this operation should fail
// validation due to OOM but with d2m fallback enabled, it gets marked
// and compiled through D2M down to a ttnn.generic operation

module attributes {"ttnn.tensor_l1_usage_cap" = 0.001 : f32} {
  // CHECK-LABEL: func.func @add_with_l1_oom
  func.func @add_with_l1_oom(%arg0: tensor<512x512xbf16>, %arg1: tensor<512x512xbf16>) -> tensor<512x512xbf16> {

    // CHECK: "ttnn.generic"
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<512x512xbf16>, tensor<512x512xbf16>) -> tensor<512x512xbf16>

    return %0 : tensor<512x512xbf16>
  }
}
