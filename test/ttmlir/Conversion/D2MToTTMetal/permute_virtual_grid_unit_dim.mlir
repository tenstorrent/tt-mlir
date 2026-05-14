// RUN: ttmlir-opt --ttcore-register-device="mock-system-desc-arch=blackhole" --ttir-to-ttmetal-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module {
  // CHECK-LABEL: func.func @permute_0
  // CHECK: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64, virtualGridForwardMapping = #map{{[0-9]*}}, virtualGridInverseMapping = #map{{[0-9]*}}}> : () -> memref<1x32x1x2x1x4x32x32xbf16
  // CHECK: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64, virtualGridForwardMapping = #map{{[0-9]*}}, virtualGridInverseMapping = #map{{[0-9]*}}}> : () -> memref<1x1x4x2x1x1x32x32xbf16
  // CHECK: #ttmetal.core_range<0x0, 2x4>
  // CHECK: "ttmetal.enqueue_read_buffer"{{.*}} : (memref<1x1x4x2x1x1x32x32xbf16
  func.func @permute_0(%arg0: tensor<1x128x1x64xbf16>) -> tensor<1x1x128x64xbf16> {
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x1x64xbf16>) -> tensor<1x1x128x64xbf16>
    return %0 : tensor<1x1x128x64xbf16>
  }
}
