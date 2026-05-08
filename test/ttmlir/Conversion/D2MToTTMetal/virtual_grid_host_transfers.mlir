// RUN: ttmlir-opt --ttcore-register-device="mock-system-desc-arch=blackhole" --ttir-to-ttmetal-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module {
  // CHECK-LABEL: func.func @slice_f32_virtual_grid_host_transfers
  // CHECK: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64, virtualGridForwardMapping = #map{{[0-9]*}}, virtualGridInverseMapping = #map{{[0-9]*}}}> : () -> memref<3x16x2x1x2x4x32x32xf32, #ttcore.shard<16384x4096x128x4, 1>, #l1>
  // CHECK: "ttmetal.enqueue_write_buffer"{{.*}}memref<3x16x2x1x2x4x32x32xf32, #ttcore.shard<16384x4096x128x4, 1>, #l1>
  // CHECK: "ttmetal.enqueue_read_buffer"{{.*}}memref<3x32x1x1x1x1x32x32xf32, #ttcore.shard<4096x4096x128x4, 1>, #l1>
  func.func @slice_f32_virtual_grid_host_transfers(%arg0: tensor<6x64x64x4xf32>) -> tensor<3x32x32x2xf32> {
    %0 = "ttir.slice_static"(%arg0) <{
      begins = [1 : i32, 15 : i32, 16 : i32, 2 : i32],
      ends = [6 : i32, 47 : i32, 48 : i32, 4 : i32],
      step = [2 : i32, 1 : i32, 1 : i32, 1 : i32]
    }> : (tensor<6x64x64x4xf32>) -> tensor<3x32x32x2xf32>
    return %0 : tensor<3x32x32x2xf32>
  }
}
