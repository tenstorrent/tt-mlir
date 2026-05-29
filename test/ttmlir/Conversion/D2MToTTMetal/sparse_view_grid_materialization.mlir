// RUN: ttmlir-opt --ttcore-register-device="mock-system-desc-arch=blackhole" --ttir-to-ttmetal-pipeline="test-assume-l1-capacity=8388608" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module {
  // CHECK-LABEL: func.func @slice_1d_strided_sparse_projection
  // CHECK-NOT: memref<{{[^>]*}}224{{[^>]*}}xf32
  // CHECK: memref<1x3x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
  // CHECK: "ttmetal.enqueue_program"{{.*}}#ttmetal.core_range<0x0, 1x1>{{.*}} : (memref<1x3x32x32xf32
  // CHECK-NOT: memref<{{[^>]*}}224{{[^>]*}}xf32
  func.func @slice_1d_strided_sparse_projection(%arg0: tensor<70xf32>) -> tensor<9xf32> {
    %0 = "ttir.slice_static"(%arg0) <{
      begins = [3 : i32],
      ends = [62 : i32],
      step = [7 : i32]
    }> : (tensor<70xf32>) -> tensor<9xf32>
    return %0 : tensor<9xf32>
  }
}
