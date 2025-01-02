// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s
module attributes{} {
  func.func @add(
    %arg0: tensor<32x32xf32>,
    %arg1: tensor<32x32xf32>,
    %arg2: tensor<32x32xf32>
  ) -> tensor<32x32xf32> {
    %1 = "ttir.add"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: {{%[0-9]+}} = linalg.add ins(%arg{{[0-9]+}}, %arg{{[0-9]+}} : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg{{[0-9]+}} : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }

  func.func @add_with_broadcast(
    %arg0: tensor<32x32xf32>,
    %arg1: tensor<32x1xf32>,
    %arg2: tensor<32x32xf32>
  ) -> tensor<32x32xf32> {
    %1 = "ttir.add"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xf32>, tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: %{{.+}} = tensor.collapse_shape
    // CHECK: %{{.+}} = tensor.empty()
    // CHECK: %{{.+}} = linalg.broadcast ins(%{{.+}} : tensor<{{.*xf32}}>) outs(%{{.+}} : tensor<{{.*xf32}}>) dimensions
    // CHECK: %{{.+}} = linalg.add ins(%{{.+}}, %{{.+}} : tensor<{{.*xf32}}>, tensor<{{.*xf32}}>) outs(%{{.+}} : tensor<{{.*xf32}}>) -> tensor<{{.*xf32}}>

    return %1 : tensor<32x32xf32>
  }

  func.func @add_with_broadcast_1(
    %arg0: tensor<32x1xf32>,
    %arg1: tensor<32x32x32xf32>,
    %arg2: tensor<32x32x32xf32>
  ) -> tensor<32x32x32xf32> {
    %1 = "ttir.add"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x1xf32>, tensor<32x32x32xf32>, tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
    // CHECK: %{{.+}} = tensor.collapse_shape
    // CHECK: %{{.+}} = tensor.empty()
    // CHECK: %{{.+}} = linalg.broadcast ins(%{{.+}} : tensor<{{.*xf32}}>) outs(%{{.+}} : tensor<{{.*xf32}}>) dimensions =
    // CHECK: %{{.+}} = linalg.add ins(%{{.+}}, %{{.+}} : tensor<{{.*xf32}}>, tensor<{{.*xf32}}>) outs(%{{.+}} : tensor<{{.*xf32}}>) -> tensor<{{.*xf32}}>
    return %1 : tensor<32x32x32xf32>
  }

  func.func @add_with_broadcast_2(
    %arg0: tensor<32x1x32xf32>,
    %arg1: tensor<32x1x1xf32>,
    %arg2: tensor<32x1x32xf32>
  ) -> tensor<32x1x32xf32> {
    %1 = "ttir.add"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x1x32xf32>, tensor<32x1x1xf32>, tensor<32x1x32xf32>) -> tensor<32x1x32xf32>
    // CHECK: %{{.+}} = tensor.collapse_shape
    // CHECK: %{{.+}} = tensor.empty()
    // CHECK: %{{.+}} = linalg.broadcast ins(%{{.+}} : tensor<{{.*xf32}}>) outs(%{{.+}} : tensor<{{.*xf32}}>) dimensions
    // CHECK: %{{.+}} = linalg.add ins(%{{.+}}, %{{.+}} : tensor<{{.*xf32}}>, tensor<{{.*xf32}}>) outs(%{{.+}} : tensor<{{.*xf32}}>) -> tensor<{{.*xf32}}>
    return %1 : tensor<32x1x32xf32>
  }

  func.func @add_with_broadcast_3(
    %arg0: tensor<32xf32>,
    %arg1: tensor<32x32xf32>,
    %arg2: tensor<32x32xf32>
  ) -> tensor<32x32xf32> {
    %1 = "ttir.add"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: %{{.+}} = tensor.empty()
    // CHECK: %{{.+}} = linalg.broadcast ins(%{{.+}} : tensor<{{.*xf32}}>) outs(%{{.+}} : tensor<{{.*xf32}}>) dimensions
    // CHECK: %{{.+}} = linalg.add ins(%{{.+}}, %{{.+}} : tensor<{{.*xf32}}>, tensor<{{.*xf32}}>) outs(%{{.+}} : tensor<{{.*xf32}}>) -> tensor<{{.*xf32}}>
    return %1 : tensor<32x32xf32>
  }
}
