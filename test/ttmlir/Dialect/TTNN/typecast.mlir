// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func public @main(%arg0: tensor<32x32xui32>) -> tensor<32x32xui16> {
    %0 = tensor.empty() : tensor<32x32xui16>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xui32>, tensor<32x32xui16>) -> tensor<32x32xui16>
    return %1 : tensor<32x32xui16>
  }
  func.func public @main1(%arg0: tensor<32x32xui32>) -> tensor<32x32xui16> {
    %0 = tensor.empty() : tensor<32x32xui16>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xui32>, tensor<32x32xui16>) -> tensor<32x32xui16>
    return %1 : tensor<32x32xui16>
  }
  func.func public @main2(%arg0: tensor<32x32xi16>) -> tensor<32x32xui16> {
    %0 = tensor.empty() : tensor<32x32xui16>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xi16>, tensor<32x32xui16>) -> tensor<32x32xui16>
    return %1 : tensor<32x32xui16>
  }
  func.func public @main3(%arg0: tensor<32x32xi32>) -> tensor<32x32xui16> {
    %0 = tensor.empty() : tensor<32x32xui16>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xi32>, tensor<32x32xui16>) -> tensor<32x32xui16>
    return %1 : tensor<32x32xui16>
  }
  func.func public @main4(%arg0: tensor<32x32xi32>) -> tensor<32x32xui16> {
    %0 = tensor.empty() : tensor<32x32xui16>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xi32>, tensor<32x32xui16>) -> tensor<32x32xui16>
    return %1 : tensor<32x32xui16>
  }
  func.func public @main5(%arg0: tensor<32x32xf32>) -> tensor<32x32xui16> {
    %0 = tensor.empty() : tensor<32x32xui16>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xf32>, tensor<32x32xui16>) -> tensor<32x32xui16>
    return %1 : tensor<32x32xui16>
  }
  func.func public @main6(%arg0: tensor<32x32xbf16>) -> tensor<32x32xui16> {
    %0 = tensor.empty() : tensor<32x32xui16>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xbf16>, tensor<32x32xui16>) -> tensor<32x32xui16>
    return %1 : tensor<32x32xui16>
  }
  func.func public @main7(%arg0: tensor<32x32xui16>) -> tensor<32x32xui32> {
    %0 = tensor.empty() : tensor<32x32xui32>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xui16>, tensor<32x32xui32>) -> tensor<32x32xui32>
    return %1 : tensor<32x32xui32>
  }
  func.func public @main8(%arg0: tensor<32x32xi16>) -> tensor<32x32xui32> {
    %0 = tensor.empty() : tensor<32x32xui32>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xi16>, tensor<32x32xui32>) -> tensor<32x32xui32>
    return %1 : tensor<32x32xui32>
  }
  func.func public @main9(%arg0: tensor<32x32xi32>) -> tensor<32x32xui32> {
    %0 = tensor.empty() : tensor<32x32xui32>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xi32>, tensor<32x32xui32>) -> tensor<32x32xui32>
    return %1 : tensor<32x32xui32>
  }
  func.func public @main10(%arg0: tensor<32x32xi32>) -> tensor<32x32xui32> {
    %0 = tensor.empty() : tensor<32x32xui32>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xi32>, tensor<32x32xui32>) -> tensor<32x32xui32>
    return %1 : tensor<32x32xui32>
  }
  func.func public @main11(%arg0: tensor<32x32xf32>) -> tensor<32x32xui32> {
    %0 = tensor.empty() : tensor<32x32xui32>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xf32>, tensor<32x32xui32>) -> tensor<32x32xui32>
    return %1 : tensor<32x32xui32>
  }
  func.func public @main12(%arg0: tensor<32x32xbf16>) -> tensor<32x32xui32> {
    %0 = tensor.empty() : tensor<32x32xui32>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xbf16>, tensor<32x32xui32>) -> tensor<32x32xui32>
    return %1 : tensor<32x32xui32>
  }
  func.func public @main13(%arg0: tensor<32x32xui16>) -> tensor<32x32xui32> {
    %0 = tensor.empty() : tensor<32x32xui32>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xui16>, tensor<32x32xui32>) -> tensor<32x32xui32>
    return %1 : tensor<32x32xui32>
  }
  func.func public @main14(%arg0: tensor<32x32xi16>) -> tensor<32x32xui32> {
    %0 = tensor.empty() : tensor<32x32xui32>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xi16>, tensor<32x32xui32>) -> tensor<32x32xui32>
    return %1 : tensor<32x32xui32>
  }
  func.func public @main15(%arg0: tensor<32x32xi32>) -> tensor<32x32xui32> {
    %0 = tensor.empty() : tensor<32x32xui32>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xi32>, tensor<32x32xui32>) -> tensor<32x32xui32>
    return %1 : tensor<32x32xui32>
  }
  func.func public @main16(%arg0: tensor<32x32xi32>) -> tensor<32x32xui32> {
    %0 = tensor.empty() : tensor<32x32xui32>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xi32>, tensor<32x32xui32>) -> tensor<32x32xui32>
    return %1 : tensor<32x32xui32>
  }
  func.func public @main17(%arg0: tensor<32x32xf32>) -> tensor<32x32xui32> {
    %0 = tensor.empty() : tensor<32x32xui32>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xf32>, tensor<32x32xui32>) -> tensor<32x32xui32>
    return %1 : tensor<32x32xui32>
  }
  func.func public @main18(%arg0: tensor<32x32xbf16>) -> tensor<32x32xui32> {
    %0 = tensor.empty() : tensor<32x32xui32>
    // CHECK: "ttnn.typecast"
    %1 = "ttir.typecast"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xbf16>, tensor<32x32xui32>) -> tensor<32x32xui32>
    return %1 : tensor<32x32xui32>
  }
}
