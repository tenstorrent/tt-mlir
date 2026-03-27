// RUN: ttmlir-opt -ttir-cpu-boolean-narrowing -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Comparison op narrowed to i1, typecast at return.
  // CHECK-LABEL: func.func @ge_narrowing
  func.func @ge_narrowing(%arg0: tensor<4x1xi32>, %arg1: tensor<1x4xi32>) -> tensor<4x4xf32> {
    // CHECK: "ttir.ge"(%arg0, %arg1) : (tensor<4x1xi32>, tensor<1x4xi32>) -> tensor<4x4xi1>
    // CHECK: "ttir.typecast"
    // CHECK-SAME: (tensor<4x4xi1>) -> tensor<4x4xf32>
    %0 = "ttir.ge"(%arg0, %arg1) : (tensor<4x1xi32>, tensor<1x4xi32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // Boolean chain: ge → logical_and, both narrowed to i1.
  // CHECK-LABEL: func.func @boolean_chain
  func.func @boolean_chain(%arg0: tensor<4x1xi32>, %arg1: tensor<1x4xi32>) -> tensor<4x4xf32> {
    // CHECK: %[[GE:.*]] = "ttir.ge"{{.*}} -> tensor<4x4xi1>
    // CHECK: "ttir.logical_and"(%[[GE]], %[[GE]]) : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
    // CHECK: "ttir.typecast"
    // CHECK-SAME: (tensor<4x4xi1>) -> tensor<4x4xf32>
    %0 = "ttir.ge"(%arg0, %arg1) : (tensor<4x1xi32>, tensor<1x4xi32>) -> tensor<4x4xf32>
    %1 = "ttir.logical_and"(%0, %0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }

  // Mixed: ge feeds both logical_and (boolean) and add (arithmetic).
  // CHECK-LABEL: func.func @mixed_consumers
  func.func @mixed_consumers(%arg0: tensor<4x1xi32>, %arg1: tensor<1x4xi32>, %arg2: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
    // CHECK: %[[GE:.*]] = "ttir.ge"{{.*}} -> tensor<4x4xi1>
    // CHECK: %[[CAST:.*]] = "ttir.typecast"(%[[GE]]){{.*}} -> tensor<4x4xf32>
    // CHECK: "ttir.logical_and"(%[[GE]], %[[GE]]) : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
    // CHECK: "ttir.add"(%[[CAST]], %arg2)
    %0 = "ttir.ge"(%arg0, %arg1) : (tensor<4x1xi32>, tensor<1x4xi32>) -> tensor<4x4xf32>
    %1 = "ttir.logical_and"(%0, %0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = "ttir.add"(%0, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1, %2 : tensor<4x4xf32>, tensor<4x4xf32>
  }

  // Already i1: no change.
  // CHECK-LABEL: func.func @already_i1
  func.func @already_i1(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi1> {
    // CHECK: "ttir.eq"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
    // CHECK-NOT: "ttir.typecast"
    %0 = "ttir.eq"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
    return %0 : tensor<4x4xi1>
  }

  // All comparison ops narrow.
  // CHECK-LABEL: func.func @all_comparisons
  func.func @all_comparisons(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
    // CHECK: "ttir.eq"{{.*}} -> tensor<4x4xi1>
    // CHECK: "ttir.ne"{{.*}} -> tensor<4x4xi1>
    // CHECK: "ttir.gt"{{.*}} -> tensor<4x4xi1>
    // CHECK: "ttir.ge"{{.*}} -> tensor<4x4xi1>
    // CHECK: "ttir.lt"{{.*}} -> tensor<4x4xi1>
    // CHECK: "ttir.le"{{.*}} -> tensor<4x4xi1>
    %0 = "ttir.eq"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = "ttir.ne"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = "ttir.gt"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = "ttir.ge"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = "ttir.lt"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %5 = "ttir.le"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0, %1, %2, %3, %4, %5 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
  }

  // Logical not narrows.
  // CHECK-LABEL: func.func @logical_not_narrowing
  func.func @logical_not_narrowing(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK: %[[GE:.*]] = "ttir.ge"{{.*}} -> tensor<4x4xi1>
    // CHECK: "ttir.logical_not"(%[[GE]]) : (tensor<4x4xi1>) -> tensor<4x4xi1>
    %0 = "ttir.ge"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = "ttir.logical_not"(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }

  // i1 propagates through slice — typecast only on the sliced result.
  // CHECK-LABEL: func.func @slice_propagation
  func.func @slice_propagation(%arg0: tensor<1x1x1x32768x1xi32>, %arg1: tensor<1x1x1x1x32768xi32>) -> tensor<1x1x1x4x4xf32> {
    // CHECK: "ttir.ge"{{.*}} -> tensor<1x1x1x32768x32768xi1>
    // CHECK: "ttir.slice_static"{{.*}} -> tensor<1x1x1x4x4xi1>
    // CHECK: "ttir.typecast"{{.*}} (tensor<1x1x1x4x4xi1>) -> tensor<1x1x1x4x4xf32>
    // CHECK-NOT: tensor<1x1x1x32768x32768xf32>
    %0 = "ttir.ge"(%arg0, %arg1) : (tensor<1x1x1x32768x1xi32>, tensor<1x1x1x1x32768xi32>) -> tensor<1x1x1x32768x32768xf32>
    %1 = "ttir.slice_static"(%0) <{begins = [0:i32,0:i32,0:i32,0:i32,0:i32], ends = [1:i32,1:i32,1:i32,4:i32,4:i32], step = [1:i32,1:i32,1:i32,1:i32,1:i32]}> : (tensor<1x1x1x32768x32768xf32>) -> tensor<1x1x1x4x4xf32>
    return %1 : tensor<1x1x1x4x4xf32>
  }

  // i1 propagates through reshape.
  // CHECK-LABEL: func.func @reshape_propagation
  func.func @reshape_propagation(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<16xf32> {
    // CHECK: "ttir.ge"{{.*}} -> tensor<4x4xi1>
    // CHECK: "ttir.reshape"{{.*}} -> tensor<16xi1>
    // CHECK: "ttir.typecast"{{.*}} (tensor<16xi1>) -> tensor<16xf32>
    %0 = "ttir.ge"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = "ttir.reshape"(%0) <{shape = [16 : i32]}> : (tensor<4x4xf32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
}
