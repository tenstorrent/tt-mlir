// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Min tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @min_single_dim_f32
module {
  func.func @min_single_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_min
    // CHECK-NOT: tosa.reshape
    %0 = "ttir.min"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @min_negative_dim_f32
module {
  func.func @min_negative_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_min
    %0 = "ttir.min"(%arg0) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @min_multi_dim_f32
module {
  func.func @min_multi_dim_f32(%arg0: tensor<1x10x10x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reduce_min
    %0 = "ttir.min"(%arg0) <{dim_arg = [1 : i32, -2 : i32], keep_dim = true}> : (tensor<1x10x10x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @min_no_keep_dim_f32
module {
  func.func @min_no_keep_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x2048xf32> {
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reshape
    %0 = "ttir.min"(%arg0) <{dim_arg = [-2 : i32], keep_dim = false}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x2048xf32>
    return %0 : tensor<1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @min_no_keep_multi_dim_f32
module {
  func.func @min_no_keep_multi_dim_f32(%arg0: tensor<1x10x10x2048xf32>) -> tensor<1x2048xf32> {
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reshape
    %0 = "ttir.min"(%arg0) <{dim_arg = [1 : i32, -2 : i32], keep_dim = false}> : (tensor<1x10x10x2048xf32>) -> tensor<1x2048xf32>
    return %0 : tensor<1x2048xf32>
  }
}

// CHECK-LABEL: func.func @min_single_dim_i32
module {
  func.func @min_single_dim_i32(%arg0: tensor<32x128x128xi32>) -> tensor<32x1x128xi32> {
    // CHECK: tosa.reduce_min
    %0 = "ttir.min"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x128x128xi32>) -> tensor<32x1x128xi32>
    return %0 : tensor<32x1x128xi32>
  }
}

// CHECK-LABEL: func.func @min_no_keep_i32
module {
  func.func @min_no_keep_i32(%arg0: tensor<32x128x128xi32>) -> tensor<32x128xi32> {
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reshape
    %0 = "ttir.min"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x128x128xi32>) -> tensor<32x128xi32>
    return %0 : tensor<32x128xi32>
  }
}

//===----------------------------------------------------------------------===//
// Max tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @max_single_dim_f32
module {
  func.func @max_single_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_max
    // CHECK-NOT: tosa.reshape
    %0 = "ttir.max"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @max_negative_dim_f32
module {
  func.func @max_negative_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_max
    %0 = "ttir.max"(%arg0) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @max_multi_dim_f32
module {
  func.func @max_multi_dim_f32(%arg0: tensor<1x10x10x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_max
    // CHECK: tosa.reduce_max
    %0 = "ttir.max"(%arg0) <{dim_arg = [1 : i32, -2 : i32], keep_dim = true}> : (tensor<1x10x10x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @max_no_keep_dim_f32
module {
  func.func @max_no_keep_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x2048xf32> {
    // CHECK: tosa.reduce_max
    // CHECK: tosa.reshape
    %0 = "ttir.max"(%arg0) <{dim_arg = [-2 : i32], keep_dim = false}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x2048xf32>
    return %0 : tensor<1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @max_no_keep_multi_dim_f32
module {
  func.func @max_no_keep_multi_dim_f32(%arg0: tensor<1x10x10x2048xf32>) -> tensor<1x2048xf32> {
    // CHECK: tosa.reduce_max
    // CHECK: tosa.reduce_max
    // CHECK: tosa.reshape
    %0 = "ttir.max"(%arg0) <{dim_arg = [1 : i32, -2 : i32], keep_dim = false}> : (tensor<1x10x10x2048xf32>) -> tensor<1x2048xf32>
    return %0 : tensor<1x2048xf32>
  }
}

// CHECK-LABEL: func.func @max_single_dim_i32
module {
  func.func @max_single_dim_i32(%arg0: tensor<32x128x128xi32>) -> tensor<32x1x128xi32> {
    // CHECK: tosa.reduce_max
    %0 = "ttir.max"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x128x128xi32>) -> tensor<32x1x128xi32>
    return %0 : tensor<32x1x128xi32>
  }
}

// CHECK-LABEL: func.func @max_no_keep_i32
module {
  func.func @max_no_keep_i32(%arg0: tensor<32x128x128xi32>) -> tensor<32x128xi32> {
    // CHECK: tosa.reduce_max
    // CHECK: tosa.reshape
    %0 = "ttir.max"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x128x128xi32>) -> tensor<32x128xi32>
    return %0 : tensor<32x128xi32>
  }
}

//===----------------------------------------------------------------------===//
// Sum tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @sum_single_dim_f32
module {
  func.func @sum_single_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_sum
    // CHECK-NOT: tosa.reshape
    %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @sum_negative_dim_f32
module {
  func.func @sum_negative_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_sum
    %0 = "ttir.sum"(%arg0) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @sum_multi_dim_f32
module {
  func.func @sum_multi_dim_f32(%arg0: tensor<1x10x10x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32, -2 : i32], keep_dim = true}> : (tensor<1x10x10x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @sum_no_keep_dim_f32
module {
  func.func @sum_no_keep_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x2048xf32> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reshape
    %0 = "ttir.sum"(%arg0) <{dim_arg = [-2 : i32], keep_dim = false}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x2048xf32>
    return %0 : tensor<1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @sum_no_keep_multi_dim_f32
module {
  func.func @sum_no_keep_multi_dim_f32(%arg0: tensor<1x10x10x2048xf32>) -> tensor<1x2048xf32> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reshape
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32, -2 : i32], keep_dim = false}> : (tensor<1x10x10x2048xf32>) -> tensor<1x2048xf32>
    return %0 : tensor<1x2048xf32>
  }
}

// CHECK-LABEL: func.func @sum_single_dim_i32
module {
  func.func @sum_single_dim_i32(%arg0: tensor<32x128x128xi32>) -> tensor<32x1x128xi32> {
    // CHECK: tosa.reduce_sum
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x128x128xi32>) -> tensor<32x1x128xi32>
    return %0 : tensor<32x1x128xi32>
  }
}

// CHECK-LABEL: func.func @sum_no_keep_i32
module {
  func.func @sum_no_keep_i32(%arg0: tensor<32x128x128xi32>) -> tensor<32x128xi32> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reshape
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x128x128xi32>) -> tensor<32x128xi32>
    return %0 : tensor<32x128xi32>
  }
}

//===----------------------------------------------------------------------===//
// Prod tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @prod_single_dim_f32
module {
  func.func @prod_single_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_product
    // CHECK-NOT: tosa.reshape
    %0 = "ttir.prod"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @prod_negative_dim_f32
module {
  func.func @prod_negative_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_product
    %0 = "ttir.prod"(%arg0) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @prod_multi_dim_f32
module {
  func.func @prod_multi_dim_f32(%arg0: tensor<1x10x10x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_product
    // CHECK: tosa.reduce_product
    %0 = "ttir.prod"(%arg0) <{dim_arg = [1 : i32, -2 : i32], keep_dim = true}> : (tensor<1x10x10x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @prod_no_keep_dim_f32
module {
  func.func @prod_no_keep_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x2048xf32> {
    // CHECK: tosa.reduce_product
    // CHECK: tosa.reshape
    %0 = "ttir.prod"(%arg0) <{dim_arg = [-2 : i32], keep_dim = false}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x2048xf32>
    return %0 : tensor<1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @prod_no_keep_multi_dim_f32
module {
  func.func @prod_no_keep_multi_dim_f32(%arg0: tensor<1x10x10x2048xf32>) -> tensor<1x2048xf32> {
    // CHECK: tosa.reduce_product
    // CHECK: tosa.reduce_product
    // CHECK: tosa.reshape
    %0 = "ttir.prod"(%arg0) <{dim_arg = [1 : i32, -2 : i32], keep_dim = false}> : (tensor<1x10x10x2048xf32>) -> tensor<1x2048xf32>
    return %0 : tensor<1x2048xf32>
  }
}

// CHECK-LABEL: func.func @prod_single_dim_i32
module {
  func.func @prod_single_dim_i32(%arg0: tensor<32x128x128xi32>) -> tensor<32x1x128xi32> {
    // CHECK: tosa.reduce_product
    %0 = "ttir.prod"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x128x128xi32>) -> tensor<32x1x128xi32>
    return %0 : tensor<32x1x128xi32>
  }
}

// CHECK-LABEL: func.func @prod_no_keep_i32
module {
  func.func @prod_no_keep_i32(%arg0: tensor<32x128x128xi32>) -> tensor<32x128xi32> {
    // CHECK: tosa.reduce_product
    // CHECK: tosa.reshape
    %0 = "ttir.prod"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x128x128xi32>) -> tensor<32x128xi32>
    return %0 : tensor<32x128xi32>
  }
}

//===----------------------------------------------------------------------===//
// Mean tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mean_single_dim_f32
module {
  func.func @mean_single_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.const
    // CHECK: linalg.div
    %0 = "ttir.mean"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @mean_negative_dim_f32
module {
  func.func @mean_negative_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.const
    // CHECK: linalg.div
    %0 = "ttir.mean"(%arg0) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @mean_multi_dim_f32
module {
  func.func @mean_multi_dim_f32(%arg0: tensor<1x10x10x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.const
    // CHECK: linalg.div
    %0 = "ttir.mean"(%arg0) <{dim_arg = [1 : i32, -2 : i32], keep_dim = true}> : (tensor<1x10x10x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %0 : tensor<1x1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @mean_no_keep_dim_f32
module {
  func.func @mean_no_keep_dim_f32(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x2048xf32> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reshape
    // CHECK: tosa.const
    // CHECK: linalg.div
    %0 = "ttir.mean"(%arg0) <{dim_arg = [-2 : i32], keep_dim = false}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x2048xf32>
    return %0 : tensor<1x1x2048xf32>
  }
}

// CHECK-LABEL: func.func @mean_no_keep_multi_dim_f32
module {
  func.func @mean_no_keep_multi_dim_f32(%arg0: tensor<1x10x10x2048xf32>) -> tensor<1x2048xf32> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reshape
    // CHECK: tosa.const
    // CHECK: linalg.div
    %0 = "ttir.mean"(%arg0) <{dim_arg = [1 : i32, -2 : i32], keep_dim = false}> : (tensor<1x10x10x2048xf32>) -> tensor<1x2048xf32>
    return %0 : tensor<1x2048xf32>
  }
}

// CHECK-LABEL: func.func @mean_i32_to_f32
module {
  func.func @mean_i32_to_f32(%arg0: tensor<32x128x128xi32>) -> tensor<32x1x128xf32> {
    // CHECK: arith.sitofp
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.const
    // CHECK: linalg.div
    %0 = "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x128x128xi32>) -> tensor<32x1x128xf32>
    return %0 : tensor<32x1x128xf32>
  }
}

// CHECK-LABEL: func.func @mean_i32_multi_dim_no_keep
module {
  func.func @mean_i32_multi_dim_no_keep(%arg0: tensor<32x128x128xi32>) -> tensor<32xf32> {
    // CHECK: arith.sitofp
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reshape
    // CHECK: tosa.const
    // CHECK: linalg.div
    %0 = "ttir.mean"(%arg0) <{dim_arg = [1 : i32, 2 : i32], keep_dim = false}> : (tensor<32x128x128xi32>) -> tensor<32xf32>
    return %0 : tensor<32xf32>
  }
}

//===----------------------------------------------------------------------===//
// ArgMax tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @argmax_dim0_f32
module {
  func.func @argmax_dim0_f32(%arg0: tensor<4x8xf32>) -> tensor<8xi32> {
    // CHECK: linalg.fill
    // CHECK: linalg.fill
    // CHECK: linalg.generic
    // CHECK: linalg.index 0
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    // CHECK: arith.select
    // CHECK: linalg.yield
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<4x8xf32>) -> tensor<8xi32>
    return %0 : tensor<8xi32>
  }
}

// CHECK-LABEL: func.func @argmax_dim1_f32
module {
  func.func @argmax_dim1_f32(%arg0: tensor<4x8xf32>) -> tensor<4xi32> {
    // CHECK: linalg.fill
    // CHECK: linalg.fill
    // CHECK: linalg.generic
    // CHECK: linalg.index 1
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    // CHECK: arith.select
    // CHECK: linalg.yield
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<4x8xf32>) -> tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}

// CHECK-LABEL: func.func @argmax_dim0_keep_dim_f32
module {
  func.func @argmax_dim0_keep_dim_f32(%arg0: tensor<4x8xf32>) -> tensor<1x8xi32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    // CHECK: arith.select
    // CHECK: linalg.yield
    // CHECK: tosa.const_shape
    // CHECK: tosa.reshape
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<4x8xf32>) -> tensor<1x8xi32>
    return %0 : tensor<1x8xi32>
  }
}

// CHECK-LABEL: func.func @argmax_all_dims_f32
module {
  func.func @argmax_all_dims_f32(%arg0: tensor<4x8xf32>) -> tensor<i32> {
    // CHECK: linalg.generic
    // CHECK: linalg.index 0
    // CHECK: linalg.index 1
    // CHECK: arith.muli
    // CHECK: arith.addi
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    // CHECK: arith.select
    // CHECK: linalg.yield
    %0 = "ttir.argmax"(%arg0) <{keep_dim = false}> : (tensor<4x8xf32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}

// CHECK-LABEL: func.func @argmax_dim1_i32
module {
  func.func @argmax_dim1_i32(%arg0: tensor<4x8xi32>) -> tensor<4xi32> {
    // CHECK: linalg.fill
    // CHECK: linalg.fill
    // CHECK: linalg.generic
    // CHECK: linalg.index 1
    // CHECK: arith.cmpi sgt
    // CHECK: arith.select
    // CHECK: arith.select
    // CHECK: linalg.yield
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<4x8xi32>) -> tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}

// CHECK-LABEL: func.func @argmax_dim0_i32_keep_dim
module {
  func.func @argmax_dim0_i32_keep_dim(%arg0: tensor<4x8xi32>) -> tensor<1x8xi32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpi sgt
    // CHECK: arith.select
    // CHECK: arith.select
    // CHECK: linalg.yield
    // CHECK: tosa.const_shape
    // CHECK: tosa.reshape
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<4x8xi32>) -> tensor<1x8xi32>
    return %0 : tensor<1x8xi32>
  }
}

//===----------------------------------------------------------------------===//
// CumSum tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cumsum_dim0
module {
  func.func @cumsum_dim0(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK-NOT: scf.for
    %0 = "ttir.cumsum"(%arg0) <{dim = 0 : i64}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}

// CHECK-LABEL: func.func @cumsum_dim1
module {
  func.func @cumsum_dim1(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK-NOT: scf.for
    %0 = "ttir.cumsum"(%arg0) <{dim = 1 : i64}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}

// CHECK-LABEL: func.func @cumsum_negative_dim
module {
  func.func @cumsum_negative_dim(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK-NOT: scf.for
    %0 = "ttir.cumsum"(%arg0) <{dim = -1 : i64}> : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}

// CHECK-LABEL: func.func @cumsum_3d_middle
module {
  func.func @cumsum_3d_middle(%arg0: tensor<2x5x3xf32>) -> tensor<2x5x3xf32> {
    // CHECK: tensor.extract_slice
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice
    // CHECK-NOT: scf.for
    %0 = "ttir.cumsum"(%arg0) <{dim = 1 : i64}> : (tensor<2x5x3xf32>) -> tensor<2x5x3xf32>
    return %0 : tensor<2x5x3xf32>
  }
}

//===----------------------------------------------------------------------===//
// ReduceAnd tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @reduce_and_single_dim_f32
module {
  func.func @reduce_and_single_dim_f32(%arg0: tensor<4x8xf32>) -> tensor<4x1xf32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpf une
    // CHECK: arith.select
    // CHECK: tosa.reduce_min
    // CHECK-NOT: ttir.reduce_and
    %0 = "ttir.reduce_and"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<4x8xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}

// CHECK-LABEL: func.func @reduce_and_no_keep_f32
module {
  func.func @reduce_and_no_keep_f32(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpf une
    // CHECK: arith.select
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reshape
    // CHECK-NOT: ttir.reduce_and
    %0 = "ttir.reduce_and"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<4x8xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @reduce_and_multi_dim_f32
module {
  func.func @reduce_and_multi_dim_f32(%arg0: tensor<4x8x16xf32>) -> tensor<4x1x1xf32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpf une
    // CHECK: arith.select
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reduce_min
    // CHECK-NOT: ttir.reduce_and
    %0 = "ttir.reduce_and"(%arg0) <{dim_arg = [1 : i32, 2 : i32], keep_dim = true}> : (tensor<4x8x16xf32>) -> tensor<4x1x1xf32>
    return %0 : tensor<4x1x1xf32>
  }
}

// CHECK-LABEL: func.func @reduce_and_single_dim_i32
module {
  func.func @reduce_and_single_dim_i32(%arg0: tensor<4x8xi32>) -> tensor<4x1xi32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpi ne
    // CHECK: arith.select
    // CHECK: tosa.reduce_min
    // CHECK-NOT: ttir.reduce_and
    %0 = "ttir.reduce_and"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<4x8xi32>) -> tensor<4x1xi32>
    return %0 : tensor<4x1xi32>
  }
}

// CHECK-LABEL: func.func @reduce_and_no_keep_i32
module {
  func.func @reduce_and_no_keep_i32(%arg0: tensor<4x8xi32>) -> tensor<4xi32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpi ne
    // CHECK: arith.select
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reshape
    %0 = "ttir.reduce_and"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<4x8xi32>) -> tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}

//===----------------------------------------------------------------------===//
// ReduceOr tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @reduce_or_single_dim_f32
module {
  func.func @reduce_or_single_dim_f32(%arg0: tensor<4x8xf32>) -> tensor<4x1xf32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpf une
    // CHECK: arith.select
    // CHECK: tosa.reduce_max
    // CHECK-NOT: ttir.reduce_or
    %0 = "ttir.reduce_or"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<4x8xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}

// CHECK-LABEL: func.func @reduce_or_no_keep_f32
module {
  func.func @reduce_or_no_keep_f32(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpf une
    // CHECK: arith.select
    // CHECK: tosa.reduce_max
    // CHECK: tosa.reshape
    // CHECK-NOT: ttir.reduce_or
    %0 = "ttir.reduce_or"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<4x8xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @reduce_or_multi_dim_f32
module {
  func.func @reduce_or_multi_dim_f32(%arg0: tensor<4x8x16xf32>) -> tensor<4x1x1xf32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpf une
    // CHECK: arith.select
    // CHECK: tosa.reduce_max
    // CHECK: tosa.reduce_max
    // CHECK-NOT: ttir.reduce_or
    %0 = "ttir.reduce_or"(%arg0) <{dim_arg = [1 : i32, 2 : i32], keep_dim = true}> : (tensor<4x8x16xf32>) -> tensor<4x1x1xf32>
    return %0 : tensor<4x1x1xf32>
  }
}

// CHECK-LABEL: func.func @reduce_or_single_dim_i32
module {
  func.func @reduce_or_single_dim_i32(%arg0: tensor<4x8xi32>) -> tensor<4x1xi32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpi ne
    // CHECK: arith.select
    // CHECK: tosa.reduce_max
    // CHECK-NOT: ttir.reduce_or
    %0 = "ttir.reduce_or"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<4x8xi32>) -> tensor<4x1xi32>
    return %0 : tensor<4x1xi32>
  }
}

// CHECK-LABEL: func.func @reduce_or_no_keep_i32
module {
  func.func @reduce_or_no_keep_i32(%arg0: tensor<4x8xi32>) -> tensor<4xi32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpi ne
    // CHECK: arith.select
    // CHECK: tosa.reduce_max
    // CHECK: tosa.reshape
    %0 = "ttir.reduce_or"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<4x8xi32>) -> tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}

//===----------------------------------------------------------------------===//
// Edge case: empty dim_arg (reduce all dims)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @sum_all_dims_f32
module {
  func.func @sum_all_dims_f32(%arg0: tensor<4x8xf32>) -> tensor<f32> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reshape
    %0 = "ttir.sum"(%arg0) <{keep_dim = false}> : (tensor<4x8xf32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}

// CHECK-LABEL: func.func @min_all_dims_keep_f32
module {
  func.func @min_all_dims_keep_f32(%arg0: tensor<4x8xf32>) -> tensor<1x1xf32> {
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reduce_min
    // CHECK-NOT: tosa.reshape
    %0 = "ttir.min"(%arg0) <{keep_dim = true}> : (tensor<4x8xf32>) -> tensor<1x1xf32>
    return %0 : tensor<1x1xf32>
  }
}

//===----------------------------------------------------------------------===//
// Edge case: mean with i1 input (UIToFP path)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mean_i1_to_f32
module {
  func.func @mean_i1_to_f32(%arg0: tensor<4x8xi1>) -> tensor<4x1xf32> {
    // CHECK: arith.uitofp
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.const
    // CHECK: linalg.div
    %0 = "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<4x8xi1>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
