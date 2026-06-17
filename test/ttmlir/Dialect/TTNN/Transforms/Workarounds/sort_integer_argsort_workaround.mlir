// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Integer argsort: indices come from an f32 rank-by-comparison (ttnn.sort
  // compares keys in bf16 and returns wrong indices for keys > 256); values
  // still come from ttnn.sort.
  func.func public @argsort_integer(%arg0: tensor<1x10xsi32>) -> (tensor<1x10xsi32>, tensor<1x10xsi32>) {
    // CHECK-LABEL: func.func public @argsort_integer
    // CHECK: "ttnn.typecast"
    // CHECK-SAME: -> tensor<{{.*}}xf32
    // CHECK: "ttnn.arange"
    // CHECK: "ttnn.lt"
    // CHECK: "ttnn.eq"
    // CHECK: "ttnn.sum"
    // CHECK: "ttnn.eq"
    // CHECK: "ttnn.sum"
    // CHECK: "ttnn.typecast"
    // CHECK-SAME: -> tensor<{{.*}}xsi32
    // CHECK: "ttnn.sort"
    %values, %indices = "ttir.sort"(%arg0) <{descending = false, dim = 1 : si32, stable = false}> : (tensor<1x10xsi32>) -> (tensor<1x10xsi32>, tensor<1x10xsi32>)
    return %values, %indices : tensor<1x10xsi32>, tensor<1x10xsi32>
  }

  // Dim > tile width: the reduction is tiled and accumulated, so slice_static
  // and a running add appear (bounds peak memory; no length cap).
  func.func public @argsort_integer_tiled(%arg0: tensor<1x600xsi32>) -> (tensor<1x600xsi32>, tensor<1x600xsi32>) {
    // CHECK-LABEL: func.func public @argsort_integer_tiled
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.sort"
    %values, %indices = "ttir.sort"(%arg0) <{descending = false, dim = 1 : si32, stable = false}> : (tensor<1x600xsi32>) -> (tensor<1x600xsi32>, tensor<1x600xsi32>)
    return %values, %indices : tensor<1x600xsi32>, tensor<1x600xsi32>
  }

  // Indices unused: ttnn.sort values are correct, so it is not decomposed.
  func.func public @sort_values_only(%arg0: tensor<1x10xsi32>) -> tensor<1x10xsi32> {
    // CHECK-LABEL: func.func public @sort_values_only
    // CHECK-NOT: "ttnn.arange"
    // CHECK: "ttnn.sort"
    %values, %indices = "ttir.sort"(%arg0) <{descending = false, dim = 1 : si32, stable = false}> : (tensor<1x10xsi32>) -> (tensor<1x10xsi32>, tensor<1x10xsi32>)
    return %values : tensor<1x10xsi32>
  }
}
