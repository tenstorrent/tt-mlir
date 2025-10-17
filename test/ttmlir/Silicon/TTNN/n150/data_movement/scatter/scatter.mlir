// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// The following are simple element-wise scatter tests
// shape(input) == shape(output)
// single index dimension (len(scatter_dims_to_operand_dims) == 1)

func.func @scatter_simple_1(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
  %0 = ttir.empty() : tensor<1x3x320x320xf32>
  %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1, 2, 3>}> : (tensor<1x3x320x320xf32>, tensor<1x1xi32>, tensor<1x3x32x32xf32>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
  // CHECK-LABEL: func.func @scatter_simple_1
  // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
  // CHECK-SAME: (tensor<1x1xsi32, {{.*}}>) -> tensor<1x1x1x1xsi32, {{.*}}>
  // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x3x32x32>}>
  // CHECK: "ttnn.scatter"({{.*}}) <{cq_id = 0 : ui32, dim = 0 : i32}>
  // CHECK-SAME: (tensor<1x3x320x320xf32, {{.*}}>, tensor<1x3x32x32xsi32, {{.*}}>, tensor<1x3x32x32xf32, {{.*}}>) -> tensor<1x3x320x320xf32, {{.*}}>
  return %1 : tensor<1x3x320x320xf32>
}

func.func @scatter_simple_2(%arg0: tensor<32x32xi32>, %arg1: tensor<16x1xi32>, %arg2: tensor<16x32xi32>) -> tensor<32x32xi32> {
  %0 = ttir.empty() : tensor<32x32xi32>
  %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1>}> : (tensor<32x32xi32>, tensor<16x1xi32>, tensor<16x32xi32>, tensor<32x32xi32>) -> tensor<32x32xi32>
  // CHECK-LABEL: func.func @scatter_simple_2
  // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x32>}>
  // CHECK: "ttnn.scatter"({{.*}}) <{cq_id = 0 : ui32, dim = 0 : i32}>
  // CHECK-SAME: (tensor<32x32xsi32, {{.*}}>, tensor<16x32xsi32, {{.*}}>, tensor<16x32xsi32, {{.*}}>) -> tensor<32x32xsi32, {{.*}}>
  return %1 : tensor<32x32xi32>
}

// gpt-oss - multi-dimensional scatter with scatter operation broken into multiple scatter operations each handling index_shape[dim] < 256
// We flatten multi-dimensional scatter operations.
func.func @scatter_simple_3(%arg0: tensor<71x32xbf16>, %arg1: tensor<71x4x2xi64>, %arg2: tensor<71x4xbf16>) -> tensor<71x32xbf16> {
  %0 = ttir.empty() : tensor<71x32xbf16>
  // CHECK-LABEL: func.func @scatter_simple_3
  %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0, 1>, scatter_dims_to_operand_dims = array<i32: 0, 1>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32>}> : (tensor<71x32xbf16>, tensor<71x4x2xi64>, tensor<71x4xbf16>, tensor<71x32xbf16>) -> tensor<71x32xbf16>
  // Scatter indices shape is [71, 4, 2], so we break it into 2 slices of shape [71, 4, 1]
  // Then we reshape each slice to [71,4] and calculate flattened indices using strides.
  // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [71 : i32, 4 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
  // CHECK: "ttnn.multiply"({{.*}}) <{dtype = #ttcore.supportedDataTypes<si32>}>
  // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 1 : i32], ends = [71 : i32, 4 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
  // CHECK: "ttnn.add"({{.*}})
  // flatten indices:
  // CHECK: "ttnn.reshape"({{.*}}) <{shape = [284 : i32]}>
  // flatten input:
  // CHECK: "ttnn.reshape"({{.*}}) <{shape = [2272 : i32]}>
  // flatten updates:
  // CHECK: "ttnn.reshape"({{.*}}) <{shape = [284 : i32]}>
  // Scatter is broken into chunks where index_shape[dim] < 256.
  // CHECK: "ttnn.scatter"({{.*}}) <{cq_id = 0 : ui32, dim = 0 : i32}> : (tensor<2272xbf16, {{.*}}>, tensor<256xsi32, {{.*}}>, tensor<256xbf16, {{.*}}>) -> tensor<2272xbf16, {{.*}}>
  // CHECK: "ttnn.scatter"({{.*}}) <{cq_id = 0 : ui32, dim = 0 : i32}> : (tensor<2272xbf16, {{.*}}>, tensor<28xsi32, {{.*}}>, tensor<28xbf16, {{.*}}>) -> tensor<2272xbf16, {{.*}}>
  // reshape flattened output to expected output shape
  // CHECK: "ttnn.reshape"({{.*}}) <{shape = [71 : i32, 32 : i32]}>
  return %1 : tensor<71x32xbf16>
}

// Scatter with f32. For f32, there is a layout conversion before and after scatter due to tt-metal f32 scatter limitations.
func.func @scatter_simple_4(%arg0: tensor<1000x32xf32>, %arg1: tensor<10x1xi64>, %arg2: tensor<10x32xf32>) -> tensor<1000x32xf32> {
  %0 = ttir.empty() : tensor<1000x32xf32>
  // CHECK-LABEL: func.func @scatter_simple_4
  // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x32>}>
  // CHECK: "ttnn.to_layout"({{.*}}) <{layout = #ttnn.layout<row_major>}>
  // CHECK: "ttnn.to_layout"({{.*}}) <{layout = #ttnn.layout<row_major>}>
  // CHECK: "ttnn.scatter"({{.*}}) <{cq_id = 0 : ui32, dim = 0 : i32}> : (tensor<1000x32xf32, {{.*}}>, tensor<10x32xsi32, {{.*}}>, tensor<10x32xf32, {{.*}}>) -> tensor<1000x32xf32, {{.*}}>
  // CHECK: "ttnn.to_layout"({{.*}}) <{layout = #ttnn.layout<tile>}>
  %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1>}> : (tensor<1000x32xf32>, tensor<10x1xi64>, tensor<10x32xf32>, tensor<1000x32xf32>) -> tensor<1000x32xf32>
  return %1 : tensor<1000x32xf32>
}
