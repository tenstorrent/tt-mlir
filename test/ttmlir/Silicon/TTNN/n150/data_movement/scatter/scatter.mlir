// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {

  // default
  func.func @scatter(%arg0: tensor<1x3x320x320xf32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> {
    %0 = ttir.empty() : tensor<1x3x320x320xf32>
    %2 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1, 2, 3>}> : (tensor<1x3x320x320xf32>, tensor<1x1xi32>, tensor<1x3x32x32xf32>, tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32>
    // CHECK-LABEL: func.func @scatter
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32], ends = [1 : i32, 1 : i32], step = [1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x1xsi32, {{.*}}>) -> tensor<1x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x1xsi32, {{.*}}>) -> tensor<1x1x1x1xsi32, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x3x32x32>}>
    // CHECK-SAME: (tensor<1x1x1x1xsi32, {{.*}}>) -> tensor<1x3x32x32xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1x3x320x320xf32, {{.*}}>, tensor<1x3x32x32xsi32, {{.*}}>, tensor<1x3x32x32xf32, {{.*}}>) -> tensor<1x3x320x320xf32, {{.*}}>
    return %2 : tensor<1x3x320x320xf32>
  }

  // gpt-oss - multi-dimensional scatter with scatter operation broken into multiple scatter operations each handling index_shape[dim] < 256
  func.func @scatter_1(%arg0: tensor<71x32xbf16>, %arg1: tensor<71x4x2xi64>, %arg2: tensor<71x4xbf16>) -> tensor<71x32xbf16> {
    %0 = ttir.empty() : tensor<71x32xbf16>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0, 1>, scatter_dims_to_operand_dims = array<i32: 0, 1>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32>}> : (tensor<71x32xbf16>, tensor<71x4x2xi64>, tensor<71x4xbf16>, tensor<71x32xbf16>) -> tensor<71x32xbf16>
    // CHECK-LABEL: func.func @scatter_1
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [71 : i32, 4 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<71x4x2xsi32, {{.*}}>) -> tensor<71x4x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [71 : i32, 4 : i32]}>
    // CHECK-SAME: (tensor<71x4x1xsi32, {{.*}}>) -> tensor<71x4xsi32, {{.*}}>
    // CHECK: "ttnn.full"({{.*}}) <{dtype = #ttcore.supportedDataTypes<si32>, fill_value = 32 : i32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<71x4>}>
    // CHECK-SAME: (!ttnn.device) -> tensor<71x4xsi32, {{.*}}>
    // CHECK: "ttnn.multiply"({{.*}}) <{dtype = #ttcore.supportedDataTypes<si32>}>
    // CHECK-SAME: (tensor<71x4xsi32, {{.*}}>, tensor<71x4xsi32, {{.*}}>) -> tensor<71x4xsi32, {{.*}}>
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 1 : i32], ends = [71 : i32, 4 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<71x4x2xsi32, {{.*}}>) -> tensor<71x4x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [71 : i32, 4 : i32]}>
    // CHECK-SAME: (tensor<71x4x1xsi32, {{.*}}>) -> tensor<71x4xsi32, {{.*}}>
    // CHECK: "ttnn.add"({{.*}}) <{dtype = #ttcore.supportedDataTypes<si32>}>
    // CHECK-SAME: (tensor<71x4xsi32, {{.*}}>, tensor<71x4xsi32, {{.*}}>) -> tensor<71x4xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [2272 : i32]}>
    // CHECK-SAME: (tensor<71x32xbf16, {{.*}}>) -> tensor<2272xbf16, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [284 : i32]}>
    // CHECK-SAME: (tensor<71x4xbf16, {{.*}}>) -> tensor<284xbf16, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [284 : i32]}>
    // CHECK-SAME: (tensor<71x4xsi32, {{.*}}>) -> tensor<284xsi32, {{.*}}>
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32], ends = [256 : i32], step = [1 : i32]}>
    // CHECK-SAME: (tensor<284xsi32, {{.*}}>) -> tensor<256xsi32, {{.*}}>
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32], ends = [256 : i32], step = [1 : i32]}>
    // CHECK-SAME: (tensor<284xbf16, {{.*}}>) -> tensor<256xbf16, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<2272xbf16, {{.*}}>, tensor<256xsi32, {{.*}}>, tensor<256xbf16, {{.*}}>) -> tensor<2272xbf16, {{.*}}>
    // Second scatter batch (remaining 28 elements)
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [256 : i32], ends = [284 : i32], step = [1 : i32]}>
    // CHECK-SAME: (tensor<284xsi32, {{.*}}>) -> tensor<28xsi32, {{.*}}>
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [256 : i32], ends = [284 : i32], step = [1 : i32]}>
    // CHECK-SAME: (tensor<284xbf16, {{.*}}>) -> tensor<28xbf16, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<2272xbf16, {{.*}}>, tensor<28xsi32, {{.*}}>, tensor<28xbf16, {{.*}}>) -> tensor<2272xbf16, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [71 : i32, 32 : i32]}>
    // CHECK-SAME: (tensor<2272xbf16, {{.*}}>) -> tensor<71x32xbf16, {{.*}}>
    return %1 : tensor<71x32xbf16>
  }

  // https://github.com/tenstorrent/tt-mlir/issues/4531
  func.func @scatter_2(%arg0: tensor<1000x32xf32>, %arg1: tensor<10x1xi64>, %arg2: tensor<10x32xf32>) -> tensor<1000x32xf32> {
    %0 = ttir.empty() : tensor<1000x32xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 1 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 1>}> : (tensor<1000x32xf32>, tensor<10x1xi64>, tensor<10x32xf32>, tensor<1000x32xf32>) -> tensor<1000x32xf32>
    // CHECK-LABEL: func.func @scatter_2
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32], ends = [10 : i32, 1 : i32], step = [1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<10x1xsi32, {{.*}}>) -> tensor<10x1xsi32, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x32>}>
    // CHECK-SAME: (tensor<10x1xsi32, {{.*}}>) -> tensor<10x32xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1000x32xf32, {{.*}}>, tensor<10x32xsi32, {{.*}}>, tensor<10x32xf32, {{.*}}>) -> tensor<1000x32xf32, {{.*}}>
    return %1 : tensor<1000x32xf32>
  }

  // https://github.com/tenstorrent/tt-mlir/issues/4792
  func.func @scatter_3(%arg0: tensor<2050x768xf32>, %arg1: tensor<1x5x1xi32>, %arg2: tensor<1x5x768xf32>) -> tensor<2050x768xf32> {
    %0 = ttir.empty() : tensor<2050x768xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<2050x768xf32>, tensor<1x5x1xi32>, tensor<1x5x768xf32>, tensor<2050x768xf32>) -> tensor<2050x768xf32>
    // CHECK-LABEL: func.func @scatter_3
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 5 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x5x1xsi32, {{.*}}>) -> tensor<1x5x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 2050 : i32, 768 : i32]}>
    // CHECK-SAME: (tensor<2050x768xf32, {{.*}}>) -> tensor<1x2050x768xf32, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x1x768>}>
    // CHECK-SAME: (tensor<1x5x1xsi32, {{.*}}>) -> tensor<1x5x768xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1x2050x768xf32, {{.*}}>, tensor<1x5x768xsi32, {{.*}}>, tensor<1x5x768xf32, {{.*}}>) -> tensor<2050x768xf32, {{.*}}>
    return %1 : tensor<2050x768xf32>
  }

  // https://github.com/tenstorrent/tt-mlir/issues/4792
  func.func @scatter_4(%arg0: tensor<50272x768xf32>, %arg1: tensor<1x5x1xi32>, %arg2: tensor<1x5x768xf32>) -> tensor<50272x768xf32> {
    %0 = ttir.empty() : tensor<50272x768xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<50272x768xf32>, tensor<1x5x1xi32>, tensor<1x5x768xf32>, tensor<50272x768xf32>) -> tensor<50272x768xf32>
    // CHECK-LABEL: func.func @scatter_4
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 5 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x5x1xsi32, {{.*}}>) -> tensor<1x5x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 50272 : i32, 768 : i32]}>
    // CHECK-SAME: (tensor<50272x768xf32, {{.*}}>) -> tensor<1x50272x768xf32, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x1x768>}>
    // CHECK-SAME: (tensor<1x5x1xsi32, {{.*}}>) -> tensor<1x5x768xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1x50272x768xf32, {{.*}}>, tensor<1x5x768xsi32, {{.*}}>, tensor<1x5x768xf32, {{.*}}>) -> tensor<50272x768xf32, {{.*}}>
    return %1 : tensor<50272x768xf32>
  }

  // https://github.com/tenstorrent/tt-mlir/issues/4792
  func.func @scatter_5(%arg0: tensor<256008x1024xbf16>, %arg1: tensor<1x8x1xi32>, %arg2: tensor<1x8x1024xbf16>) -> tensor<256008x1024xbf16> {
    %0 = ttir.empty() : tensor<256008x1024xbf16>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<256008x1024xbf16>, tensor<1x8x1xi32>, tensor<1x8x1024xbf16>, tensor<256008x1024xbf16>) -> tensor<256008x1024xbf16>
    // CHECK-LABEL: func.func @scatter_5
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x8x1xsi32, {{.*}}>) -> tensor<1x8x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 256008 : i32, 1024 : i32]}>
    // CHECK-SAME: (tensor<256008x1024xbf16, {{.*}}>) -> tensor<1x256008x1024xbf16, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x1x1024>}>
    // CHECK-SAME: (tensor<1x8x1xsi32, {{.*}}>) -> tensor<1x8x1024xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1x256008x1024xbf16, {{.*}}>, tensor<1x8x1024xsi32, {{.*}}>, tensor<1x8x1024xbf16, {{.*}}>) -> tensor<256008x1024xbf16, {{.*}}>

    return %1 : tensor<256008x1024xbf16>
  }

  // https://github.com/tenstorrent/tt-mlir/issues/4792
  func.func @scatter_6(%arg0: tensor<1026x768xbf16>, %arg1: tensor<1x13x1xi32>, %arg2: tensor<1x13x768xbf16>) -> tensor<1026x768xbf16> {
    %0 = ttir.empty() : tensor<1026x768xbf16>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<1026x768xbf16>, tensor<1x13x1xi32>, tensor<1x13x768xbf16>, tensor<1026x768xbf16>) -> tensor<1026x768xbf16>
    // CHECK-LABEL: func.func @scatter_6
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 13 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x13x1xsi32, {{.*}}>) -> tensor<1x13x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 1026 : i32, 768 : i32]}>
    // CHECK-SAME: (tensor<1026x768xbf16, {{.*}}>) -> tensor<1x1026x768xbf16, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x1x768>}>
    // CHECK-SAME: (tensor<1x13x1xsi32, {{.*}}>) -> tensor<1x13x768xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1x1026x768xbf16, {{.*}}>, tensor<1x13x768xsi32, {{.*}}>, tensor<1x13x768xbf16, {{.*}}>) -> tensor<1026x768xbf16, {{.*}}>

    return %1 : tensor<1026x768xbf16>
  }

  // https://github.com/tenstorrent/tt-mlir/issues/4792
  func.func @scatter_7(%arg0: tensor<50265x768xbf16>, %arg1: tensor<1x13x1xi32>, %arg2: tensor<1x13x768xbf16>) -> tensor<50265x768xbf16> {
    %0 = ttir.empty() : tensor<50265x768xbf16>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<50265x768xbf16>, tensor<1x13x1xi32>, tensor<1x13x768xbf16>, tensor<50265x768xbf16>) -> tensor<50265x768xbf16>
    // CHECK-LABEL: func.func @scatter_7
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 13 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x13x1xsi32, {{.*}}>) -> tensor<1x13x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 50265 : i32, 768 : i32]}>
    // CHECK-SAME: (tensor<50265x768xbf16, {{.*}}>) -> tensor<1x50265x768xbf16, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x1x768>}>
    // CHECK-SAME: (tensor<1x13x1xsi32, {{.*}}>) -> tensor<1x13x768xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1x50265x768xbf16, {{.*}}>, tensor<1x13x768xsi32, {{.*}}>, tensor<1x13x768xbf16, {{.*}}>) -> tensor<50265x768xbf16, {{.*}}>

    return %1 : tensor<50265x768xbf16>
  }

  // https://github.com/tenstorrent/tt-mlir/issues/4792
  func.func @scatter_8(%arg0: tensor<250880x1024xf32>, %arg1: tensor<1x4x1xi32>, %arg2: tensor<1x4x1024xf32>) -> tensor<250880x1024xf32> {
    %0 = ttir.empty() : tensor<250880x1024xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<250880x1024xf32>, tensor<1x4x1xi32>, tensor<1x4x1024xf32>, tensor<250880x1024xf32>) -> tensor<250880x1024xf32>
    // CHECK-LABEL: func.func @scatter_8
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 4 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x4x1xsi32, {{.*}}>) -> tensor<1x4x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 250880 : i32, 1024 : i32]}>
    // CHECK-SAME: (tensor<250880x1024xf32, {{.*}}>) -> tensor<1x250880x1024xf32, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x1x1024>}>
    // CHECK-SAME: (tensor<1x4x1xsi32, {{.*}}>) -> tensor<1x4x1024xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1x250880x1024xf32, {{.*}}>, tensor<1x4x1024xsi32, {{.*}}>, tensor<1x4x1024xf32, {{.*}}>) -> tensor<250880x1024xf32, {{.*}}>

    return %1 : tensor<250880x1024xf32>
  }

  // https://github.com/tenstorrent/tt-mlir/issues/4792
  func.func @scatter_9(%arg0: tensor<512x768xf32>, %arg1: tensor<1x5x1xi32>, %arg2: tensor<1x5x768xf32>) -> tensor<512x768xf32> {
    %0 = ttir.empty() : tensor<512x768xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<512x768xf32>, tensor<1x5x1xi32>, tensor<1x5x768xf32>, tensor<512x768xf32>) -> tensor<512x768xf32>
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 5 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x5x1xsi32, {{.*}}>) -> tensor<1x5x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 512 : i32, 768 : i32]}>
    // CHECK-SAME: (tensor<512x768xf32, {{.*}}>) -> tensor<1x512x768xf32, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x1x768>}>
    // CHECK-SAME: (tensor<1x5x1xsi32, {{.*}}>) -> tensor<1x5x768xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1x512x768xf32, {{.*}}>, tensor<1x5x768xsi32, {{.*}}>, tensor<1x5x768xf32, {{.*}}>) -> tensor<512x768xf32, {{.*}}>
    return %1 : tensor<512x768xf32>
  }

  // https://github.com/tenstorrent/tt-mlir/issues/4792
  func.func @scatter_10(%arg0: tensor<30522x768xf32>, %arg1: tensor<1x5x1xi32>, %arg2: tensor<1x5x768xf32>) -> tensor<30522x768xf32> {
    %0 = ttir.empty() : tensor<30522x768xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<30522x768xf32>, tensor<1x5x1xi32>, tensor<1x5x768xf32>, tensor<30522x768xf32>) -> tensor<30522x768xf32>
    // CHECK-LABEL: func.func @scatter_10
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 5 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x5x1xsi32, {{.*}}>) -> tensor<1x5x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 30522 : i32, 768 : i32]}>
    // CHECK-SAME: (tensor<30522x768xf32, {{.*}}>) -> tensor<1x30522x768xf32, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x1x768>}>
    // CHECK-SAME: (tensor<1x5x1xsi32, {{.*}}>) -> tensor<1x5x768xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1x30522x768xf32, {{.*}}>, tensor<1x5x768xsi32, {{.*}}>, tensor<1x5x768xf32, {{.*}}>) -> tensor<30522x768xf32, {{.*}}>
    return %1 : tensor<30522x768xf32>
  }

  // https://github.com/tenstorrent/tt-mlir/issues/4792
  func.func @scatter_11(%arg0: tensor<512x128xf32>, %arg1: tensor<1x10x1xi32>, %arg2: tensor<1x10x128xf32>) -> tensor<512x128xf32> {
    %0 = ttir.empty() : tensor<512x128xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<512x128xf32>, tensor<1x10x1xi32>, tensor<1x10x128xf32>, tensor<512x128xf32>) -> tensor<512x128xf32>
    // CHECK-LABEL: func.func @scatter_11
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 10 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x10x1xsi32, {{.*}}>) -> tensor<1x10x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 512 : i32, 128 : i32]}>
    // CHECK-SAME: (tensor<512x128xf32, {{.*}}>) -> tensor<1x512x128xf32, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x1x128>}>
    // CHECK-SAME: (tensor<1x10x1xsi32, {{.*}}>) -> tensor<1x10x128xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1x512x128xf32, {{.*}}>, tensor<1x10x128xsi32, {{.*}}>, tensor<1x10x128xf32, {{.*}}>) -> tensor<512x128xf32, {{.*}}>
    return %1 : tensor<512x128xf32>
  }

  // https://github.com/tenstorrent/tt-mlir/issues/4792
  func.func @scatter_12(%arg0: tensor<2x128xf32>, %arg1: tensor<1x10x1xi32>, %arg2: tensor<1x10x128xf32>) -> tensor<2x128xf32> {
    %0 = ttir.empty() : tensor<2x128xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<2x128xf32>, tensor<1x10x1xi32>, tensor<1x10x128xf32>, tensor<2x128xf32>) -> tensor<2x128xf32>
    // CHECK-LABEL: func.func @scatter_12
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 10 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x10x1xsi32, {{.*}}>) -> tensor<1x10x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 2 : i32, 128 : i32]}>
    // CHECK-SAME: (tensor<2x128xf32, {{.*}}>) -> tensor<1x2x128xf32, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x1x128>}>
    // CHECK-SAME: (tensor<1x10x1xsi32, {{.*}}>) -> tensor<1x10x128xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1x2x128xf32, {{.*}}>, tensor<1x10x128xsi32, {{.*}}>, tensor<1x10x128xf32, {{.*}}>) -> tensor<2x128xf32, {{.*}}>
    return %1 : tensor<2x128xf32>
  }

  // https://github.com/tenstorrent/tt-mlir/issues/4792
  func.func @scatter_13(%arg0: tensor<30000x128xf32>, %arg1: tensor<1x10x1xi32>, %arg2: tensor<1x10x128xf32>) -> tensor<30000x128xf32> {
    %0 = ttir.empty() : tensor<30000x128xf32>
    %1 = "ttir.scatter"(%arg0, %arg1, %arg2, %0) <{index_vector_dim = 2 : i32, indices_are_sorted = false, input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>, scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims = array<i32>, unique_indices = false, update_window_dims = array<i32: 2>}> : (tensor<30000x128xf32>, tensor<1x10x1xi32>, tensor<1x10x128xf32>, tensor<30000x128xf32>) -> tensor<30000x128xf32>
    // CHECK-LABEL: func.func @scatter_13
    // CHECK: "ttnn.slice_static"({{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 10 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    // CHECK-SAME: (tensor<1x10x1xsi32, {{.*}}>) -> tensor<1x10x1xsi32, {{.*}}>
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [1 : i32, 30000 : i32, 128 : i32]}>
    // CHECK-SAME: (tensor<30000x128xf32, {{.*}}>) -> tensor<1x30000x128xf32, {{.*}}>
    // CHECK: "ttnn.repeat"({{.*}}) <{repeat_dims = #ttnn.shape<1x1x128>}>
    // CHECK-SAME: (tensor<1x10x1xsi32, {{.*}}>) -> tensor<1x10x128xsi32, {{.*}}>
    // CHECK: "ttnn.scatter"({{.*}}) <{dim = 0 : i32}>
    // CHECK-SAME: (tensor<1x30000x128xf32, {{.*}}>, tensor<1x10x128xsi32, {{.*}}>, tensor<1x10x128xf32, {{.*}}>) -> tensor<30000x128xf32, {{.*}}>
    return %1 : tensor<30000x128xf32>
  }
}
