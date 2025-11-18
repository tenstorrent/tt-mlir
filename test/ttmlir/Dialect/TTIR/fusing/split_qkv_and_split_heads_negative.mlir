// RUN: ttmlir-opt -ttir-to-ttir-decomposition -ttir-implicit-broadcast-fold -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @inconsistent_head_dims_0(%arg0: tensor<1x61x512xbf16>, %arg1: tensor<384x512xbf16>, %arg2: tensor<384x512xbf16>, %arg3: tensor<384x512xbf16>) -> (tensor<1x61x64x6xbf16>, tensor<1x6x64x61xbf16>, tensor<1x6x61x64xbf16>) {
    // CHECK-NOT: ttnn.split_query_key_value_and_split_heads

    %0 = ttir.empty() : tensor<61x512xbf16>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [61 : i32, 512 : i32]}> : (tensor<1x61x512xbf16>, tensor<61x512xbf16>) -> tensor<61x512xbf16> // [batch_size, sequence_length, hidden_dimensions]

    // Query projection - wrong output dims %7.
    %2 = ttir.empty() : tensor<61x384xbf16>
    %3 = "ttir.matmul"(%1, %arg1, %2) <{transpose_a = false, transpose_b = true}> : (tensor<61x512xbf16>, tensor<384x512xbf16>, tensor<61x384xbf16>) -> tensor<61x384xbf16>
    %4 = ttir.empty() : tensor<1x61x6x64xbf16>
    %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 61 : i32, 6 : i32, 64 : i32]}> : (tensor<61x384xbf16>, tensor<1x61x6x64xbf16>) -> tensor<1x61x6x64xbf16>
    %6 = ttir.empty() : tensor<1x61x64x6xbf16>
    %7 = "ttir.permute"(%5, %6) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x61x6x64xbf16>, tensor<1x61x64x6xbf16>) -> tensor<1x61x64x6xbf16>

    // Key projection
    %8 = ttir.empty() : tensor<61x384xbf16>
    %9 = "ttir.matmul"(%1, %arg2, %8) <{transpose_a = false, transpose_b = true}> : (tensor<61x512xbf16>, tensor<384x512xbf16>, tensor<61x384xbf16>) -> tensor<61x384xbf16>
    %10 = ttir.empty() : tensor<1x61x6x64xbf16>
    %11 = "ttir.reshape"(%9, %10) <{shape = [1 : i32, 61 : i32, 6 : i32, 64 : i32]}> : (tensor<61x384xbf16>, tensor<1x61x6x64xbf16>) -> tensor<1x61x6x64xbf16>
    %12 = ttir.empty() : tensor<1x6x64x61xbf16>
    %13 = "ttir.permute"(%11, %12) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x61x6x64xbf16>, tensor<1x6x64x61xbf16>) -> tensor<1x6x64x61xbf16>

    // Value projection
    %14 = ttir.empty() : tensor<61x384xbf16>
    %15 = "ttir.matmul"(%1, %arg3, %14) <{transpose_a = false, transpose_b = true}> : (tensor<61x512xbf16>, tensor<384x512xbf16>, tensor<61x384xbf16>) -> tensor<61x384xbf16>
    %16 = ttir.empty() : tensor<1x61x6x64xbf16>
    %17 = "ttir.reshape"(%15, %16) <{shape = [1 : i32, 61 : i32, 6 : i32, 64 : i32]}> : (tensor<61x384xbf16>, tensor<1x61x6x64xbf16>) -> tensor<1x61x6x64xbf16>
    %18 = ttir.empty() : tensor<1x6x61x64xbf16>
    %19 = "ttir.permute"(%17, %18) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x61x6x64xbf16>, tensor<1x6x61x64xbf16>) -> tensor<1x6x61x64xbf16>

    return %7, %13, %19 : tensor<1x61x64x6xbf16>, tensor<1x6x64x61xbf16>, tensor<1x6x61x64xbf16>
  }
}

module {
  func.func @inconsistent_head_dims_1(%arg0: tensor<1x32x4096xbf16>, // input [batch, seq, hidden]
                             %arg1: tensor<4096x4096xbf16>, // query weight
                             %arg2: tensor<4096x4096xbf16>, // key weight
                             %arg3: tensor<4096x4096xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x128x32x32xbf16>, tensor<1x32x32x128xbf16>)
  {

    // CHECK-NOT: ttnn.split_query_key_value_and_split_heads

    %0 = ttir.empty() : tensor<32x4096xbf16>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [32 : i32, 4096 : i32]}> : (tensor<1x32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>

    // Query head

    %2 = ttir.empty() : tensor<4096x4096xbf16>
    %3 = "ttir.permute"(%arg1, %2) <{permutation = array<i64: 1, 0>}> : (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>

    %4 = "ttir.dot_general"(%1, %3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %5 = ttir.empty() : tensor<1x32x32x128xbf16>
    %6 = "ttir.reshape"(%4, %5) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %7 = ttir.empty() : tensor<1x32x32x128xbf16>
    %8 = "ttir.permute"(%6, %7) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>

    // Key head

    %9 = ttir.empty() : tensor<4096x4096xbf16>
    %10 = "ttir.permute"(%arg2, %9) <{permutation = array<i64: 1, 0>}> : (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>

    %11 = "ttir.dot_general"(%1, %10) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %12 = ttir.empty() : tensor<1x32x32x128xbf16>
    %13 = "ttir.reshape"(%11, %12) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %14 = ttir.empty() : tensor<1x128x32x32xbf16>
    %15 = "ttir.permute"(%13, %14) <{permutation = array<i64: 0, 3, 2, 1>}> : (tensor<1x32x32x128xbf16>, tensor<1x128x32x32xbf16>) -> tensor<1x128x32x32xbf16>

    // Value head

    %16 = ttir.empty() : tensor<4096x4096xbf16>
    %17 = "ttir.permute"(%arg3, %16) <{permutation = array<i64: 1, 0>}> : (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>

    %18 = "ttir.dot_general"(%1, %17) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %19 = ttir.empty() : tensor<1x32x32x128xbf16>
    %20 = "ttir.reshape"(%18, %19) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %21 = ttir.empty() : tensor<1x32x32x128xbf16>
    %22 = "ttir.permute"(%20, %21) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>

    return %8, %15, %22 : tensor<1x32x32x128xbf16>, tensor<1x128x32x32xbf16>, tensor<1x32x32x128xbf16>
  }
}

module {
  func.func @incorrect_permutation(%arg0: tensor<1x32x4096xbf16>, %arg1: tensor<4096x4096xbf16>, %arg2: tensor<4096x4096xbf16>, %arg3: tensor<4096x4096xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) {

    %0 = ttir.empty() : tensor<32x4096xbf16>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [32 : i32, 4096 : i32]}> : (tensor<1x32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %2 = ttir.empty() : tensor<32x4096xbf16>

    // Query projection.
    %3 = "ttir.matmul"(%1, %arg1, %2) <{transpose_a = false, transpose_b = true}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %4 = ttir.empty() : tensor<1x32x32x128xbf16>
    %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %6 = ttir.empty() : tensor<1x32x32x128xbf16>
    %7 = "ttir.permute"(%5, %6) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %8 = ttir.empty() : tensor<32x4096xbf16>

    // Key projection.
    %9 = "ttir.matmul"(%1, %arg2, %8) <{transpose_a = false, transpose_b = true}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %10 = ttir.empty() : tensor<1x32x32x128xbf16>
    %11 = "ttir.reshape"(%9, %10) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %12 = ttir.empty() : tensor<1x32x32x128xbf16>
    %13 = "ttir.permute"(%11, %12) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %14 = ttir.empty() : tensor<32x4096xbf16>

    // Value projection with incorrect permutation.
    %15 = "ttir.matmul"(%1, %arg3, %14) <{transpose_a = false, transpose_b = true}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %16 = ttir.empty() : tensor<1x32x32x128xbf16>
    %17 = "ttir.reshape"(%15, %16) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %18 = ttir.empty() : tensor<1x32x32x128xbf16>
    %19 = "ttir.permute"(%17, %18) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    return %7, %13, %19 : tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>
  }
}
