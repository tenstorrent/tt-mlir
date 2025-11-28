// RUN: ttmlir-opt -ttir-to-ttir-decomposition -ttir-implicit-broadcast-fold -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @falcon_qkv_projection(%arg0: tensor<1x10x2048xbf16>,  // input
                                    %arg1: tensor<2048x2048xbf16>,  // query weight
                                    %arg2: tensor<1024x2048xbf16>,  // key weight
                                    %arg3: tensor<1024x2048xbf16>)  // value weight
                                    -> (tensor<1x8x10x256xbf16>, tensor<1x4x10x256xbf16>, tensor<1x4x10x256xbf16>) {

    // CHECK: func.func @falcon_qkv_projection

    // Concatenate KV weights:
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 1 : si32}>
    // CHECK-SAME: (tensor<2048x1024xbf16>, tensor<2048x1024xbf16>) -> tensor<2048x2048xbf16>

    // Matmul KV weight concat with input to get K and V projections together:
    // CHECK: "ttir.matmul"
    // CHECK-SAME: <{transpose_a = false, transpose_b = false}>
    // CHECK-SAME: (tensor<10x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<10x2048xbf16>

    // Reshape Q projection:
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 10 : i32, 2048 : i32]}>
    // CHECK-SAME: (tensor<10x2048xbf16>) -> tensor<1x10x2048xbf16>

    // Reshape combined K and V projections:
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 10 : i32, 2048 : i32]}>
    // CHECK-SAME: (tensor<10x2048xbf16>) -> tensor<1x10x2048xbf16>

    // Split Q,K,V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 8 : ui32, num_kv_heads = 4 : ui32, transpose_key = false}>
    // CHECK-SAME: (tensor<1x10x2048xbf16>, tensor<1x10x2048xbf16>) -> (tensor<1x8x10x256xbf16>, tensor<1x4x10x256xbf16>, tensor<1x4x10x256xbf16>)

    // Reshape input to 2D for matmul
    %0 = "ttir.reshape"(%arg0) <{shape = [10 : i32, 2048 : i32]}> : (tensor<1x10x2048xbf16>) -> tensor<10x2048xbf16>

    // Query projection
    %1 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 2048 : i32, 2048 : i32]}> : (tensor<2048x2048xbf16>) -> tensor<1x2048x2048xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [2048 : i32, 2048 : i32]}> : (tensor<1x2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %3 = "ttir.permute"(%2) <{permutation = array<i64: 1, 0>}> : (tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %4 = "ttir.dot_general"(%0, %3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<10x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<10x2048xbf16>

    // Reshape and split query heads: [10, 2048] -> [1, 10, 8, 256] -> [1, 8, 10, 256]
    %5 = "ttir.reshape"(%4) <{shape = [1 : i32, 10 : i32, 8 : i32, 256 : i32]}> : (tensor<10x2048xbf16>) -> tensor<1x10x8x256xbf16>
    %6 = "ttir.permute"(%5) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x10x8x256xbf16>) -> tensor<1x8x10x256xbf16>

    // Key projection
    %7 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 1024 : i32, 2048 : i32]}> : (tensor<1024x2048xbf16>) -> tensor<1x1024x2048xbf16>
    %8 = "ttir.reshape"(%7) <{shape = [1024 : i32, 2048 : i32]}> : (tensor<1x1024x2048xbf16>) -> tensor<1024x2048xbf16>
    %9 = "ttir.permute"(%8) <{permutation = array<i64: 1, 0>}> : (tensor<1024x2048xbf16>) -> tensor<2048x1024xbf16>
    %10 = "ttir.dot_general"(%0, %9) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<10x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<10x1024xbf16>

    // Reshape and split key heads: [10, 1024] -> [1, 10, 4, 256] -> [1, 4, 10, 256]
    %11 = "ttir.reshape"(%10) <{shape = [1 : i32, 10 : i32, 4 : i32, 256 : i32]}> : (tensor<10x1024xbf16>) -> tensor<1x10x4x256xbf16>
    %12 = "ttir.permute"(%11) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x10x4x256xbf16>) -> tensor<1x4x10x256xbf16>

    // Value projection
    %13 = "ttir.reshape"(%arg3) <{shape = [1 : i32, 1024 : i32, 2048 : i32]}> : (tensor<1024x2048xbf16>) -> tensor<1x1024x2048xbf16>
    %14 = "ttir.reshape"(%13) <{shape = [1024 : i32, 2048 : i32]}> : (tensor<1x1024x2048xbf16>) -> tensor<1024x2048xbf16>
    %15 = "ttir.permute"(%14) <{permutation = array<i64: 1, 0>}> : (tensor<1024x2048xbf16>) -> tensor<2048x1024xbf16>
    %16 = "ttir.dot_general"(%0, %15) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<10x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<10x1024xbf16>

    // Reshape and split value heads: [10, 1024] -> [1, 10, 4, 256] -> [1, 4, 10, 256]
    %17 = "ttir.reshape"(%16) <{shape = [1 : i32, 10 : i32, 4 : i32, 256 : i32]}> : (tensor<10x1024xbf16>) -> tensor<1x10x4x256xbf16>
    %18 = "ttir.permute"(%17) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x10x4x256xbf16>) -> tensor<1x4x10x256xbf16>

    return %6, %12, %18 : tensor<1x8x10x256xbf16>, tensor<1x4x10x256xbf16>, tensor<1x4x10x256xbf16>
  }
}
module {
  func.func @llama_3.2(%arg0: tensor<3072x3072xbf16>, %arg1: tensor<1x1024x3072xbf16>, %arg2: tensor<1024x3072xbf16>, %arg3: tensor<1024x3072xbf16>) -> (tensor<1x24x1024x128xbf16>, tensor<1x8x1024x128xbf16>, tensor<1x8x1024x128xbf16>) {
    // CHECK: func.func @llama_3.2

    // Concatenate KV weights:
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 1 : si32}>
    // CHECK-SAME: (tensor<3072x1024xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x2048xbf16>

    // Matmul KV weight concat with input to get K and V projections together:
    // CHECK: "ttir.matmul"
    // CHECK-SAME: <{transpose_a = false, transpose_b = false}>
    // CHECK-SAME: (tensor<1024x3072xbf16>, tensor<3072x2048xbf16>) -> tensor<1024x2048xbf16>

    // Reshape Q projection:
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 1024 : i32, 3072 : i32]}>
    // CHECK-SAME: (tensor<1024x3072xbf16>) -> tensor<1x1024x3072xbf16>

    // Reshape combined K and V projections:
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 1024 : i32, 2048 : i32]}>
    // CHECK-SAME: (tensor<1024x2048xbf16>) -> tensor<1x1024x2048xbf16>

    // Split Q,K,V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 24 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}>
    // CHECK-SAME: (tensor<1x1024x3072xbf16>, tensor<1x1024x2048xbf16>) -> (tensor<1x24x1024x128xbf16>, tensor<1x8x1024x128xbf16>, tensor<1x8x1024x128xbf16>)


    %0 = "ttir.reshape"(%arg1) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %3 = "ttir.permute"(%2) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %4 = "ttir.dot_general"(%0, %3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1024x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<1024x3072xbf16>
    %5 = "ttir.reshape"(%4) <{shape = [1 : i32, 1024 : i32, 24 : i32, 128 : i32]}> : (tensor<1024x3072xbf16>) -> tensor<1x1024x24x128xbf16>
    %6 = "ttir.permute"(%5) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x1024x24x128xbf16>) -> tensor<1x24x1024x128xbf16>
    %7 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %8 = "ttir.reshape"(%7) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %9 = "ttir.permute"(%8) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>) -> tensor<3072x1024xbf16>
    %10 = "ttir.dot_general"(%0, %9) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1024x1024xbf16>
    %11 = "ttir.reshape"(%10) <{shape = [1 : i32, 1024 : i32, 8 : i32, 128 : i32]}> : (tensor<1024x1024xbf16>) -> tensor<1x1024x8x128xbf16>
    %12 = "ttir.permute"(%11) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x1024x8x128xbf16>) -> tensor<1x8x1024x128xbf16>
    %13 = "ttir.reshape"(%arg3) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %14 = "ttir.reshape"(%13) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %15 = "ttir.permute"(%14) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>) -> tensor<3072x1024xbf16>
    %16 = "ttir.dot_general"(%0, %15) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1024x1024xbf16>
    %17 = "ttir.reshape"(%16) <{shape = [1 : i32, 1024 : i32, 8 : i32, 128 : i32]}> : (tensor<1024x1024xbf16>) -> tensor<1x1024x8x128xbf16>
    %18 = "ttir.permute"(%17) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x1024x8x128xbf16>) -> tensor<1x8x1024x128xbf16>
    return %6, %12, %18 : tensor<1x24x1024x128xbf16>, tensor<1x8x1024x128xbf16>, tensor<1x8x1024x128xbf16>
  }
}

module {
  func.func @llama_3_8b(%arg0: tensor<32x32x3072xbf16>,    // input (reshaped from 1024x3072)
                        %arg1: tensor<3072x3072xbf16>,    // query weight (24 heads)
                        %arg2: tensor<1024x3072xbf16>,    // key weight (8 heads)
                        %arg3: tensor<1024x3072xbf16>)    // value weight (8 heads)
                        -> (tensor<32x24x32x128xbf16>, tensor<32x8x32x128xbf16>, tensor<32x8x32x128xbf16>) {

    // It is ordered as K-Q-V. This could happen due to torch.compile.
    // We still need to match it to Q-K-V and replace with fused op.

    // CHECK: func.func @llama_3_8b
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 1 : si32}>
    // CHECK-SAME: (tensor<3072x1024xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x2048xbf16>

    // CHECK: "ttir.matmul"
    // CHECK-SAME: <{transpose_a = false, transpose_b = false}>
    // CHECK-SAME: (tensor<1024x3072xbf16>, tensor<3072x2048xbf16>) -> tensor<1024x2048xbf16>

    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [32 : i32, 32 : i32, 3072 : i32]}>
    // CHECK-SAME: (tensor<1024x3072xbf16>) -> tensor<32x32x3072xbf16>

    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [32 : i32, 32 : i32, 2048 : i32]}>
    // CHECK-SAME: (tensor<1024x2048xbf16>) -> tensor<32x32x2048xbf16>

    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 24 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}>
    // CHECK-SAME: (tensor<32x32x3072xbf16>, tensor<32x32x2048xbf16>) -> (tensor<32x24x32x128xbf16>, tensor<32x8x32x128xbf16>, tensor<32x8x32x128xbf16>)

    // Reshape input to 2D for matrix multiplication
    %0 = "ttir.reshape"(%arg0) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<32x32x3072xbf16>) -> tensor<1024x3072xbf16>

    // Key projection (8 heads, 128 dim each = 1024 total)
    %1 = "ttir.permute"(%arg2) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>) -> tensor<3072x1024xbf16>
    %2 = "ttir.dot_general"(%0, %1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1024x1024xbf16>

    %3 = "ttir.reshape"(%2) <{shape = [32 : i32, 32 : i32, 8 : i32, 128 : i32]}> : (tensor<1024x1024xbf16>) -> tensor<32x32x8x128xbf16>
    %4 = "ttir.permute"(%3) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x32x8x128xbf16>) -> tensor<32x8x32x128xbf16>

    // Query projection (24 heads, 128 dim each = 3072 total)
    %5 = "ttir.permute"(%arg1) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6 = "ttir.dot_general"(%0, %5) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1024x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<1024x3072xbf16>

    %7 = "ttir.reshape"(%6) <{shape = [32 : i32, 32 : i32, 24 : i32, 128 : i32]}> : (tensor<1024x3072xbf16>) -> tensor<32x32x24x128xbf16>
    %8 = "ttir.permute"(%7) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x32x24x128xbf16>) -> tensor<32x24x32x128xbf16>

    // Value projection (8 heads, 128 dim each = 1024 total)
    %9 = "ttir.permute"(%arg3) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>) -> tensor<3072x1024xbf16>
    %10 = "ttir.dot_general"(%0, %9) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1024x1024xbf16>

    %11 = "ttir.reshape"(%10) <{shape = [32 : i32, 32 : i32, 8 : i32, 128 : i32]}> : (tensor<1024x1024xbf16>) -> tensor<32x32x8x128xbf16>
    %12 = "ttir.permute"(%11) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x32x8x128xbf16>) -> tensor<32x8x32x128xbf16>

    return %8, %4, %12 : tensor<32x24x32x128xbf16>, tensor<32x8x32x128xbf16>, tensor<32x8x32x128xbf16>
  }
}

module {
  func.func @llama_3_8b_transpose_b(%arg0: tensor<32x32x3072xbf16>, %arg1: tensor<3072x3072xbf16>, %arg2: tensor<1024x3072xbf16>, %arg3: tensor<1024x3072xbf16>) -> (tensor<32x24x32x128xbf16>, tensor<32x8x32x128xbf16>, tensor<32x8x32x128xbf16>) {

    // CHECK: func.func @llama_3_8b_transpose_b
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 0 : si32}>
    // CHECK-SAME: (tensor<1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<2048x3072xbf16>

    // CHECK: "ttir.matmul"
    // CHECK-SAME: <{transpose_a = false, transpose_b = true}>
    // CHECK-SAME: (tensor<1024x3072xbf16>, tensor<2048x3072xbf16>) -> tensor<1024x2048xbf16>

    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [32 : i32, 32 : i32, 3072 : i32]}>
    // CHECK-SAME: (tensor<1024x3072xbf16>) -> tensor<32x32x3072xbf16>

    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [32 : i32, 32 : i32, 2048 : i32]}>
    // CHECK-SAME: (tensor<1024x2048xbf16>) -> tensor<32x32x2048xbf16>

    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 24 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}>
    // CHECK-SAME: (tensor<32x32x3072xbf16>, tensor<32x32x2048xbf16>) -> (tensor<32x24x32x128xbf16>, tensor<32x8x32x128xbf16>, tensor<32x8x32x128xbf16>)

    %0 = "ttir.reshape"(%arg0) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<32x32x3072xbf16>) -> tensor<1024x3072xbf16>
    %1 = "ttir.matmul"(%0, %arg2) <{transpose_a = false, transpose_b = true}> : (tensor<1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x1024xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 32 : i32, 8 : i32, 128 : i32]}> : (tensor<1024x1024xbf16>) -> tensor<32x32x8x128xbf16>
    %3 = "ttir.permute"(%2) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x32x8x128xbf16>) -> tensor<32x8x32x128xbf16>
    %4 = "ttir.matmul"(%0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<1024x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<1024x3072xbf16>
    %5 = "ttir.reshape"(%4) <{shape = [32 : i32, 32 : i32, 24 : i32, 128 : i32]}> : (tensor<1024x3072xbf16>) -> tensor<32x32x24x128xbf16>
    %6 = "ttir.permute"(%5) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x32x24x128xbf16>) -> tensor<32x24x32x128xbf16>
    %7 = "ttir.matmul"(%0, %arg3) <{transpose_a = false, transpose_b = true}> : (tensor<1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x1024xbf16>
    %8 = "ttir.reshape"(%7) <{shape = [32 : i32, 32 : i32, 8 : i32, 128 : i32]}> : (tensor<1024x1024xbf16>) -> tensor<32x32x8x128xbf16>
    %9 = "ttir.permute"(%8) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x32x8x128xbf16>) -> tensor<32x8x32x128xbf16>
    return %6, %3, %9 : tensor<32x24x32x128xbf16>, tensor<32x8x32x128xbf16>, tensor<32x8x32x128xbf16>
  }
}
