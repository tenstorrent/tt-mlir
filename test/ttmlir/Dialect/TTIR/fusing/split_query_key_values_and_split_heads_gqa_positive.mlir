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
    // CHECK-SAME: (tensor<2048x1024xbf16>, tensor<2048x1024xbf16>, tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>

    // Matmul KV weight concat with input to get K and V projections together:
    // CHECK: "ttir.matmul"
    // CHECK-SAME: <{transpose_a = false, transpose_b = false}>
    // CHECK-SAME: (tensor<10x2048xbf16>, tensor<2048x2048xbf16>, tensor<10x2048xbf16>) -> tensor<10x2048xbf16>

    // Reshape Q projection:
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 10 : i32, 2048 : i32]}>
    // CHECK-SAME: (tensor<10x2048xbf16>, tensor<1x10x2048xbf16>) -> tensor<1x10x2048xbf16>

    // Reshape combined K and V projections:
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 10 : i32, 2048 : i32]}>
    // CHECK-SAME: (tensor<10x2048xbf16>, tensor<1x10x2048xbf16>) -> tensor<1x10x2048xbf16>

    // Split Q,K,V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 8 : ui32, num_kv_heads = 4 : ui32, transpose_key = false}>
    // CHECK-SAME: (tensor<1x10x2048xbf16>, tensor<1x10x2048xbf16>, tensor<1x8x10x256xbf16>, tensor<1x4x10x256xbf16>, tensor<1x4x10x256xbf16>) -> (tensor<1x8x10x256xbf16>, tensor<1x4x10x256xbf16>, tensor<1x4x10x256xbf16>)

    // Reshape input to 2D for matmul
    %0 = ttir.empty() : tensor<10x2048xbf16>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [10 : i32, 2048 : i32]}> : (tensor<1x10x2048xbf16>, tensor<10x2048xbf16>) -> tensor<10x2048xbf16>

    // Query projection
    %2 = ttir.empty() : tensor<1x2048x2048xbf16>
    %3 = "ttir.reshape"(%arg1, %2) <{shape = [1 : i32, 2048 : i32, 2048 : i32]}> : (tensor<2048x2048xbf16>, tensor<1x2048x2048xbf16>) -> tensor<1x2048x2048xbf16>
    %4 = ttir.empty() : tensor<2048x2048xbf16>
    %5 = "ttir.reshape"(%3, %4) <{shape = [2048 : i32, 2048 : i32]}> : (tensor<1x2048x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %6 = ttir.empty() : tensor<2048x2048xbf16>
    %7 = "ttir.permute"(%5, %6) <{permutation = array<i64: 1, 0>}> : (tensor<2048x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %8 = "ttir.dot_general"(%1, %7) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<10x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<10x2048xbf16>

    // Reshape and split query heads: [10, 2048] -> [1, 10, 8, 256] -> [1, 8, 10, 256]
    %9 = ttir.empty() : tensor<1x10x8x256xbf16>
    %10 = "ttir.reshape"(%8, %9) <{shape = [1 : i32, 10 : i32, 8 : i32, 256 : i32]}> : (tensor<10x2048xbf16>, tensor<1x10x8x256xbf16>) -> tensor<1x10x8x256xbf16>
    %11 = ttir.empty() : tensor<1x8x10x256xbf16>
    %12 = "ttir.permute"(%10, %11) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x10x8x256xbf16>, tensor<1x8x10x256xbf16>) -> tensor<1x8x10x256xbf16>

    // Key projection
    %13 = ttir.empty() : tensor<1x1024x2048xbf16>
    %14 = "ttir.reshape"(%arg2, %13) <{shape = [1 : i32, 1024 : i32, 2048 : i32]}> : (tensor<1024x2048xbf16>, tensor<1x1024x2048xbf16>) -> tensor<1x1024x2048xbf16>
    %15 = ttir.empty() : tensor<1024x2048xbf16>
    %16 = "ttir.reshape"(%14, %15) <{shape = [1024 : i32, 2048 : i32]}> : (tensor<1x1024x2048xbf16>, tensor<1024x2048xbf16>) -> tensor<1024x2048xbf16>
    %17 = ttir.empty() : tensor<2048x1024xbf16>
    %18 = "ttir.permute"(%16, %17) <{permutation = array<i64: 1, 0>}> : (tensor<1024x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<2048x1024xbf16>
    %19 = "ttir.dot_general"(%1, %18) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<10x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<10x1024xbf16>

    // Reshape and split key heads: [10, 1024] -> [1, 10, 4, 256] -> [1, 4, 10, 256]
    %20 = ttir.empty() : tensor<1x10x4x256xbf16>
    %21 = "ttir.reshape"(%19, %20) <{shape = [1 : i32, 10 : i32, 4 : i32, 256 : i32]}> : (tensor<10x1024xbf16>, tensor<1x10x4x256xbf16>) -> tensor<1x10x4x256xbf16>
    %22 = ttir.empty() : tensor<1x4x10x256xbf16>
    %23 = "ttir.permute"(%21, %22) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x10x4x256xbf16>, tensor<1x4x10x256xbf16>) -> tensor<1x4x10x256xbf16>

    // Value projection
    %24 = ttir.empty() : tensor<1x1024x2048xbf16>
    %25 = "ttir.reshape"(%arg3, %24) <{shape = [1 : i32, 1024 : i32, 2048 : i32]}> : (tensor<1024x2048xbf16>, tensor<1x1024x2048xbf16>) -> tensor<1x1024x2048xbf16>
    %26 = ttir.empty() : tensor<1024x2048xbf16>
    %27 = "ttir.reshape"(%25, %26) <{shape = [1024 : i32, 2048 : i32]}> : (tensor<1x1024x2048xbf16>, tensor<1024x2048xbf16>) -> tensor<1024x2048xbf16>
    %28 = ttir.empty() : tensor<2048x1024xbf16>
    %29 = "ttir.permute"(%27, %28) <{permutation = array<i64: 1, 0>}> : (tensor<1024x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<2048x1024xbf16>
    %30 = "ttir.dot_general"(%1, %29) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<10x2048xbf16>, tensor<2048x1024xbf16>) -> tensor<10x1024xbf16>

    // Reshape and split value heads: [10, 1024] -> [1, 10, 4, 256] -> [1, 4, 10, 256]
    %31 = ttir.empty() : tensor<1x10x4x256xbf16>
    %32 = "ttir.reshape"(%30, %31) <{shape = [1 : i32, 10 : i32, 4 : i32, 256 : i32]}> : (tensor<10x1024xbf16>, tensor<1x10x4x256xbf16>) -> tensor<1x10x4x256xbf16>
    %33 = ttir.empty() : tensor<1x4x10x256xbf16>
    %34 = "ttir.permute"(%32, %33) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x10x4x256xbf16>, tensor<1x4x10x256xbf16>) -> tensor<1x4x10x256xbf16>

    return %12, %23, %34 : tensor<1x8x10x256xbf16>, tensor<1x4x10x256xbf16>, tensor<1x4x10x256xbf16>
  }
}
module {
  func.func @llama_3.2(%arg0: tensor<3072x3072xbf16>, %arg1: tensor<1x1024x3072xbf16>, %arg2: tensor<1024x3072xbf16>, %arg3: tensor<1024x3072xbf16>) -> (tensor<1x24x1024x128xbf16>, tensor<1x8x1024x128xbf16>, tensor<1x8x1024x128xbf16>) {
    // CHECK: func.func @llama_3.2

    // Concatenate KV weights:
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 1 : si32}>
    // CHECK-SAME: (tensor<3072x1024xbf16>, tensor<3072x1024xbf16>, tensor<3072x2048xbf16>) -> tensor<3072x2048xbf16>

    // Matmul KV weight concat with input to get K and V projections together:
    // CHECK: "ttir.matmul"
    // CHECK-SAME: <{transpose_a = false, transpose_b = false}>
    // CHECK-SAME: (tensor<1024x3072xbf16>, tensor<3072x2048xbf16>, tensor<1024x2048xbf16>) -> tensor<1024x2048xbf16>

    // Reshape Q projection:
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 1024 : i32, 3072 : i32]}>
    // CHECK-SAME: (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>

    // Reshape combined K and V projections:
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 1024 : i32, 2048 : i32]}>
    // CHECK-SAME: (tensor<1024x2048xbf16>, tensor<1x1024x2048xbf16>) -> tensor<1x1024x2048xbf16>

    // Split Q,K,V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 24 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}>
    // CHECK-SAME: (tensor<1x1024x3072xbf16>, tensor<1x1024x2048xbf16>, tensor<1x24x1024x128xbf16>, tensor<1x8x1024x128xbf16>, tensor<1x8x1024x128xbf16>) -> (tensor<1x24x1024x128xbf16>, tensor<1x8x1024x128xbf16>, tensor<1x8x1024x128xbf16>)


    %0 = ttir.empty() : tensor<1024x3072xbf16>
    %1 = "ttir.reshape"(%arg1, %0) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %2 = ttir.empty() : tensor<1x3072x3072xbf16>
    %3 = "ttir.reshape"(%arg0, %2) <{shape = [1 : i32, 3072 : i32, 3072 : i32]}> : (tensor<3072x3072xbf16>, tensor<1x3072x3072xbf16>) -> tensor<1x3072x3072xbf16>
    %4 = ttir.empty() : tensor<3072x3072xbf16>
    %5 = "ttir.reshape"(%3, %4) <{shape = [3072 : i32, 3072 : i32]}> : (tensor<1x3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %6 = ttir.empty() : tensor<3072x3072xbf16>
    %7 = "ttir.permute"(%5, %6) <{permutation = array<i64: 1, 0>}> : (tensor<3072x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<3072x3072xbf16>
    %8 = "ttir.dot_general"(%1, %7) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1024x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<1024x3072xbf16>
    %9 = ttir.empty() : tensor<1x1024x24x128xbf16>
    %10 = "ttir.reshape"(%8, %9) <{shape = [1 : i32, 1024 : i32, 24 : i32, 128 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x24x128xbf16>) -> tensor<1x1024x24x128xbf16>
    %11 = ttir.empty() : tensor<1x24x1024x128xbf16>
    %12 = "ttir.permute"(%10, %11) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x1024x24x128xbf16>, tensor<1x24x1024x128xbf16>) -> tensor<1x24x1024x128xbf16>
    %13 = ttir.empty() : tensor<1x1024x3072xbf16>
    %14 = "ttir.reshape"(%arg2, %13) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %15 = ttir.empty() : tensor<1024x3072xbf16>
    %16 = "ttir.reshape"(%14, %15) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %17 = ttir.empty() : tensor<3072x1024xbf16>
    %18 = "ttir.permute"(%16, %17) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %19 = "ttir.dot_general"(%1, %18) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1024x1024xbf16>
    %20 = ttir.empty() : tensor<1x1024x8x128xbf16>
    %21 = "ttir.reshape"(%19, %20) <{shape = [1 : i32, 1024 : i32, 8 : i32, 128 : i32]}> : (tensor<1024x1024xbf16>, tensor<1x1024x8x128xbf16>) -> tensor<1x1024x8x128xbf16>
    %22 = ttir.empty() : tensor<1x8x1024x128xbf16>
    %23 = "ttir.permute"(%21, %22) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x1024x8x128xbf16>, tensor<1x8x1024x128xbf16>) -> tensor<1x8x1024x128xbf16>
    %24 = ttir.empty() : tensor<1x1024x3072xbf16>
    %25 = "ttir.reshape"(%arg3, %24) <{shape = [1 : i32, 1024 : i32, 3072 : i32]}> : (tensor<1024x3072xbf16>, tensor<1x1024x3072xbf16>) -> tensor<1x1024x3072xbf16>
    %26 = ttir.empty() : tensor<1024x3072xbf16>
    %27 = "ttir.reshape"(%25, %26) <{shape = [1024 : i32, 3072 : i32]}> : (tensor<1x1024x3072xbf16>, tensor<1024x3072xbf16>) -> tensor<1024x3072xbf16>
    %28 = ttir.empty() : tensor<3072x1024xbf16>
    %29 = "ttir.permute"(%27, %28) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %30 = "ttir.dot_general"(%1, %29) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<1024x1024xbf16>
    %31 = ttir.empty() : tensor<1x1024x8x128xbf16>
    %32 = "ttir.reshape"(%30, %31) <{shape = [1 : i32, 1024 : i32, 8 : i32, 128 : i32]}> : (tensor<1024x1024xbf16>, tensor<1x1024x8x128xbf16>) -> tensor<1x1024x8x128xbf16>
    %33 = ttir.empty() : tensor<1x8x1024x128xbf16>
    %34 = "ttir.permute"(%32, %33) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x1024x8x128xbf16>, tensor<1x8x1024x128xbf16>) -> tensor<1x8x1024x128xbf16>
    return %12, %23, %34 : tensor<1x24x1024x128xbf16>, tensor<1x8x1024x128xbf16>, tensor<1x8x1024x128xbf16>
  }
}
