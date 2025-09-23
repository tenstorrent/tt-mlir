// RUN: ttmlir-opt %s --ttir-fusing


// From customer model - bge
// module {
//   func.func @split_qkv_and_split_heads(%arg0: tensor<2x34x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024x1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024x1024xf32>, %arg6: tensor<1024xf32>) -> (tensor<32x34x64xf32>, tensor<32x64x34xf32>, tensor<32x34x64xf32>) {

//     %0 = ttir.empty() : tensor<68x1024xf32>
//     %1 = "ttir.reshape"(%arg0, %0) <{shape = [68 : i32, 1024 : i32]}> : (tensor<2x34x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>

//     // Query projection
//     %2 = "ttir.dot_general"(%1, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
//     %3 = ttir.empty() : tensor<1x1024xf32>
//     %4 = "ttir.reshape"(%arg2, %3) <{shape = [1 : i32, 1024 : i32]}> : (tensor<1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
//     %5 = ttir.empty() : tensor<68x1024xf32>
//     %6 = "ttir.broadcast"(%4, %5) <{broadcast_dimensions = array<i64: 68, 1>}> : (tensor<1x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
//     %7 = ttir.empty() : tensor<68x1024xf32>
//     %8 = "ttir.add"(%2, %6, %7) : (tensor<68x1024xf32>, tensor<68x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
//     %9 = ttir.empty() : tensor<2x34x16x64xf32>
//     %10 = "ttir.reshape"(%8, %9) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x16x64xf32>) -> tensor<2x34x16x64xf32>
//     %11 = ttir.empty() : tensor<2x16x34x64xf32>
//     %12 = "ttir.permute"(%10, %11) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x34x16x64xf32>, tensor<2x16x34x64xf32>) -> tensor<2x16x34x64xf32>
//     %13 = ttir.empty() : tensor<32x34x64xf32>
//     %14 = "ttir.reshape"(%12, %13) <{shape = [32 : i32, 34 : i32, 64 : i32]}> : (tensor<2x16x34x64xf32>, tensor<32x34x64xf32>) -> tensor<32x34x64xf32>

//     // Key projection
//     %15 = "ttir.dot_general"(%1, %arg3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
//     %16 = ttir.empty() : tensor<1x1024xf32>
//     %17 = "ttir.reshape"(%arg4, %16) <{shape = [1 : i32, 1024 : i32]}> : (tensor<1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
//     %18 = ttir.empty() : tensor<68x1024xf32>
//     %19 = "ttir.broadcast"(%17, %18) <{broadcast_dimensions = array<i64: 68, 1>}> : (tensor<1x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
//     %20 = ttir.empty() : tensor<68x1024xf32>
//     %21 = "ttir.add"(%15, %19, %20) : (tensor<68x1024xf32>, tensor<68x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
//     %22 = ttir.empty() : tensor<2x34x16x64xf32>
//     %23 = "ttir.reshape"(%21, %22) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x16x64xf32>) -> tensor<2x34x16x64xf32>
//     %24 = ttir.empty() : tensor<2x16x64x34xf32>
//     %25 = "ttir.permute"(%23, %24) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<2x34x16x64xf32>, tensor<2x16x64x34xf32>) -> tensor<2x16x64x34xf32>
//     %26 = ttir.empty() : tensor<32x64x34xf32>
//     %27 = "ttir.reshape"(%25, %26) <{shape = [32 : i32, 64 : i32, 34 : i32]}> : (tensor<2x16x64x34xf32>, tensor<32x64x34xf32>) -> tensor<32x64x34xf32>

//     // Value projection
//     %28 = "ttir.dot_general"(%1, %arg5) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
//     %29 = ttir.empty() : tensor<1x1024xf32>
//     %30 = "ttir.reshape"(%arg6, %29) <{shape = [1 : i32, 1024 : i32]}> : (tensor<1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
//     %31 = ttir.empty() : tensor<68x1024xf32>
//     %32 = "ttir.broadcast"(%30, %31) <{broadcast_dimensions = array<i64: 68, 1>}> : (tensor<1x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
//     %33 = ttir.empty() : tensor<68x1024xf32>
//     %34 = "ttir.add"(%28, %32, %33) : (tensor<68x1024xf32>, tensor<68x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
//     %35 = ttir.empty() : tensor<2x34x16x64xf32>
//     %36 = "ttir.reshape"(%34, %35) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x16x64xf32>) -> tensor<2x34x16x64xf32>
//     %37 = ttir.empty() : tensor<2x16x34x64xf32>
//     %38 = "ttir.permute"(%36, %37) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x34x16x64xf32>, tensor<2x16x34x64xf32>) -> tensor<2x16x34x64xf32>
//     %39 = ttir.empty() : tensor<32x34x64xf32>
//     %40 = "ttir.reshape"(%38, %39) <{shape = [32 : i32, 34 : i32, 64 : i32]}> : (tensor<2x16x34x64xf32>, tensor<32x34x64xf32>) -> tensor<32x34x64xf32>

//     return %14, %27, %40 : tensor<32x34x64xf32>, tensor<32x64x34xf32>, tensor<32x34x64xf32>
//   }
// }

module {

  func.func @split_qkv_and_split_heads_2(%arg0: tensor<1x32x3072xbf16>, %arg1: tensor<3072x3072xbf16>, %arg2: tensor<1024x3072xbf16>, %arg3: tensor<1024x3072xbf16>) -> (tensor<1x24x32x128xbf16>, tensor<1x8x32x128xbf16>, tensor<1x8x32x128xbf16>) {

    %0 = ttir.empty() : tensor<32x3072xbf16>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [32 : i32, 3072 : i32]}> : (tensor<1x32x3072xbf16>, tensor<32x3072xbf16>) -> tensor<32x3072xbf16>

    // Query projection
    %2 = "ttir.dot_general"(%1, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x3072xbf16>, tensor<3072x3072xbf16>) -> tensor<32x3072xbf16>
    %3 = ttir.empty() : tensor<1x32x3072xbf16>
    %4 = "ttir.reshape"(%2, %3) <{shape = [1 : i32, 32 : i32, 3072 : i32]}> : (tensor<32x3072xbf16>, tensor<1x32x3072xbf16>) -> tensor<1x32x3072xbf16>
    %5 = ttir.empty() : tensor<1x32x24x128xbf16>
    %6 = "ttir.reshape"(%4, %5) <{shape = [1 : i32, 32 : i32, 24 : i32, 128 : i32]}> : (tensor<1x32x3072xbf16>, tensor<1x32x24x128xbf16>) -> tensor<1x32x24x128xbf16>
    %7 = ttir.empty() : tensor<1x24x32x128xbf16>
    %8 = "ttir.permute"(%6, %7) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x24x128xbf16>, tensor<1x24x32x128xbf16>) -> tensor<1x24x32x128xbf16>

    // Key projection
    %9 = ttir.empty() : tensor<3072x1024xbf16>
    %10 = "ttir.permute"(%arg2, %9) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %11 = "ttir.dot_general"(%1, %10) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<32x1024xbf16>
    %12 = ttir.empty() : tensor<1x32x1024xbf16>
    %13 = "ttir.reshape"(%11, %12) <{shape = [1 : i32, 32 : i32, 1024 : i32]}> : (tensor<32x1024xbf16>, tensor<1x32x1024xbf16>) -> tensor<1x32x1024xbf16>
    %14 = ttir.empty() : tensor<1x32x8x128xbf16>
    %15 = "ttir.reshape"(%13, %14) <{shape = [1 : i32, 32 : i32, 8 : i32, 128 : i32]}> : (tensor<1x32x1024xbf16>, tensor<1x32x8x128xbf16>) -> tensor<1x32x8x128xbf16>
    %16 = ttir.empty() : tensor<1x8x32x128xbf16>
    %17 = "ttir.permute"(%15, %16) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x128xbf16>, tensor<1x8x32x128xbf16>) -> tensor<1x8x32x128xbf16>

    // Value projection
    %18 = ttir.empty() : tensor<3072x1024xbf16>
    %19 = "ttir.permute"(%arg3, %18) <{permutation = array<i64: 1, 0>}> : (tensor<1024x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<3072x1024xbf16>
    %20 = "ttir.dot_general"(%1, %19) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x3072xbf16>, tensor<3072x1024xbf16>) -> tensor<32x1024xbf16>
    %21 = ttir.empty() : tensor<1x32x1024xbf16>
    %22 = "ttir.reshape"(%20, %21) <{shape = [1 : i32, 32 : i32, 1024 : i32]}> : (tensor<32x1024xbf16>, tensor<1x32x1024xbf16>) -> tensor<1x32x1024xbf16>
    %23 = ttir.empty() : tensor<1x32x8x128xbf16>
    %24 = "ttir.reshape"(%22, %23) <{shape = [1 : i32, 32 : i32, 8 : i32, 128 : i32]}> : (tensor<1x32x1024xbf16>, tensor<1x32x8x128xbf16>) -> tensor<1x32x8x128xbf16>
    %25 = ttir.empty() : tensor<1x8x32x128xbf16>
    %26 = "ttir.permute"(%24, %25) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x128xbf16>, tensor<1x8x32x128xbf16>) -> tensor<1x8x32x128xbf16>

    return %8, %17, %26 : tensor<1x24x32x128xbf16>, tensor<1x8x32x128xbf16>, tensor<1x8x32x128xbf16>
  }
}
