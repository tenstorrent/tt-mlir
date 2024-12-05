// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module {
  func.func @main(%arg0: tensor<5xi32>, %arg1: tensor<1x32x64x64xbf16>, %arg2: tensor<1x32x64x64xbf16>, %arg3: tensor<1x32x5x64xbf16>, %arg4: tensor<1x32x5x64xbf16>) -> tensor<1x32x64x64xbf16> {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %c_0 = stablehlo.constant dense<1> : tensor<1xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<32xi64>
    %c_2 = stablehlo.constant dense<1> : tensor<32xi64>
    %c_3 = stablehlo.constant dense<32> : tensor<i64>
    %c_4 = stablehlo.constant dense<0> : tensor<64xi64>
    %c_5 = stablehlo.constant dense<1> : tensor<64xi64>
    %cst = arith.constant dense<1> : tensor<1xi64>
    %c_6 = stablehlo.constant dense<1> : tensor<i64>
    %c_7 = stablehlo.constant dense<64> : tensor<i64>
    %0 = stablehlo.convert %c_6 : (tensor<i64>) -> tensor<f64>
    %1 = stablehlo.convert %c_7 : (tensor<i64>) -> tensor<f64>
    %2 = stablehlo.divide %1, %0 : tensor<f64>
    %3 = stablehlo.ceil %2 : tensor<f64>
    %4 = stablehlo.convert %3 : (tensor<f64>) -> tensor<i64>
    %5 = stablehlo.reshape %4 : (tensor<i64>) -> tensor<1xi64>
    %6 = stablehlo.dynamic_iota %5, dim = 0 : (tensor<1xi64>) -> tensor<64xi64>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<64xi64>) -> tensor<64xi64>
    %8 = stablehlo.multiply %7, %c_5 : tensor<64xi64>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<64xi64>) -> tensor<64xi64>
    %10 = stablehlo.add %9, %c_4 : tensor<64xi64>
    %11 = stablehlo.reshape %arg0 : (tensor<5xi32>) -> tensor<5x1xi32>
    %12 = stablehlo.convert %c_3 : (tensor<i64>) -> tensor<f64>
    %13 = stablehlo.divide %12, %0 : tensor<f64>
    %14 = stablehlo.ceil %13 : tensor<f64>
    %15 = stablehlo.convert %14 : (tensor<f64>) -> tensor<i64>
    %16 = stablehlo.reshape %15 : (tensor<i64>) -> tensor<1xi64>
    %17 = stablehlo.dynamic_iota %16, dim = 0 : (tensor<1xi64>) -> tensor<32xi64>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<32xi64>) -> tensor<32xi64>
    %19 = stablehlo.multiply %18, %c_2 : tensor<32xi64>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<32xi64>) -> tensor<32xi64>
    %21 = stablehlo.add %20, %c_1 : tensor<32xi64>
    %22 = stablehlo.reshape %21 : (tensor<32xi64>) -> tensor<32x1xi64>
    %23 = stablehlo.reshape %22 : (tensor<32x1xi64>) -> tensor<32x1x1xi64>
    %24 = stablehlo.convert %c_6 : (tensor<i64>) -> tensor<f64>
    %25 = stablehlo.divide %24, %0 : tensor<f64>
    %26 = stablehlo.ceil %25 : tensor<f64>
    %27 = stablehlo.convert %26 : (tensor<f64>) -> tensor<i64>
    %28 = stablehlo.reshape %27 : (tensor<i64>) -> tensor<1xi64>
    %29 = stablehlo.dynamic_iota %28, dim = 0 : (tensor<1xi64>) -> tensor<1xi64>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0] : (tensor<1xi64>) -> tensor<1xi64>
    %31 = stablehlo.multiply %30, %c_0 : tensor<1xi64>
    %32 = stablehlo.broadcast_in_dim %31, dims = [0] : (tensor<1xi64>) -> tensor<1xi64>
    %33 = stablehlo.add %32, %c : tensor<1xi64>
    %34 = stablehlo.reshape %33 : (tensor<1xi64>) -> tensor<1x1xi64>
    %35 = stablehlo.reshape %34 : (tensor<1x1xi64>) -> tensor<1x1x1xi64>
    %36 = stablehlo.reshape %35 : (tensor<1x1x1xi64>) -> tensor<1x1x1x1xi64>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xi64>) -> tensor<1x32x5x64xi64>
    %38 = stablehlo.reshape %37 : (tensor<1x32x5x64xi64>) -> tensor<1x32x5x64x1xi64>
    %39 = stablehlo.broadcast_in_dim %23, dims = [1, 2, 3] : (tensor<32x1x1xi64>) -> tensor<1x32x5x64xi64>
    %40 = stablehlo.reshape %39 : (tensor<1x32x5x64xi64>) -> tensor<1x32x5x64x1xi64>
    %41 = stablehlo.convert %11 : (tensor<5x1xi32>) -> tensor<5x1xi64>
    %42 = stablehlo.broadcast_in_dim %41, dims = [2, 3] : (tensor<5x1xi64>) -> tensor<1x32x5x64xi64>
    %43 = stablehlo.reshape %42 : (tensor<1x32x5x64xi64>) -> tensor<1x32x5x64x1xi64>
    %44 = stablehlo.broadcast_in_dim %10, dims = [3] : (tensor<64xi64>) -> tensor<1x32x5x64xi64>
    %45 = stablehlo.reshape %44 : (tensor<1x32x5x64xi64>) -> tensor<1x32x5x64x1xi64>
    %46 = stablehlo.concatenate %38, %40, %43, %45, dim = 4 : (tensor<1x32x5x64x1xi64>, tensor<1x32x5x64x1xi64>, tensor<1x32x5x64x1xi64>, tensor<1x32x5x64x1xi64>) -> tensor<1x32x5x64x4xi64>
    // CHECK: ttnn.fill_cache
    %47 = "stablehlo.scatter"(%arg1, %46, %arg3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1, 2, 3], scatter_dims_to_operand_dims = [0, 1, 2, 3], index_vector_dim = 4>, unique_indices = false}> ({
    ^bb0(%arg5: tensor<bf16>, %arg6: tensor<bf16>):
      stablehlo.return %arg6 : tensor<bf16>
    }) : (tensor<1x32x64x64xbf16>, tensor<1x32x5x64x4xi64>, tensor<1x32x5x64xbf16>) -> tensor<1x32x64x64xbf16>
    %48 = stablehlo.convert %cst : (tensor<1xi64>) -> tensor<1xbf16>
    %49 = stablehlo.reshape %48 : (tensor<1xbf16>) -> tensor<bf16>
    %50 = stablehlo.broadcast_in_dim %47, dims = [0, 1, 2, 3] : (tensor<1x32x64x64xbf16>) -> tensor<1x32x64x64xbf16>
    %51 = stablehlo.broadcast_in_dim %49, dims = [] : (tensor<bf16>) -> tensor<1x32x64x64xbf16>
    %52 = stablehlo.add %50, %51 : tensor<1x32x64x64xbf16>
    return %52 : tensor<1x32x64x64xbf16>
  }
}
