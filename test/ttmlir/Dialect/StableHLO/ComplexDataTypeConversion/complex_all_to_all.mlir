// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-complex-data-type-conversion -o %t %s
// RUN: FileCheck %s --input-file=%t

// Shardy emits data-movement collectives (stablehlo.all_to_all) on complex
// freqs_cis when resharding RoPE under tensor parallelism. The collective must
// be converted to the unpacked real (...x2xf32) representation; previously it
// stayed complex<f32> and left an unresolved materialization (tt-xla #5313).
module {
    // Mirrors the @sdy.all_slice reshard helper: reshape -> all_to_all -> slice -> reshape.
    func.func @test_complex_all_to_all(%arg0: tensor<42x64xcomplex<f32>>) -> tensor<42x16xcomplex<f32>> {
        // CHECK: stablehlo.reshape
        // CHECK-SAME: -> tensor<42x4x16x2xf32>
        %0 = stablehlo.reshape %arg0 : (tensor<42x64xcomplex<f32>>) -> tensor<42x4x16xcomplex<f32>>
        // CHECK: stablehlo.all_to_all
        // CHECK-SAME: (tensor<42x4x16x2xf32>) -> tensor<42x4x16x2xf32>
        %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, split_count = 4 : i64, split_dimension = 1 : i64}> : (tensor<42x4x16xcomplex<f32>>) -> tensor<42x4x16xcomplex<f32>>
        // CHECK: stablehlo.slice
        // CHECK-SAME: -> tensor<42x1x16x2xf32>
        %2 = stablehlo.slice %1 [0:42, 0:1, 0:16] : (tensor<42x4x16xcomplex<f32>>) -> tensor<42x1x16xcomplex<f32>>
        %3 = stablehlo.reshape %2 : (tensor<42x1x16xcomplex<f32>>) -> tensor<42x16xcomplex<f32>>
        return %3 : tensor<42x16xcomplex<f32>>
    }
}
