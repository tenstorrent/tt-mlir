// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file  --stablehlo-pipeline --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @TwoGelus attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
    func.func @main(%arg0: tensor<32x32xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}"} ) -> tensor<32x32xf32> {
        %0 = stablehlo.reshape %arg0 : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
        %1 = stablehlo.reshape %0 : (tensor<1x32x32xf32>) -> tensor<32x32xf32>
        // CHECK: ttir.gelu
        // CHECK-SAME: (tensor<32x16xf32>, tensor<32x16xf32>) -> tensor<32x16xf32>
        // CHECK: ttir.gelu
        // CHECK-SAME: (tensor<32x16xf32>, tensor<32x16xf32>) -> tensor<32x16xf32>
        %2 = stablehlo.composite "tenstorrent.gelu_tanh" %1 {composite_attributes = {approximate = "tanh"}, decomposition = @tenstorrent.gelu_tanh.impl_0} : (tensor<32x32xf32>) -> tensor<32x32xf32>
        %3 = stablehlo.composite "tenstorrent.gelu_tanh" %2 {composite_attributes = {approximate = "tanh"}, decomposition = @tenstorrent.gelu_tanh.impl} : (tensor<32x32xf32>) -> tensor<32x32xf32>
        return %3 : tensor<32x32xf32>
    }
    func.func private @tenstorrent.gelu_tanh.impl(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
        %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
        %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %cst_1 = stablehlo.constant dense<0.797884583> : tensor<f32>
        %cst_2 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
        %cst_3 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
        %0 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
        %1 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
        %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
        %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
        %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
        %5 = stablehlo.multiply %arg0, %4 : tensor<32x32xf32>
        %6 = stablehlo.power %arg0, %0 : tensor<32x32xf32>
        %7 = stablehlo.multiply %6, %1 : tensor<32x32xf32>
        %8 = stablehlo.add %arg0, %7 : tensor<32x32xf32>
        %9 = stablehlo.multiply %8, %2 : tensor<32x32xf32>
        %10 = stablehlo.tanh %9 : tensor<32x32xf32>
        %11 = stablehlo.add %10, %3 : tensor<32x32xf32>
        %12 = stablehlo.multiply %5, %11 : tensor<32x32xf32>
        return %12 : tensor<32x32xf32>
    }
    func.func private @tenstorrent.gelu_tanh.impl_0(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
        %cst = stablehlo.constant dense<5.000000e-01> : tensor<f32>
        %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %cst_1 = stablehlo.constant dense<0.797884583> : tensor<f32>
        %cst_2 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
        %cst_3 = stablehlo.constant dense<3.000000e+00> : tensor<f32>
        %0 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
        %1 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
        %2 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
        %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
        %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
        %5 = stablehlo.multiply %arg0, %4 : tensor<32x32xf32>
        %6 = stablehlo.power %arg0, %0 : tensor<32x32xf32>
        %7 = stablehlo.multiply %6, %1 : tensor<32x32xf32>
        %8 = stablehlo.add %arg0, %7 : tensor<32x32xf32>
        %9 = stablehlo.multiply %8, %2 : tensor<32x32xf32>
        %10 = stablehlo.tanh %9 : tensor<32x32xf32>
        %11 = stablehlo.add %10, %3 : tensor<32x32xf32>
        %12 = stablehlo.multiply %5, %11 : tensor<32x32xf32>
        return %12 : tensor<32x32xf32>
    }
}
