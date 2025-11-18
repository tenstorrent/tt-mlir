// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline="enable-cpu-fallback=true" -o %t.mlir %s
// // RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t1.mlir %t.mlir
// // RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t1.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module {
    // CHECK: ttcore.device_module {
    // CHECK: builtin.module {

    // CHECK-LABEL: func.func @test_add_einsum_mixed
    func.func @test_add_einsum_mixed(%arg0: tensor<32x64xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<64x48xf32>) -> tensor<32x48xf32> {
        // CHECK: %[[ADD:.*]] = "ttir.add"
        %0 = stablehlo.add %arg0, %arg1 : tensor<32x64xf32>

        // CHECK: %[[MUL:.*]] = "ttir.multiply"
        %scale = stablehlo.constant dense<2.0> : tensor<32x64xf32>
        %1 = stablehlo.multiply %0, %scale : tensor<32x64xf32>

        // CHECK: %[[EINSUM:.*]] = call @hoisted_stablehlo_einsum_32x64_64x48_func_decl
        %2 = "stablehlo.einsum"(%1, %arg2) {
        einsum_config = "ij,jk->ik"
        } : (tensor<32x64xf32>, tensor<64x48xf32>) -> tensor<32x48xf32>

        return %2 : tensor<32x48xf32>
    }

    // CHECK-LABEL: func.func @test_dynamic_update_slice
    func.func @test_dynamic_update_slice() -> tensor<4xi32> {
        %base = stablehlo.constant dense<[0, 0, 0, 0]> : tensor<4xi32>
        %update = stablehlo.constant dense<[9, 9]> : tensor<2xi32>
        %start = stablehlo.constant dense<1> : tensor<i32>


        // CHECK: %[[DYNAMIC_UPDATE_SLICE:.*]] = call @hoisted_stablehlo_dynamic_update_slice_4_2__func_decl
        %result = stablehlo.dynamic_update_slice %base, %update, %start
            : (tensor<4xi32>, tensor<2xi32>, tensor<i32>) -> tensor<4xi32>

        return %result : tensor<4xi32>
    }


    // CHECK: func.func private @hoisted_stablehlo_einsum_32x64_64x48_func_decl
    // CHECK: func.func private @hoisted_stablehlo_dynamic_update_slice_4_2__func_decl

    // CHECK: ttcore.cpu_module {
    // CHECK: builtin.module {
    // CHECK: func.func @hoisted_stablehlo_einsum_32x64_64x48_func
    // CHECK: func.func @hoisted_stablehlo_dynamic_update_slice_4_2__func
}
