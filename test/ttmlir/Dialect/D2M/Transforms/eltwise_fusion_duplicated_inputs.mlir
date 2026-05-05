// RUN: ttmlir-opt --ttir-to-ttmetal-fe-pipeline --d2m-elementwise-fusion %s | FileCheck %s

module {

    //CHECK-LABEL: func @test_duplicate_unary_to_binary
    func.func @test_duplicate_unary_to_binary(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
        // CHECK: d2m.generic
        // CHECK: d2m.tile_abs
        // CHECK-NOT: d2m.generic
        // CHECK: d2m.tile_add
        %1 = "ttir.abs"(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
        %2 = "ttir.add"(%1, %1) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
        return %2 : tensor<64x64xbf16>
    }

    // CHECK-LABEL: func @test_duplicate_small_unary_chain_to_binary
    func.func @test_duplicate_small_unary_chain_to_binary(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
        // CHECK: d2m.generic
        // CHECK: d2m.tile_abs
        // CHECK-NOT: d2m.generic
        // CHECK: d2m.tile_exp
        // CHECK-NOT: d2m.generic
        // CHECK: d2m.tile_add
        %1 = "ttir.abs"(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
        %2 = "ttir.exp"(%1) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
        %3 = "ttir.add"(%1, %2) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
        return %3 : tensor<64x64xbf16>
    }

    // CHECK-LABEL: func @test_duplicate_large_subgraph
    func.func @test_duplicate_large_subgraph(%arg0: tensor<2x32x3072xbf16>, %arg1: tensor<2x32x3072xbf16>, %arg2: tensor<2x32x3072xbf16>, %arg3: tensor<2x32x3072xbf16>, %arg4: tensor<2x32x3072xbf16>, %arg5: tensor<2x32x3072xbf16>, %arg6: tensor<2x32x3072xbf16>, %arg7: tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16> {

        // This graph fuses in such a way that the last fusion step is between the %2 clamp op and a fused generic op. This
        // generic op takes %2 twice as input. This should fuse into a single generic op.

        // CHECK: d2m.generic
        // CHECK: d2m.tile_maximum
        // CHECK-NOT: d2m.generic
        // CHECK: d2m.tile_minimum
        // CHECK-NOT: d2m.generic
        // CHECK: d2m.tile_mul
        // CHECK-NOT: d2m.generic
        // CHECK: d2m.tile_sigmoid
        // CHECK-NOT: d2m.generic
        // CHECK: d2m.tile_mul
        // CHECK-NOT: d2m.generic
        // CHECK: d2m.tile_maximum
        // CHECK-NOT: d2m.generic
        // CHECK: d2m.tile_minimum
        // CHECK-NOT: d2m.generic
        // CHECK: d2m.tile_add
        // CHECK-NOT: d2m.generic
        // CHECK: d2m.tile_mul
        %0 = "ttir.clamp_tensor"(%arg0, %arg2, %arg3) : (tensor<2x32x3072xbf16>, tensor<2x32x3072xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
        %1 = "ttir.add"(%0, %arg6) : (tensor<2x32x3072xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
        %2 = "ttir.clamp_tensor"(%arg1, %arg4, %arg5) : (tensor<2x32x3072xbf16>, tensor<2x32x3072xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
        %3 = "ttir.multiply"(%2, %arg7) : (tensor<2x32x3072xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
        %4 = "ttir.sigmoid"(%3) : (tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
        %5 = "ttir.multiply"(%2, %4) : (tensor<2x32x3072xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
        %6 = "ttir.multiply"(%1, %5) : (tensor<2x32x3072xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
        return %6 : tensor<2x32x3072xbf16>
    }
}
