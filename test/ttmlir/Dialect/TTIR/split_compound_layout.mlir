// RUN: ttmlir-opt --ttir-split-compound-layout %s | FileCheck %s

#dram = #tt.memory_space<dram>
#l1_ = #tt.memory_space<l1>

// CHECK-DAG: #[[row_major1x1:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>>
// CHECK-DAG: #[[row_major1x1_T:.*]] = #tt.layout<(d0, d1) -> (d1, d0), undef, <1x1>, memref<64x128xf32, #l1_>>
// CHECK-DAG: #[[row_major2x2:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<32x64xf32, #l1_>>
// CHECK-DAG: #[[tile1x1_f32:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<2x4x!tt.tile<32x32, f32>, #l1_>>
// CHECK-DAG: #[[tile1x1_bf16:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<2x4x!tt.tile<32x32, bf16>, #l1_>>
// CHECK-DAG: #[[tile1x1_f32_dram:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<2x4x!tt.tile<32x32, f32>, #dram>>
// CHECK-DAG: #[[tile2x2_f32:.*]] = #tt.layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<1x2x!tt.tile<32x32, f32>, #l1_>>

#row_major1x1 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf32, #l1_>>
#row_major1x1_T = #tt.layout<(d0, d1) -> (d1, d0), undef, <1x1>, memref<64x128xf32, #l1_>>
#row_major2x2 = #tt.layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<32x64xf32, #l1_>>
#tile1x1_f32 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<2x4x!tt.tile<32x32, f32>, #l1_>>
#tile1x1_bf16 = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<2x4x!tt.tile<32x32, bf16>, #l1_>>
#tile1x1_f32_dram = #tt.layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<2x4x!tt.tile<32x32, f32>, #dram>>
#tile2x2_f32 = #tt.layout<(d0, d1) -> (d0, d1), undef, <2x2>, memref<1x2x!tt.tile<32x32, f32>, #l1_>>

func.func @noncompound_linear(%in: tensor<64x128xf32, #row_major1x1>) -> tensor<64x128xf32, #row_major1x1_T> {
    %out = tensor.empty() : tensor<64x128xf32, #row_major1x1_T>
    // CHECK-COUNT-1: %[[C:.*]] = "ttir.to_layout"(%[[IN:.*]], %[[OUT:.*]]) : (tensor<64x128xf32, #[[row_major1x1]]>, tensor<64x128xf32, #[[row_major1x1_T]]>) -> tensor<64x128xf32, #[[row_major1x1_T]]>
    // CHECK-NOT: %[[C:.*]] = "ttir.to_layout"[[LAYOUT:.*]]
    %0 = "ttir.to_layout"(%in, %out) : (tensor<64x128xf32, #row_major1x1>, tensor<64x128xf32, #row_major1x1_T>) -> tensor<64x128xf32, #row_major1x1_T>
    return %0 : tensor<64x128xf32, #row_major1x1_T>
}

func.func @noncompound_grid(%in: tensor<64x128xf32, #row_major1x1>) -> tensor<64x128xf32, #row_major2x2> {
    %out = tensor.empty() : tensor<64x128xf32, #row_major2x2>
    // CHECK-COUNT-1: %[[C:.*]] = "ttir.to_layout"(%[[IN:.*]], %[[OUT:.*]]) : (tensor<64x128xf32, #[[row_major1x1]]>, tensor<64x128xf32, #[[row_major2x2]]>) -> tensor<64x128xf32, #[[row_major2x2]]>
    // CHECK-NOT: %[[C:.*]] = "ttir.to_layout"[[LAYOUT:.*]]
    %0 = "ttir.to_layout"(%in, %out) : (tensor<64x128xf32, #row_major1x1>, tensor<64x128xf32, #row_major2x2>) -> tensor<64x128xf32, #row_major2x2>
    return %0 : tensor<64x128xf32, #row_major2x2>
}

func.func @noncompound_format(%in: tensor<64x128xf32, #tile1x1_f32>) -> tensor<64x128xf32, #tile1x1_bf16> {
    %out = tensor.empty() : tensor<64x128xf32, #tile1x1_bf16>
    // CHECK-COUNT-1: %[[C:.*]] = "ttir.to_layout"(%[[IN:.*]], %[[OUT:.*]]) : (tensor<64x128xf32, #[[tile1x1_f32]]>, tensor<64x128xf32, #[[tile1x1_bf16]]>) -> tensor<64x128xf32, #[[tile1x1_bf16]]>
    // CHECK-NOT: %[[C:.*]] = "ttir.to_layout"[[LAYOUT:.*]]
    %0 = "ttir.to_layout"(%in, %out) : (tensor<64x128xf32, #tile1x1_f32>, tensor<64x128xf32, #tile1x1_bf16>) -> tensor<64x128xf32, #tile1x1_bf16>
    return %0 : tensor<64x128xf32, #tile1x1_bf16>
}

func.func @noncompound_memspace(%in: tensor<64x128xf32, #tile1x1_f32>) -> tensor<64x128xf32, #tile1x1_f32_dram> {
    %out = tensor.empty() : tensor<64x128xf32, #tile1x1_f32_dram>
    // CHECK-COUNT-1: %[[C:.*]] = "ttir.to_layout"(%[[IN:.*]], %[[OUT:.*]]) : (tensor<64x128xf32, #[[tile1x1_f32]]>, tensor<64x128xf32, #[[tile1x1_f32_dram]]>) -> tensor<64x128xf32, #[[tile1x1_f32_dram]]>
    // CHECK-NOT: %[[C:.*]] = "ttir.to_layout"[[LAYOUT:.*]]
    %0 = "ttir.to_layout"(%in, %out) : (tensor<64x128xf32, #tile1x1_f32>, tensor<64x128xf32, #tile1x1_f32_dram>) -> tensor<64x128xf32, #tile1x1_f32_dram>
    return %0 : tensor<64x128xf32, #tile1x1_f32_dram>
}

func.func @compound_gridformat(%in: tensor<64x128xf32, #row_major1x1>) -> tensor<64x128xf32, #tile2x2_f32> {
    %out = tensor.empty() : tensor<64x128xf32, #tile2x2_f32>
    // CHECK-COUNT-1: %[[C:.*]] = "ttir.to_layout"(%[[IN:.*]], %[[OUT:.*]]) : (tensor<64x128xf32, #[[row_major1x1]]>, tensor<64x128xf32, #[[tile1x1_f32]]>) -> tensor<64x128xf32, #[[tile1x1_f32]]>
    // CHECK-COUNT-1: %[[C:.*]] = "ttir.to_layout"(%[[IN:.*]], %[[OUT:.*]]) : (tensor<64x128xf32, #[[tile1x1_f32]]>, tensor<64x128xf32, #[[tile2x2_f32]]>) -> tensor<64x128xf32, #[[tile2x2_f32]]>
    // CHECK-NOT: %[[C:.*]] = "ttir.to_layout"[[LAYOUT:.*]]
    %0 = "ttir.to_layout"(%in, %out) : (tensor<64x128xf32, #row_major1x1>, tensor<64x128xf32, #tile2x2_f32>) -> tensor<64x128xf32, #tile2x2_f32>
    return %0 : tensor<64x128xf32, #tile2x2_f32>
}

func.func @compound_gridmemspace(%in: tensor<64x128xf32, #tile1x1_f32_dram>) -> tensor<64x128xf32, #tile2x2_f32> {
    %out = tensor.empty() : tensor<64x128xf32, #tile2x2_f32>
    // CHECK-COUNT-1: %[[C:.*]] = "ttir.to_layout"(%[[IN:.*]], %[[OUT:.*]]) : (tensor<64x128xf32, #[[tile1x1_f32_dram]]>, tensor<64x128xf32, #[[tile1x1_f32]]>) -> tensor<64x128xf32, #[[tile1x1_f32]]>
    // CHECK-COUNT-1: %[[C:.*]] = "ttir.to_layout"(%[[IN:.*]], %[[OUT:.*]]) : (tensor<64x128xf32, #[[tile1x1_f32]]>, tensor<64x128xf32, #[[tile2x2_f32]]>) -> tensor<64x128xf32, #[[tile2x2_f32]]>
    // CHECK-NOT: %[[C:.*]] = "ttir.to_layout"[[LAYOUT:.*]]
    %0 = "ttir.to_layout"(%in, %out) : (tensor<64x128xf32, #tile1x1_f32_dram>, tensor<64x128xf32, #tile2x2_f32>) -> tensor<64x128xf32, #tile2x2_f32>
    return %0 : tensor<64x128xf32, #tile2x2_f32>
}

func.func @compound_gridmemspaceformat(%in: tensor<64x128xf32, #tile1x1_f32_dram>) -> tensor<64x128xf32, #row_major2x2> {
    %out = tensor.empty() : tensor<64x128xf32, #row_major2x2>
    // CHECK-COUNT-1: %[[C:.*]] = "ttir.to_layout"(%[[IN:.*]], %[[OUT:.*]]) : (tensor<64x128xf32, #[[tile1x1_f32_dram]]>, tensor<64x128xf32, #[[tile1x1_f32]]>) -> tensor<64x128xf32, #[[tile1x1_f32]]>
    // CHECK-COUNT-1: %[[C:.*]] = "ttir.to_layout"(%[[IN:.*]], %[[OUT:.*]]) : (tensor<64x128xf32, #[[tile1x1_f32]]>, tensor<64x128xf32, #[[tile2x2_f32]]>) -> tensor<64x128xf32, #[[tile2x2_f32]]>
    // CHECK-COUNT-1: %[[C:.*]] = "ttir.to_layout"(%[[IN:.*]], %[[OUT:.*]]) : (tensor<64x128xf32, #[[tile2x2_f32]]>, tensor<64x128xf32, #[[row_major2x2]]>) -> tensor<64x128xf32, #[[row_major2x2]]>
    // CHECK-NOT: %[[C:.*]] = "ttir.to_layout"[[LAYOUT:.*]]
    %0 = "ttir.to_layout"(%in, %out) : (tensor<64x128xf32, #tile1x1_f32_dram>, tensor<64x128xf32, #row_major2x2>) -> tensor<64x128xf32, #row_major2x2>
    return %0 : tensor<64x128xf32, #row_major2x2>
}
