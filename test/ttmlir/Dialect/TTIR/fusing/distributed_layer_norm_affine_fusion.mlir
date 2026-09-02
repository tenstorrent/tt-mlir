// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

module {
    // Graph A / STATE B AdaLN: distributed_layer_norm on the D/tp shard,
    // then activation-sized *(1+scale)+shift. The per-channel 1+scale add
    // stays; the full-activation multiply and add fold into weight/bias.
    // CHECK-LABEL: func.func @distributed_layer_norm_affine_adaln
    func.func @distributed_layer_norm_affine_adaln(%arg0: tensor<1x4096x1280xf32>, %scale: tensor<1x1x1280xf32>, %shift: tensor<1x1x1280xf32>) -> tensor<1x4096x1280xf32> {
        // CHECK: %[[SCALE:.*]] = "ttir.add"({{.*}}) : (tensor<1x1x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x1x1280xf32>
        // CHECK: %[[WEIGHT:.*]] = "ttir.reshape"(%[[SCALE]])
        // CHECK: %[[BIAS:.*]] = "ttir.reshape"(%arg2)
        // CHECK: %[[RESULT:.*]] = "ttir.distributed_layer_norm"(%arg0, %[[WEIGHT]], %[[BIAS]])
        // CHECK-SAME: cluster_axis = 1
        // CHECK-SAME: (tensor<1x4096x1280xf32>, tensor<1280xf32>, tensor<1280xf32>) -> tensor<1x4096x1280xf32>
        // CHECK-NEXT: return %[[RESULT]]
        %one = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1x1x1280xf32>}> : () -> tensor<1x1x1280xf32>
        %0 = "ttir.distributed_layer_norm"(%arg0) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
        %1 = "ttir.add"(%one, %scale) : (tensor<1x1x1280xf32>, tensor<1x1x1280xf32>) -> tensor<1x1x1280xf32>
        %2 = "ttir.broadcast"(%1) <{broadcast_dimensions = array<i64: 1, 4096, 1>}> : (tensor<1x1x1280xf32>) -> tensor<1x4096x1280xf32>
        %3 = "ttir.multiply"(%0, %2) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
        %4 = "ttir.broadcast"(%shift) <{broadcast_dimensions = array<i64: 1, 4096, 1>}> : (tensor<1x1x1280xf32>) -> tensor<1x4096x1280xf32>
        %5 = "ttir.add"(%3, %4) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
        return %5 : tensor<1x4096x1280xf32>
    }

    // Same AdaLN with the fp32-norm / bf16-modulation typecast used by the
    // Wan benchmark patch.
    // CHECK-LABEL: func.func @distributed_layer_norm_affine_cast_between
    func.func @distributed_layer_norm_affine_cast_between(%arg0: tensor<1x4096x1280xf32>, %w: tensor<1x1x1280xbf16>, %b: tensor<1x1x1280xbf16>) -> tensor<1x4096x1280xbf16> {
        // CHECK: %[[W:.*]] = "ttir.typecast"{{.*}}(tensor<1280xbf16>) -> tensor<1280xf32>
        // CHECK: %[[B:.*]] = "ttir.typecast"{{.*}}(tensor<1280xbf16>) -> tensor<1280xf32>
        // CHECK: %[[NORM:.*]] = "ttir.distributed_layer_norm"(%arg0, %[[W]], %[[B]])
        // CHECK-SAME: (tensor<1x4096x1280xf32>, tensor<1280xf32>, tensor<1280xf32>) -> tensor<1x4096x1280xf32>
        // CHECK-NEXT: %[[RESULT:.*]] = "ttir.typecast"(%[[NORM]])
        // CHECK-NEXT: return %[[RESULT]]
        %0 = "ttir.distributed_layer_norm"(%arg0) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
        %1 = "ttir.typecast"(%0) : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xbf16>
        %2 = "ttir.broadcast"(%w) <{broadcast_dimensions = array<i64: 1, 4096, 1>}> : (tensor<1x1x1280xbf16>) -> tensor<1x4096x1280xbf16>
        %3 = "ttir.multiply"(%1, %2) : (tensor<1x4096x1280xbf16>, tensor<1x4096x1280xbf16>) -> tensor<1x4096x1280xbf16>
        %4 = "ttir.broadcast"(%b) <{broadcast_dimensions = array<i64: 1, 4096, 1>}> : (tensor<1x1x1280xbf16>) -> tensor<1x4096x1280xbf16>
        %5 = "ttir.add"(%3, %4) : (tensor<1x4096x1280xbf16>, tensor<1x4096x1280xbf16>) -> tensor<1x4096x1280xbf16>
        return %5 : tensor<1x4096x1280xbf16>
    }

    // Cross-attn norm2: already affine (γ/β on the D shard). Do not compose
    // a second trailing mul/add into those operands.
    // CHECK-LABEL: func.func @distributed_layer_norm_affine_already_affine
    func.func @distributed_layer_norm_affine_already_affine(%arg0: tensor<1x4096x1280xf32>, %g: tensor<1280xf32>, %be: tensor<1280xf32>, %w: tensor<1x1x1280xf32>, %b: tensor<1x1x1280xf32>) -> tensor<1x4096x1280xf32> {
        // CHECK: "ttir.multiply"
        %0 = "ttir.distributed_layer_norm"(%arg0, %g, %be) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0>}> : (tensor<1x4096x1280xf32>, tensor<1280xf32>, tensor<1280xf32>) -> tensor<1x4096x1280xf32>
        %1 = "ttir.broadcast"(%w) <{broadcast_dimensions = array<i64: 1, 4096, 1>}> : (tensor<1x1x1280xf32>) -> tensor<1x4096x1280xf32>
        %2 = "ttir.multiply"(%0, %1) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
        %3 = "ttir.broadcast"(%b) <{broadcast_dimensions = array<i64: 1, 4096, 1>}> : (tensor<1x1x1280xf32>) -> tensor<1x4096x1280xf32>
        %4 = "ttir.add"(%2, %3) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
        return %4 : tensor<1x4096x1280xf32>
    }

    // Gated residual after the norm: x + gate * attn. Full-activation addend,
    // not AdaLN shift. Must not become the norm's bias.
    // CHECK-LABEL: func.func @distributed_layer_norm_gated_residual
    func.func @distributed_layer_norm_gated_residual(%arg0: tensor<1x4096x1280xf32>, %w: tensor<1x1x1280xf32>, %gate: tensor<1x4096x1280xf32>, %attn: tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32> {
        // CHECK: "ttir.distributed_layer_norm"
        // CHECK-SAME: operandSegmentSizes = array<i32: 1, 0, 0, 0>
        // CHECK: "ttir.multiply"
        // CHECK: "ttir.add"
        %0 = "ttir.distributed_layer_norm"(%arg0) <{cluster_axis = 1 : ui32, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
        %1 = "ttir.broadcast"(%w) <{broadcast_dimensions = array<i64: 1, 4096, 1>}> : (tensor<1x1x1280xf32>) -> tensor<1x4096x1280xf32>
        %2 = "ttir.multiply"(%0, %1) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
        %3 = "ttir.multiply"(%gate, %attn) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
        %4 = "ttir.add"(%2, %3) : (tensor<1x4096x1280xf32>, tensor<1x4096x1280xf32>) -> tensor<1x4096x1280xf32>
        return %4 : tensor<1x4096x1280xf32>
    }
}
