// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

module {
    // adaLN modulation: layer_norm(x) * (1 + scale) + shift.
    // The `1 + scale` add stays (it is per-channel), but the multiply and add
    // over the full activation are absorbed into the norm's weight and bias.
    // CHECK-LABEL: func.func @layer_norm_affine_adaln
    func.func @layer_norm_affine_adaln(%arg0: tensor<1x256x512xf32>, %scale: tensor<1x1x512xf32>, %shift: tensor<1x1x512xf32>) -> tensor<1x256x512xf32> {
        // The `1 + scale` add survives, but on the per-channel tensor.
        // CHECK: %[[SCALE:.*]] = "ttir.add"({{.*}}) : (tensor<1x1x512xf32>, tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
        // CHECK: %[[WEIGHT:.*]] = "ttir.reshape"(%[[SCALE]])
        // CHECK: %[[BIAS:.*]] = "ttir.reshape"(%arg2)
        // CHECK: %[[RESULT:.*]] = "ttir.layer_norm"(%arg0, %[[WEIGHT]], %[[BIAS]])
        // CHECK-SAME: (tensor<1x256x512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x256x512xf32>
        // The activation-sized multiply and add are gone.
        // CHECK-NEXT: return %[[RESULT]]
        %one = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1x1x512xf32>}> : () -> tensor<1x1x512xf32>
        %0 = "ttir.layer_norm"(%arg0) <{normalized_shape = array<i64: 512>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %1 = "ttir.add"(%one, %scale) : (tensor<1x1x512xf32>, tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
        %2 = "ttir.broadcast"(%1) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xf32>) -> tensor<1x256x512xf32>
        %3 = "ttir.multiply"(%0, %2) : (tensor<1x256x512xf32>, tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %4 = "ttir.broadcast"(%shift) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xf32>) -> tensor<1x256x512xf32>
        %5 = "ttir.add"(%3, %4) : (tensor<1x256x512xf32>, tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        return %5 : tensor<1x256x512xf32>
    }

    // Affine params already 1D and unbroadcast - no reshape needed.
    // CHECK-LABEL: func.func @layer_norm_affine_1d_params
    func.func @layer_norm_affine_1d_params(%arg0: tensor<1x256x512xf32>, %w: tensor<512xf32>, %b: tensor<512xf32>) -> tensor<1x256x512xf32> {
        // CHECK: %[[RESULT:.*]] = "ttir.layer_norm"(%arg0, %arg1, %arg2)
        // CHECK-SAME: (tensor<1x256x512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x256x512xf32>
        // CHECK-NEXT: return %[[RESULT]]
        %0 = "ttir.layer_norm"(%arg0) <{normalized_shape = array<i64: 512>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %1 = "ttir.reshape"(%w) <{shape = [1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xf32>) -> tensor<1x1x512xf32>
        %2 = "ttir.broadcast"(%1) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xf32>) -> tensor<1x256x512xf32>
        %3 = "ttir.multiply"(%0, %2) : (tensor<1x256x512xf32>, tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %4 = "ttir.reshape"(%b) <{shape = [1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xf32>) -> tensor<1x1x512xf32>
        %5 = "ttir.broadcast"(%4) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xf32>) -> tensor<1x256x512xf32>
        %6 = "ttir.add"(%5, %3) : (tensor<1x256x512xf32>, tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        return %6 : tensor<1x256x512xf32>
    }

    // Negative: the norm already carries its own affine, so there is nothing to
    // substitute into. (Wan's cross-attention norm2 is this shape.)
    // CHECK-LABEL: func.func @layer_norm_affine_already_affine
    func.func @layer_norm_affine_already_affine(%arg0: tensor<1x256x512xf32>, %g: tensor<512xf32>, %be: tensor<512xf32>, %w: tensor<1x1x512xf32>, %b: tensor<1x1x512xf32>) -> tensor<1x256x512xf32> {
        // CHECK: "ttir.multiply"
        %0 = "ttir.layer_norm"(%arg0, %g, %be) <{normalized_shape = array<i64: 512>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<1x256x512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x256x512xf32>
        %1 = "ttir.broadcast"(%w) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xf32>) -> tensor<1x256x512xf32>
        %2 = "ttir.multiply"(%0, %1) : (tensor<1x256x512xf32>, tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %3 = "ttir.broadcast"(%b) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xf32>) -> tensor<1x256x512xf32>
        %4 = "ttir.add"(%2, %3) : (tensor<1x256x512xf32>, tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        return %4 : tensor<1x256x512xf32>
    }

    // Negative: the addend is a full activation (a residual), not a per-channel
    // bias. Folding it into the norm would be wrong.
    // CHECK-LABEL: func.func @layer_norm_affine_residual_addend
    func.func @layer_norm_affine_residual_addend(%arg0: tensor<1x256x512xf32>, %w: tensor<1x1x512xf32>, %res: tensor<1x256x512xf32>) -> tensor<1x256x512xf32> {
        // CHECK: "ttir.multiply"
        %0 = "ttir.layer_norm"(%arg0) <{normalized_shape = array<i64: 512>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %1 = "ttir.broadcast"(%w) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xf32>) -> tensor<1x256x512xf32>
        %2 = "ttir.multiply"(%0, %1) : (tensor<1x256x512xf32>, tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %3 = "ttir.add"(%2, %res) : (tensor<1x256x512xf32>, tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        return %3 : tensor<1x256x512xf32>
    }

    // Negative: the scale is a full activation, so the multiply is not an
    // affine and cannot become the norm's weight.
    // CHECK-LABEL: func.func @layer_norm_affine_activation_scale
    func.func @layer_norm_affine_activation_scale(%arg0: tensor<1x256x512xf32>, %w: tensor<1x256x512xf32>, %b: tensor<1x1x512xf32>) -> tensor<1x256x512xf32> {
        // CHECK: "ttir.multiply"
        %0 = "ttir.layer_norm"(%arg0) <{normalized_shape = array<i64: 512>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %1 = "ttir.multiply"(%0, %w) : (tensor<1x256x512xf32>, tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %2 = "ttir.broadcast"(%b) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xf32>) -> tensor<1x256x512xf32>
        %3 = "ttir.add"(%1, %2) : (tensor<1x256x512xf32>, tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        return %3 : tensor<1x256x512xf32>
    }

    // A typecast between the norm and the modulation: the norm runs in f32
    // while the affine is bf16. The affine moves inside the norm and is cast
    // up to the norm's type; the result is cast back down once at the end.
    // CHECK-LABEL: func.func @layer_norm_affine_cast_between
    func.func @layer_norm_affine_cast_between(%arg0: tensor<1x256x512xf32>, %w: tensor<1x1x512xbf16>, %b: tensor<1x1x512xbf16>) -> tensor<1x256x512xbf16> {
        // CHECK: %[[W:.*]] = "ttir.typecast"{{.*}}(tensor<512xbf16>) -> tensor<512xf32>
        // CHECK: %[[B:.*]] = "ttir.typecast"{{.*}}(tensor<512xbf16>) -> tensor<512xf32>
        // CHECK: %[[NORM:.*]] = "ttir.layer_norm"(%arg0, %[[W]], %[[B]])
        // CHECK-SAME: (tensor<1x256x512xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x256x512xf32>
        // CHECK-NEXT: %[[RESULT:.*]] = "ttir.typecast"(%[[NORM]])
        // CHECK-NEXT: return %[[RESULT]]
        %0 = "ttir.layer_norm"(%arg0) <{normalized_shape = array<i64: 512>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %1 = "ttir.typecast"(%0) : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
        %2 = "ttir.broadcast"(%w) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xbf16>) -> tensor<1x256x512xbf16>
        %3 = "ttir.multiply"(%1, %2) : (tensor<1x256x512xbf16>, tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
        %4 = "ttir.broadcast"(%b) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xbf16>) -> tensor<1x256x512xbf16>
        %5 = "ttir.add"(%3, %4) : (tensor<1x256x512xbf16>, tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
        return %5 : tensor<1x256x512xbf16>
    }

    // Negative: the typecast between the norm and the modulation has a second
    // consumer. Folding would leave the norm and its cast live, so the norm
    // would be computed twice.
    // CHECK-LABEL: func.func @layer_norm_affine_multi_use_cast
    func.func @layer_norm_affine_multi_use_cast(%arg0: tensor<1x256x512xf32>, %w: tensor<1x1x512xbf16>, %b: tensor<1x1x512xbf16>) -> (tensor<1x256x512xbf16>, tensor<1x256x512xbf16>) {
        // CHECK: "ttir.multiply"
        %0 = "ttir.layer_norm"(%arg0) <{normalized_shape = array<i64: 512>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %1 = "ttir.typecast"(%0) : (tensor<1x256x512xf32>) -> tensor<1x256x512xbf16>
        %2 = "ttir.broadcast"(%w) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xbf16>) -> tensor<1x256x512xbf16>
        %3 = "ttir.multiply"(%1, %2) : (tensor<1x256x512xbf16>, tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
        %4 = "ttir.broadcast"(%b) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xbf16>) -> tensor<1x256x512xbf16>
        %5 = "ttir.add"(%3, %4) : (tensor<1x256x512xbf16>, tensor<1x256x512xbf16>) -> tensor<1x256x512xbf16>
        return %5, %1 : tensor<1x256x512xbf16>, tensor<1x256x512xbf16>
    }

    // Negative: the norm result feeds a second consumer, so folding would leave
    // the unmodulated norm live and duplicate the work.
    // CHECK-LABEL: func.func @layer_norm_affine_multi_use_norm
    func.func @layer_norm_affine_multi_use_norm(%arg0: tensor<1x256x512xf32>, %w: tensor<1x1x512xf32>, %b: tensor<1x1x512xf32>) -> (tensor<1x256x512xf32>, tensor<1x256x512xf32>) {
        // CHECK: "ttir.multiply"
        %0 = "ttir.layer_norm"(%arg0) <{normalized_shape = array<i64: 512>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %1 = "ttir.broadcast"(%w) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xf32>) -> tensor<1x256x512xf32>
        %2 = "ttir.multiply"(%0, %1) : (tensor<1x256x512xf32>, tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        %3 = "ttir.broadcast"(%b) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x512xf32>) -> tensor<1x256x512xf32>
        %4 = "ttir.add"(%2, %3) : (tensor<1x256x512xf32>, tensor<1x256x512xf32>) -> tensor<1x256x512xf32>
        return %4, %0 : tensor<1x256x512xf32>, tensor<1x256x512xf32>
    }
}
