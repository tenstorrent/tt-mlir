// TTIR subgraph from GPT OSS: dot_general -> bias -> slice gate/up -> clamp -> SiLU -> gate * SiLU(up)
// Inputs: lhs, rhs, bias, clamp_min_gate, clamp_max, gate_add, clamp_min_up, up_scale

module {
  func.func @gpt_oss_20b(
      %arg0: tensor<4x16x2880xbf16>,
      %arg1: tensor<4x2880x5760xbf16>,
      %arg2: tensor<4x5760xbf16>,
      %arg3: tensor<4x16x2880xbf16>,
      %arg4: tensor<4x16x2880xbf16>,
      %arg5: tensor<4x16x2880xbf16>,
      %arg6: tensor<4x16x2880xbf16>,
      %arg7: tensor<4x16x2880xbf16>)
      -> tensor<4x16x2880xbf16> {
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<4x16x2880xbf16>, tensor<4x2880x5760xbf16>) -> tensor<4x16x5760xbf16>
    %1 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 4 : i32, 5760 : i32]}> : (tensor<4x5760xbf16>) -> tensor<1x4x5760xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [4 : i32, 5760 : i32]}> : (tensor<1x4x5760xbf16>) -> tensor<4x5760xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [4 : i32, 1 : i32, 5760 : i32]}> : (tensor<4x5760xbf16>) -> tensor<4x1x5760xbf16>
    %4 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 16, 1>}> : (tensor<4x1x5760xbf16>) -> tensor<4x16x5760xbf16>
    %5 = "ttir.add"(%0, %4) : (tensor<4x16x5760xbf16>, tensor<4x16x5760xbf16>) -> tensor<4x16x5760xbf16>
    %6 = "ttir.slice_static"(%5) <{begins = [0 : i32, 0 : i32, 1 : i32], ends = [4 : i32, 16 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 2 : i32]}> : (tensor<4x16x5760xbf16>) -> tensor<4x16x2880xbf16>
    %7 = "ttir.clamp_tensor"(%6, %arg3, %arg4) : (tensor<4x16x2880xbf16>, tensor<4x16x2880xbf16>, tensor<4x16x2880xbf16>) -> tensor<4x16x2880xbf16>
    %8 = "ttir.add"(%7, %arg5) : (tensor<4x16x2880xbf16>, tensor<4x16x2880xbf16>) -> tensor<4x16x2880xbf16>
    %9 = "ttir.slice_static"(%5) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [4 : i32, 16 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 2 : i32]}> : (tensor<4x16x5760xbf16>) -> tensor<4x16x2880xbf16>
    %10 = "ttir.clamp_tensor"(%9, %arg6, %arg4) : (tensor<4x16x2880xbf16>, tensor<4x16x2880xbf16>, tensor<4x16x2880xbf16>) -> tensor<4x16x2880xbf16>
    %11 = "ttir.multiply"(%10, %arg7) : (tensor<4x16x2880xbf16>, tensor<4x16x2880xbf16>) -> tensor<4x16x2880xbf16>
    %12 = "ttir.sigmoid"(%11) : (tensor<4x16x2880xbf16>) -> tensor<4x16x2880xbf16>
    %13 = "ttir.multiply"(%10, %12) : (tensor<4x16x2880xbf16>, tensor<4x16x2880xbf16>) -> tensor<4x16x2880xbf16>
    %14 = "ttir.multiply"(%8, %13) : (tensor<4x16x2880xbf16>, tensor<4x16x2880xbf16>) -> tensor<4x16x2880xbf16>
    return %14 : tensor<4x16x2880xbf16>
  }
}
