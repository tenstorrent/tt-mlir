// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=BFInterleaved" %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device|any_device_tile|l1_block_sharded>
#loc = loc("SelfAttention":0:0)
module @SelfAttention attributes {} {
  func.func @forward(%arg0: tensor<1x12x3200xf32> {ttir.name = "hidden_states_1"} loc("SelfAttention":0:0), %arg1: tensor<1x1x12x12xf32> {ttir.name = "attention_mask"} loc("SelfAttention":0:0), %arg2: tensor<1xf32> {ttir.name = "input_1_multiply_20"} loc("SelfAttention":0:0), %arg3: tensor<3200x3200xf32> {ttir.name = "model.q_proj.weight"} loc("SelfAttention":0:0), %arg4: tensor<3200x3200xf32> {ttir.name = "model.k_proj.weight"} loc("SelfAttention":0:0), %arg5: tensor<3200x3200xf32> {ttir.name = "model.v_proj.weight"} loc("SelfAttention":0:0), %arg6: tensor<3200x3200xf32> {ttir.name = "model.o_proj.weight"} loc("SelfAttention":0:0)) -> (tensor<1x12x3200xf32> {ttir.name = "SelfAttention.output_reshape_38"}) {
    // CHECK-DAG: #[[L1_:.*]] = #ttnn.buffer_type<l1>
    // CHECK-DAG: #[[LAYOUT_8:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<2x400xf32, #l1_>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_9:.*]] = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <8x8>, memref<48x13xf32, #l1_>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_10:.*]] = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 12 + d2, d3), <8x8>, memref<48x13xf32, #l1_>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_11:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 12 + d1, d2), <8x8>, memref<48x13xf32, #l1_>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_12:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 100 + d1, d2), <8x8>, memref<400x2xf32, #l1_>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_13:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 12 + d1, d2), <8x8>, memref<48x2xf32, #l1_>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_14:.*]] = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 12 + d2, d3), <8x8>, memref<48x2xf32, #l1_>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_15:.*]] = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 3200 + d1 * 100 + d2, d3), <8x8>, memref<400x2xf32, #l1_>, <interleaved>>
    // CHECK-DAG: #[[LAYOUT_16:.*]] = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 12 + d1, d2), <8x8>, memref<2x400xf32, #l1_>, <interleaved>>
    %0 = tensor.empty() : tensor<12x3200xf32> loc(#loc30)
    // CHECK-DAG: %{{.*}} = "ttnn.reshape"{{.*}} -> tensor<12x3200xf32, #[[LAYOUT_8]]>
    %1 = "ttir.squeeze"(%arg0, %0) <{dim = 0 : si32, operand_constraints = [#any_device, #any_device, #any_device, #any_device]}> : (tensor<1x12x3200xf32>, tensor<12x3200xf32>) -> tensor<12x3200xf32> loc(#loc30)
    %2 = tensor.empty() : tensor<12x3200xf32> loc(#loc31)
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<12x3200xf32, #[[LAYOUT_8]]>
    %3 = "ttir.matmul"(%1, %arg3, %2) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<12x3200xf32>, tensor<3200x3200xf32>, tensor<12x3200xf32>) -> tensor<12x3200xf32> loc(#loc31)
    %4 = tensor.empty() : tensor<1x12x32x100xf32> loc(#loc32)
    // CHECK-DAG: %{{.*}} = "ttnn.reshape"{{.*}} -> tensor<1x12x32x100xf32, #[[LAYOUT_9]]>
    %5 = "ttir.reshape"(%3, %4) <{operand_constraints = [#any_device, #any_device], shape = [1 : i32, 12 : i32, 32 : i32, 100 : i32]}> : (tensor<12x3200xf32>, tensor<1x12x32x100xf32>) -> tensor<1x12x32x100xf32> loc(#loc32)
    %6 = tensor.empty() : tensor<1x32x12x100xf32> loc(#loc33)
    // CHECK-DAG: %{{.*}} = "ttnn.transpose"{{.*}} -> tensor<1x32x12x100xf32, #[[LAYOUT_10]]>
    %7 = "ttir.transpose"(%5, %6) <{dim0 = -3 : si32, dim1 = -2 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<1x12x32x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32> loc(#loc33)
    %8 = tensor.empty() : tensor<32x12x100xf32> loc(#loc34)
    // CHECK-DAG: %{{.*}} = "ttnn.reshape"{{.*}} -> tensor<32x12x100xf32, #[[LAYOUT_11]]>
    %9 = "ttir.squeeze"(%7, %8) <{dim = 0 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<1x32x12x100xf32>, tensor<32x12x100xf32>) -> tensor<32x12x100xf32> loc(#loc34)
    %10 = tensor.empty() : tensor<12x3200xf32> loc(#loc35)
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<12x3200xf32, #[[LAYOUT_8]]>
    %11 = "ttir.matmul"(%1, %arg4, %10) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<12x3200xf32>, tensor<3200x3200xf32>, tensor<12x3200xf32>) -> tensor<12x3200xf32> loc(#loc35)
    %12 = tensor.empty() : tensor<1x12x32x100xf32> loc(#loc36)
    // CHECK-DAG: %{{.*}} = "ttnn.reshape"{{.*}} -> tensor<1x12x32x100xf32, #[[LAYOUT_9]]>
    %13 = "ttir.reshape"(%11, %12) <{operand_constraints = [#any_device, #any_device], shape = [1 : i32, 12 : i32, 32 : i32, 100 : i32]}> : (tensor<12x3200xf32>, tensor<1x12x32x100xf32>) -> tensor<1x12x32x100xf32> loc(#loc36)
    %14 = tensor.empty() : tensor<1x32x12x100xf32> loc(#loc37)
    %15 = "ttir.transpose"(%13, %14) <{dim0 = -3 : si32, dim1 = -2 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<1x12x32x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32> loc(#loc37)
    %16 = tensor.empty() : tensor<32x12x100xf32> loc(#loc38)
    %17 = "ttir.squeeze"(%15, %16) <{dim = 0 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<1x32x12x100xf32>, tensor<32x12x100xf32>) -> tensor<32x12x100xf32> loc(#loc38)
    %18 = tensor.empty() : tensor<32x100x12xf32> loc(#loc39)
    // CHECK-DAG: %{{.*}} = "ttnn.transpose"{{.*}} -> tensor<1x32x12x100xf32, #[[LAYOUT_10]]>
    %19 = "ttir.transpose"(%17, %18) <{dim0 = -2 : si32, dim1 = -1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<32x12x100xf32>, tensor<32x100x12xf32>) -> tensor<32x100x12xf32> loc(#loc39)
    %20 = tensor.empty() : tensor<32x12x12xf32> loc(#loc40)
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<12x3200xf32, #[[LAYOUT_8]]>
    %21 = "ttir.matmul"(%9, %19, %20) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<32x12x100xf32>, tensor<32x100x12xf32>, tensor<32x12x12xf32>) -> tensor<32x12x12xf32> loc(#loc40)
    %22 = tensor.empty() : tensor<1x32x12x12xf32> loc(#loc41)
    %23 = "ttir.unsqueeze"(%21, %22) <{dim = 0 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<32x12x12xf32>, tensor<1x32x12x12xf32>) -> tensor<1x32x12x12xf32> loc(#loc41)
    %24 = tensor.empty() : tensor<1x32x12x12xf32> loc(#loc42)
    // CHECK-DAG: %{{.*}} = "ttnn.multiply"{{.*}} -> tensor<1x32x12x12xf32, #[[LAYOUT_14]]>
    %25 = "ttir.multiply"(%23, %arg2, %24) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x32x12x12xf32>, tensor<1xf32>, tensor<1x32x12x12xf32>) -> tensor<1x32x12x12xf32> loc(#loc42)
    %26 = tensor.empty() : tensor<1x32x12x12xf32> loc(#loc43)
    // CHECK-DAG: %{{.*}} = "ttnn.add"{{.*}} -> tensor<1x32x12x12xf32, #[[LAYOUT_14]]>
    %27 = "ttir.add"(%25, %arg1, %26) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x32x12x12xf32>, tensor<1x1x12x12xf32>, tensor<1x32x12x12xf32>) -> tensor<1x32x12x12xf32> loc(#loc43)
    %28 = tensor.empty() : tensor<1x32x12x12xf32> loc(#loc44)
    // CHECK-DAG: %{{.*}} = "ttnn.softmax"{{.*}} -> tensor<1x32x12x12xf32, #[[LAYOUT_14]]>
    %29 = "ttir.softmax"(%27, %28) <{dimension = -1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<1x32x12x12xf32>, tensor<1x32x12x12xf32>) -> tensor<1x32x12x12xf32> loc(#loc44)
    %30 = tensor.empty() : tensor<32x12x12xf32> loc(#loc45)
    %31 = "ttir.squeeze"(%29, %30) <{dim = 0 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<1x32x12x12xf32>, tensor<32x12x12xf32>) -> tensor<32x12x12xf32> loc(#loc45)
    %32 = tensor.empty() : tensor<12x3200xf32> loc(#loc46)
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<32x12x12xf32, #[[LAYOUT_13]]>
    %33 = "ttir.matmul"(%1, %arg5, %32) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<12x3200xf32>, tensor<3200x3200xf32>, tensor<12x3200xf32>) -> tensor<12x3200xf32> loc(#loc46)
    %34 = tensor.empty() : tensor<1x12x32x100xf32> loc(#loc47)
    // CHECK-DAG: %{{.*}} = "ttnn.reshape"{{.*}} -> tensor<1x32x12x12xf32, #[[LAYOUT_14]]>
    %35 = "ttir.reshape"(%33, %34) <{operand_constraints = [#any_device, #any_device], shape = [1 : i32, 12 : i32, 32 : i32, 100 : i32]}> : (tensor<12x3200xf32>, tensor<1x12x32x100xf32>) -> tensor<1x12x32x100xf32> loc(#loc47)
    %36 = tensor.empty() : tensor<1x32x12x100xf32> loc(#loc48)
    // CHECK-DAG: %{{.*}} = "ttnn.transpose"{{.*}} -> tensor<1x32x12x100xf32, #[[LAYOUT_10]]>
    %37 = "ttir.transpose"(%35, %36) <{dim0 = -3 : si32, dim1 = -2 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<1x12x32x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32> loc(#loc48)
    %38 = tensor.empty() : tensor<1x32x100x12xf32> loc(#loc49)
    %39 = "ttir.transpose"(%37, %38) <{dim0 = -2 : si32, dim1 = -1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<1x32x12x100xf32>, tensor<1x32x100x12xf32>) -> tensor<1x32x100x12xf32> loc(#loc49)
    %40 = tensor.empty() : tensor<32x100x12xf32> loc(#loc50)
    // CHECK-DAG: %{{.*}} = "ttnn.reshape"{{.*}} -> tensor<32x100x12xf32, #[[LAYOUT_12]]>
    %41 = "ttir.squeeze"(%39, %40) <{dim = 0 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<1x32x100x12xf32>, tensor<32x100x12xf32>) -> tensor<32x100x12xf32> loc(#loc50)
    %42 = tensor.empty() : tensor<32x12x100xf32> loc(#loc51)
    // CHECK-DAG: %{{.*}} = "ttnn.transpose"{{.*}} -> tensor<32x12x100xf32, #[[LAYOUT_11]]>
    %43 = "ttir.transpose"(%41, %42) <{dim0 = -2 : si32, dim1 = -1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<32x100x12xf32>, tensor<32x12x100xf32>) -> tensor<32x12x100xf32> loc(#loc51)
    %44 = tensor.empty() : tensor<32x12x100xf32> loc(#loc52)
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<32x12x100xf32, #[[LAYOUT_11]]>
    %45 = "ttir.matmul"(%31, %43, %44) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<32x12x12xf32>, tensor<32x12x100xf32>, tensor<32x12x100xf32>) -> tensor<32x12x100xf32> loc(#loc52)
    %46 = tensor.empty() : tensor<1x32x12x100xf32> loc(#loc53)
    // CHECK-DAG: %{{.*}} = "ttnn.reshape"{{.*}} -> tensor<1x32x12x100xf32, #[[LAYOUT_10]]>
    %47 = "ttir.unsqueeze"(%45, %46) <{dim = 0 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<32x12x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32> loc(#loc53)
    %48 = tensor.empty() : tensor<1x12x32x100xf32> loc(#loc54)
    // CHECK-DAG: %{{.*}} = "ttnn.transpose"{{.*}} -> tensor<1x12x32x100xf32, #[[LAYOUT_9]]>
    %49 = "ttir.transpose"(%47, %48) <{dim0 = -3 : si32, dim1 = -2 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<1x32x12x100xf32>, tensor<1x12x32x100xf32>) -> tensor<1x12x32x100xf32> loc(#loc54)
    %50 = tensor.empty() : tensor<12x3200xf32> loc(#loc55)
    // CHECK-DAG: %{{.*}} = "ttnn.reshape"{{.*}} -> tensor<12x3200xf32, #[[LAYOUT_8]]>
    %51 = "ttir.reshape"(%49, %50) <{operand_constraints = [#any_device, #any_device], shape = [12 : i32, 3200 : i32]}> : (tensor<1x12x32x100xf32>, tensor<12x3200xf32>) -> tensor<12x3200xf32> loc(#loc55)
    %52 = tensor.empty() : tensor<12x3200xf32> loc(#loc56)
    // CHECK-DAG: %{{.*}} = "ttnn.matmul"{{.*}} -> tensor<12x3200xf32, #[[LAYOUT_8]]>
    %53 = "ttir.matmul"(%51, %arg6, %52) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<12x3200xf32>, tensor<3200x3200xf32>, tensor<12x3200xf32>) -> tensor<12x3200xf32> loc(#loc56)
    %54 = tensor.empty() : tensor<1x12x3200xf32> loc(#loc57)
    // CHECK-DAG: %{{.*}} = "ttnn.reshape"{{.*}} -> tensor<1x12x3200xf32, #[[LAYOUT_16]]>
    %55 = "ttir.unsqueeze"(%53, %54) <{dim = 0 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<12x3200xf32>, tensor<1x12x3200xf32>) -> tensor<1x12x3200xf32> loc(#loc57)
    return %55 : tensor<1x12x3200xf32> loc(#loc29)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("forward":4294967295:63)
#loc2 = loc("forward":4294967295:65)
#loc3 = loc("forward":4294967295:66)
#loc4 = loc("forward":4294967295:67)
#loc5 = loc("forward":4294967295:68)
#loc6 = loc("forward":4294967295:70)
#loc7 = loc("forward":4294967295:71)
#loc8 = loc("forward":4294967295:72)
#loc9 = loc("forward":4294967295:73)
#loc10 = loc("forward":4294967295:74)
#loc11 = loc("forward":4294967295:75)
#loc12 = loc("forward":4294967295:76)
#loc13 = loc("forward":4294967295:78)
#loc14 = loc("forward":4294967295:80)
#loc15 = loc("forward":4294967295:81)
#loc16 = loc("forward":4294967295:82)
#loc17 = loc("forward":4294967295:84)
#loc18 = loc("forward":4294967295:85)
#loc19 = loc("forward":4294967295:86)
#loc20 = loc("forward":4294967295:87)
#loc21 = loc("forward":4294967295:88)
#loc22 = loc("forward":4294967295:89)
#loc23 = loc("forward":4294967295:90)
#loc24 = loc("forward":4294967295:91)
#loc25 = loc("forward":4294967295:92)
#loc26 = loc("forward":4294967295:93)
#loc27 = loc("forward":4294967295:95)
#loc28 = loc("forward":4294967295:96)
#loc29 = loc(unknown)
#loc30 = loc("reshape_6.dc.squeeze.0"(#loc1))
#loc31 = loc("matmul_8"(#loc2))
#loc32 = loc("reshape_9"(#loc3))
#loc33 = loc("transpose_10"(#loc4))
#loc34 = loc("reshape_11.dc.squeeze.0"(#loc5))
#loc35 = loc("matmul_13"(#loc6))
#loc36 = loc("reshape_14"(#loc7))
#loc37 = loc("transpose_15"(#loc8))
#loc38 = loc("reshape_16.dc.squeeze.0"(#loc9))
#loc39 = loc("transpose_17"(#loc10))
#loc40 = loc("matmul_18"(#loc11))
#loc41 = loc("reshape_19.dc.unsqueeze.0"(#loc12))
#loc42 = loc("multiply_20"(#loc13))
#loc43 = loc("add_21"(#loc14))
#loc44 = loc("softmax_22"(#loc15))
#loc45 = loc("reshape_24.dc.squeeze.0"(#loc16))
#loc46 = loc("matmul_26"(#loc17))
#loc47 = loc("reshape_27"(#loc18))
#loc48 = loc("transpose_28"(#loc19))
#loc49 = loc("transpose_29"(#loc20))
#loc50 = loc("reshape_30.dc.squeeze.0"(#loc21))
#loc51 = loc("transpose_31"(#loc22))
#loc52 = loc("matmul_32"(#loc23))
#loc53 = loc("reshape_33.dc.unsqueeze.0"(#loc24))
#loc54 = loc("transpose_34"(#loc25))
#loc55 = loc("reshape_35"(#loc26))
#loc56 = loc("matmul_37"(#loc27))
#loc57 = loc("reshape_38.dc.unsqueeze.0"(#loc28))
