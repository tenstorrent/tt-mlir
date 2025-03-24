// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s

module @SelfAttention attributes {} {
  func.func @forward(%arg0: tensor<1x12x3200xf32> {ttir.name = "hidden_states_1"}, %arg1: tensor<1x1x12x12xf32> {ttir.name = "attention_mask"}, %arg2: tensor<1x12xf32> {ttir.name = "position_ids"}, %arg3: tensor<1x50x1xf32> {ttir.name = "input_0_unsqueeze_12"}, %arg4: tensor<1x32x50x100xf32> {ttir.name = "dc.input_tensor.index_25.2"}, %arg5: tensor<1xf32> {ttir.name = "input_1_multiply_26"}, %arg6: tensor<1x32x50x100xf32> {ttir.name = "dc.input_tensor.index_27.2"}, %arg7: tensor<1x32x50x100xf32> {ttir.name = "dc.input_tensor.index_39.2"}, %arg8: tensor<1xf32> {ttir.name = "input_1_multiply_40"}, %arg9: tensor<1x32x50x100xf32> {ttir.name = "dc.input_tensor.index_41.2"}, %arg10: tensor<1xf32> {ttir.name = "input_1_multiply_49"}, %arg11: tensor<3200x3200xf32> {ttir.name = "model.q_proj.weight"}, %arg12: tensor<3200x3200xf32> {ttir.name = "model.k_proj.weight"}, %arg13: tensor<3200x3200xf32> {ttir.name = "model.v_proj.weight"}, %arg14: tensor<3200x3200xf32> {ttir.name = "model.o_proj.weight"}) -> (tensor<1x12x3200xf32> {ttir.name = "SelfAttention.output_reshape_67"}) {
    %0 = tensor.empty() : tensor<12x3200xf32>
    %1 = "ttir.squeeze"(%arg0, %0) <{dim = 0 : si32}> : (tensor<1x12x3200xf32>, tensor<12x3200xf32>) -> tensor<12x3200xf32>
    %2 = tensor.empty() : tensor<12x3200xf32>
    %3 = "ttir.matmul"(%1, %arg11, %2) : (tensor<12x3200xf32>, tensor<3200x3200xf32>, tensor<12x3200xf32>) -> tensor<12x3200xf32>
    %4 = tensor.empty() : tensor<1x12x32x100xf32>
    %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 12 : i32, 32 : i32, 100 : i32]}> : (tensor<12x3200xf32>, tensor<1x12x32x100xf32>) -> tensor<1x12x32x100xf32>
    %6 = tensor.empty() : tensor<1x32x12x100xf32>
    %7 = "ttir.transpose"(%5, %6) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x12x32x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
    %8 = tensor.empty() : tensor<1x1x12xf32>
    %9 = "ttir.unsqueeze"(%arg2, %8) <{dim = 1 : si32}> : (tensor<1x12xf32>, tensor<1x1x12xf32>) -> tensor<1x1x12xf32>
    %10 = tensor.empty() : tensor<1x50x12xf32>
    %11 = "ttir.matmul"(%arg3, %9, %10) : (tensor<1x50x1xf32>, tensor<1x1x12xf32>, tensor<1x50x12xf32>) -> tensor<1x50x12xf32>
    %12 = tensor.empty() : tensor<1x12x50xf32>
    %13 = "ttir.transpose"(%11, %12) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x50x12xf32>, tensor<1x12x50xf32>) -> tensor<1x12x50xf32>
    %14 = tensor.empty() : tensor<1x12x100xf32>
    %15 = "ttir.concat"(%13, %13, %14) <{dim = -1 : si32}> : (tensor<1x12x50xf32>, tensor<1x12x50xf32>, tensor<1x12x100xf32>) -> tensor<1x12x100xf32>
    %16 = tensor.empty() : tensor<1x12x100xf32>
    %17 = "ttir.cos"(%15, %16) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x12x100xf32>, tensor<1x12x100xf32>) -> tensor<1x12x100xf32>
    %18 = tensor.empty() : tensor<1x1x12x100xf32>
    %19 = "ttir.unsqueeze"(%17, %18) <{dim = 1 : si32}> : (tensor<1x12x100xf32>, tensor<1x1x12x100xf32>) -> tensor<1x1x12x100xf32>
    %20 = tensor.empty() : tensor<1x32x12x100xf32>
    %21 = "ttir.multiply"(%7, %19, %20) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x12x100xf32>, tensor<1x1x12x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
    %22 = tensor.empty() : tensor<1x32x100x12xf32>
    %23 = "ttir.transpose"(%7, %22) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x12x100xf32>, tensor<1x32x100x12xf32>) -> tensor<1x32x100x12xf32>
    %24 = tensor.empty() : tensor<1x32x50x12xf32>
    %25 = "ttir.matmul"(%arg4, %23, %24) : (tensor<1x32x50x100xf32>, tensor<1x32x100x12xf32>, tensor<1x32x50x12xf32>) -> tensor<1x32x50x12xf32>
    %26 = tensor.empty() : tensor<1x32x12x50xf32>
    %27 = "ttir.transpose"(%25, %26) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x50x12xf32>, tensor<1x32x12x50xf32>) -> tensor<1x32x12x50xf32>
    %28 = tensor.empty() : tensor<1x32x12x50xf32>
    %29 = "ttir.multiply"(%27, %arg5, %28) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x12x50xf32>, tensor<1xf32>, tensor<1x32x12x50xf32>) -> tensor<1x32x12x50xf32>
    %30 = tensor.empty() : tensor<1x32x100x12xf32>
    %31 = "ttir.transpose"(%7, %30) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x12x100xf32>, tensor<1x32x100x12xf32>) -> tensor<1x32x100x12xf32>
    %32 = tensor.empty() : tensor<1x32x50x12xf32>
    %33 = "ttir.matmul"(%arg6, %31, %32) : (tensor<1x32x50x100xf32>, tensor<1x32x100x12xf32>, tensor<1x32x50x12xf32>) -> tensor<1x32x50x12xf32>
    %34 = tensor.empty() : tensor<1x32x12x50xf32>
    %35 = "ttir.transpose"(%33, %34) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x50x12xf32>, tensor<1x32x12x50xf32>) -> tensor<1x32x12x50xf32>
    %36 = tensor.empty() : tensor<1x32x12x100xf32>
    %37 = "ttir.concat"(%29, %35, %36) <{dim = -1 : si32}> : (tensor<1x32x12x50xf32>, tensor<1x32x12x50xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
    %38 = tensor.empty() : tensor<1x12x100xf32>
    %39 = "ttir.sin"(%15, %38) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<1x12x100xf32>, tensor<1x12x100xf32>) -> tensor<1x12x100xf32>
    %40 = tensor.empty() : tensor<1x1x12x100xf32>
    %41 = "ttir.unsqueeze"(%39, %40) <{dim = 1 : si32}> : (tensor<1x12x100xf32>, tensor<1x1x12x100xf32>) -> tensor<1x1x12x100xf32>
    %42 = tensor.empty() : tensor<1x32x12x100xf32>
    %43 = "ttir.multiply"(%37, %41, %42) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x12x100xf32>, tensor<1x1x12x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
    %44 = tensor.empty() : tensor<1x32x12x100xf32>
    %45 = "ttir.add"(%21, %43, %44) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x12x100xf32>, tensor<1x32x12x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
    %46 = tensor.empty() : tensor<32x12x100xf32>
    %47 = "ttir.squeeze"(%45, %46) <{dim = 0 : si32}> : (tensor<1x32x12x100xf32>, tensor<32x12x100xf32>) -> tensor<32x12x100xf32>
    %48 = tensor.empty() : tensor<12x3200xf32>
    %49 = "ttir.matmul"(%1, %arg12, %48) : (tensor<12x3200xf32>, tensor<3200x3200xf32>, tensor<12x3200xf32>) -> tensor<12x3200xf32>
    %50 = tensor.empty() : tensor<1x12x32x100xf32>
    %51 = "ttir.reshape"(%49, %50) <{shape = [1 : i32, 12 : i32, 32 : i32, 100 : i32]}> : (tensor<12x3200xf32>, tensor<1x12x32x100xf32>) -> tensor<1x12x32x100xf32>
    %52 = tensor.empty() : tensor<1x32x12x100xf32>
    %53 = "ttir.transpose"(%51, %52) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x12x32x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
    %54 = tensor.empty() : tensor<1x32x12x100xf32>
    %55 = "ttir.multiply"(%53, %19, %54) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x12x100xf32>, tensor<1x1x12x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
    %56 = tensor.empty() : tensor<1x32x100x12xf32>
    %57 = "ttir.transpose"(%53, %56) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x12x100xf32>, tensor<1x32x100x12xf32>) -> tensor<1x32x100x12xf32>
    %58 = tensor.empty() : tensor<1x32x50x12xf32>
    %59 = "ttir.matmul"(%arg7, %57, %58) : (tensor<1x32x50x100xf32>, tensor<1x32x100x12xf32>, tensor<1x32x50x12xf32>) -> tensor<1x32x50x12xf32>
    %60 = tensor.empty() : tensor<1x32x12x50xf32>
    %61 = "ttir.transpose"(%59, %60) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x50x12xf32>, tensor<1x32x12x50xf32>) -> tensor<1x32x12x50xf32>
    %62 = tensor.empty() : tensor<1x32x12x50xf32>
    %63 = "ttir.multiply"(%61, %arg8, %62) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x12x50xf32>, tensor<1xf32>, tensor<1x32x12x50xf32>) -> tensor<1x32x12x50xf32>
    %64 = tensor.empty() : tensor<1x32x100x12xf32>
    %65 = "ttir.transpose"(%53, %64) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x12x100xf32>, tensor<1x32x100x12xf32>) -> tensor<1x32x100x12xf32>
    %66 = tensor.empty() : tensor<1x32x50x12xf32>
    %67 = "ttir.matmul"(%arg9, %65, %66) : (tensor<1x32x50x100xf32>, tensor<1x32x100x12xf32>, tensor<1x32x50x12xf32>) -> tensor<1x32x50x12xf32>
    %68 = tensor.empty() : tensor<1x32x12x50xf32>
    %69 = "ttir.transpose"(%67, %68) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x50x12xf32>, tensor<1x32x12x50xf32>) -> tensor<1x32x12x50xf32>
    %70 = tensor.empty() : tensor<1x32x12x100xf32>
    %71 = "ttir.concat"(%63, %69, %70) <{dim = -1 : si32}> : (tensor<1x32x12x50xf32>, tensor<1x32x12x50xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
    %72 = tensor.empty() : tensor<1x32x12x100xf32>
    %73 = "ttir.multiply"(%71, %41, %72) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x12x100xf32>, tensor<1x1x12x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
    %74 = tensor.empty() : tensor<1x32x12x100xf32>
    %75 = "ttir.add"(%55, %73, %74) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x12x100xf32>, tensor<1x32x12x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
    %76 = tensor.empty() : tensor<32x12x100xf32>
    %77 = "ttir.squeeze"(%75, %76) <{dim = 0 : si32}> : (tensor<1x32x12x100xf32>, tensor<32x12x100xf32>) -> tensor<32x12x100xf32>
    %78 = tensor.empty() : tensor<32x100x12xf32>
    %79 = "ttir.transpose"(%77, %78) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x12x100xf32>, tensor<32x100x12xf32>) -> tensor<32x100x12xf32>
    %80 = tensor.empty() : tensor<32x12x12xf32>
    %81 = "ttir.matmul"(%47, %79, %80) : (tensor<32x12x100xf32>, tensor<32x100x12xf32>, tensor<32x12x12xf32>) -> tensor<32x12x12xf32>
    %82 = tensor.empty() : tensor<1x32x12x12xf32>
    %83 = "ttir.unsqueeze"(%81, %82) <{dim = 0 : si32}> : (tensor<32x12x12xf32>, tensor<1x32x12x12xf32>) -> tensor<1x32x12x12xf32>
    %84 = tensor.empty() : tensor<1x32x12x12xf32>
    %85 = "ttir.multiply"(%83, %arg10, %84) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x12x12xf32>, tensor<1xf32>, tensor<1x32x12x12xf32>) -> tensor<1x32x12x12xf32>
    %86 = tensor.empty() : tensor<1x32x12x12xf32>
    %87 = "ttir.add"(%85, %arg1, %86) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x12x12xf32>, tensor<1x1x12x12xf32>, tensor<1x32x12x12xf32>) -> tensor<1x32x12x12xf32>
    %88 = tensor.empty() : tensor<1x32x12x12xf32>
    %89 = "ttir.softmax"(%87, %88) <{dimension = -1 : si32}> : (tensor<1x32x12x12xf32>, tensor<1x32x12x12xf32>) -> tensor<1x32x12x12xf32>
    %90 = tensor.empty() : tensor<32x12x12xf32>
    %91 = "ttir.squeeze"(%89, %90) <{dim = 0 : si32}> : (tensor<1x32x12x12xf32>, tensor<32x12x12xf32>) -> tensor<32x12x12xf32>
    %92 = tensor.empty() : tensor<12x3200xf32>
    %93 = "ttir.matmul"(%1, %arg13, %92) : (tensor<12x3200xf32>, tensor<3200x3200xf32>, tensor<12x3200xf32>) -> tensor<12x3200xf32>
    %94 = tensor.empty() : tensor<1x12x32x100xf32>
    %95 = "ttir.reshape"(%93, %94) <{shape = [1 : i32, 12 : i32, 32 : i32, 100 : i32]}> : (tensor<12x3200xf32>, tensor<1x12x32x100xf32>) -> tensor<1x12x32x100xf32>
    %96 = tensor.empty() : tensor<1x32x12x100xf32>
    %97 = "ttir.transpose"(%95, %96) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x12x32x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
    %98 = tensor.empty() : tensor<1x32x100x12xf32>
    %99 = "ttir.transpose"(%97, %98) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x12x100xf32>, tensor<1x32x100x12xf32>) -> tensor<1x32x100x12xf32>
    %100 = tensor.empty() : tensor<32x100x12xf32>
    %101 = "ttir.squeeze"(%99, %100) <{dim = 0 : si32}> : (tensor<1x32x100x12xf32>, tensor<32x100x12xf32>) -> tensor<32x100x12xf32>
    %102 = tensor.empty() : tensor<32x12x100xf32>
    %103 = "ttir.transpose"(%101, %102) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<32x100x12xf32>, tensor<32x12x100xf32>) -> tensor<32x12x100xf32>
    %104 = tensor.empty() : tensor<32x12x100xf32>
    %105 = "ttir.matmul"(%91, %103, %104) : (tensor<32x12x12xf32>, tensor<32x12x100xf32>, tensor<32x12x100xf32>) -> tensor<32x12x100xf32>
    %106 = tensor.empty() : tensor<1x32x12x100xf32>
    %107 = "ttir.unsqueeze"(%105, %106) <{dim = 0 : si32}> : (tensor<32x12x100xf32>, tensor<1x32x12x100xf32>) -> tensor<1x32x12x100xf32>
    %108 = tensor.empty() : tensor<1x12x32x100xf32>
    %109 = "ttir.transpose"(%107, %108) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<1x32x12x100xf32>, tensor<1x12x32x100xf32>) -> tensor<1x12x32x100xf32>
    %110 = tensor.empty() : tensor<12x3200xf32>
    %111 = "ttir.reshape"(%109, %110) <{shape = [12 : i32, 3200 : i32]}> : (tensor<1x12x32x100xf32>, tensor<12x3200xf32>) -> tensor<12x3200xf32>
    %112 = tensor.empty() : tensor<12x3200xf32>
    %113 = "ttir.matmul"(%111, %arg14, %112) : (tensor<12x3200xf32>, tensor<3200x3200xf32>, tensor<12x3200xf32>) -> tensor<12x3200xf32>
    %114 = tensor.empty() : tensor<1x12x3200xf32>
    %115 = "ttir.unsqueeze"(%113, %114) <{dim = 0 : si32}> : (tensor<12x3200xf32>, tensor<1x12x3200xf32>) -> tensor<1x12x3200xf32>
    return %115 : tensor<1x12x3200xf32>
  }
}
