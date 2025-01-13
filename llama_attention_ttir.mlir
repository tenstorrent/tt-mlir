// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

// ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=/code/tt-mlir/ttrt-artifacts/system_desc.ttsys" llama_attention_ttir.mlir > taps.mlir
// ttmlir-translate --ttnn-to-flatbuffer taps.mlir > taps.ttnn

module @SelfAttention attributes {tt.system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_cores = {worker = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  1x0,  1x1,  1x2,  1x3,  1x4,  1x5,  1x6,  1x7,  2x0,  2x1,  2x2,  2x3,  2x4,  2x5,  2x6,  2x7,  3x0,  3x1,  3x2,  3x3,  3x4,  3x5,  3x6,  3x7,  4x0,  4x1,  4x2,  4x3,  4x4,  4x5,  4x6,  4x7,  5x0,  5x1,  5x2,  5x3,  5x4,  5x5,  5x6,  5x7,  6x0,  6x1,  6x2,  6x3,  6x4,  6x5,  6x6,  6x7,  7x0,  7x1,  7x2,  7x3,  7x4,  7x5,  7x6,  7x7] dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0], [3 : i32], [ 0x0x0x0]>} {
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