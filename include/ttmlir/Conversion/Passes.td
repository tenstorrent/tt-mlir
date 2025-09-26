// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_PASSES
#define TTMLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

#ifdef TTMLIR_ENABLE_STABLEHLO
def ConvertStableHLOToTTIR : Pass<"convert-stablehlo-to-ttir", "::mlir::ModuleOp"> {
let summary = "Convert StableHLO dialect to TTIR dialect.";
  let constructor = "createConvertStableHLOToTTIRPass()";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect", "mlir::sdy::SdyDialect", "mlir::tt::ttir::TTIRDialect"];
  let options = [
    Option<"enablePartialConversion", "enable-partial-conversion", "bool",
      /*default=*/"false",
      "Enable partial conversion mode, allowing unconverted ops to remain">
  ];
}
def ConvertArithToStableHLO : Pass<"convert-arith-to-stablehlo", "::mlir::ModuleOp"> {
let summary = "Convert Arith Dialect to StableHLO dialect.";
  let constructor = "createConvertArithToStableHLOPass()";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect", "mlir::sdy::SdyDialect", "mlir::arith::ArithDialect"];
}
def LegalizeStableHLOCompositeToTTIR : Pass<"legalize-stablehlo-composite-to-ttir", "::mlir::ModuleOp"> {
  let summary = "Legalize StableHLO composite operations directly to TTIR.";
  let constructor = "createLegalizeStableHLOCompositeToTTIRPass()";
  let description = [{
    This pass legalizes StableHLO composite operations based on the name attribute
    directly to the corresponding TTIR operations.
    This allows us to avoid the need for brittle fusion patterns.
  }];

  let dependentDialects = ["mlir::stablehlo::StablehloDialect", "mlir::tt::ttir::TTIRDialect"];
}
#endif

def ConvertTosaToTTIR : Pass<"convert-tosa-to-ttir", "::mlir::ModuleOp"> {
  let summary = "Convert TOSA dialect to TTIR dialect.";
  let constructor = "createConvertTosaToTTIRPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect"];
}

def TTIRToTTIRDecomposition: Pass<"ttir-to-ttir-decomposition", "::mlir::ModuleOp"> {
  let summary = "Decomposes TTIR operations into simpler TTIR operations.";
  let constructor = "createTTIRToTTIRDecompositionPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect"];

  let options = [
    Option<"decompConfig", "config", "::mlir::tt::DecompMode", /*default=*/"::mlir::tt::DecompMode::TTNN",
           "Decomposition configuration for different backends",
           [{::llvm::cl::values(
             clEnumValN(::mlir::tt::DecompMode::TTNN, "ttnn",
                        "Default configuration for TTNN backend"),
             clEnumValN(::mlir::tt::DecompMode::TTMetal, "ttmetal",
                        "Configuration for TTMetal backend"),
             clEnumValN(::mlir::tt::DecompMode::CPUFallback, "cpu-fallback",
                        "Configuration for CPU fallback path")
           )}]>
  ];
}


def TTIRToD2M: Pass<"ttir-to-d2m", "::mlir::ModuleOp"> {
  let summary = "Convert named TTIR operations to their D2M form.";
  let description = [{
    This pass converts ttir "named" ops to a nested d2m.generic/linalg.generic structure, with
    d2m.generic denoting the available degrees of parallelism across a grid of cores
    and linalg.generic adding another level of nesting for a single core's data movement/compute
    task. This conversion will do an appropriate decomposition of the original op
    into lower-level tiled/blocked ops.
    Additionally it handles layout transitions from the original graph on tensors, to tensors
    with a ttcore.metal_layout encoding attribute.  This enables ops to override the default layout
    if needed.
    Example:
    ```mlir
      %0 = ttir.empty() : tensor<256x1024xf32>
      %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<256x768xf32>, tensor<768x1024xf32>, tensor<256x1024xf32>) -> tensor<256x1024xf32>
    ```
    becomes (post-canonicalization)
    ```mlir
      #l1_ = #ttcore.memory_space<l1>
      #map = affine_map<(d0, d1, d2) -> (d0, d2)>
      #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
      #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
      #parallel = #ttcore.iterator_type<parallel>
      #reduction = #ttcore.iterator_type<reduction>
      #layout = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<8x24x!ttcore.tile<32x32, f32>, #l1_>>
      #layout1 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<24x32x!ttcore.tile<32x32, f32>, #l1_>>
      #layout2 = #ttcore.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<8x32x!ttcore.tile<32x32, f32>, #l1_>>
      %0 = d2m.empty() : tensor<256x768xf32, #layout>
      %1 = "d2m.to_layout"(%arg0, %0) : (tensor<256x768xf32>, tensor<256x768xf32, #layout>) -> tensor<256x768xf32, #layout>
      %2 = d2m.empty() : tensor<768x1024xf32, #layout1>
      %3 = "d2m.to_layout"(%arg1, %2) : (tensor<768x1024xf32>, tensor<768x1024xf32, #layout1>) -> tensor<768x1024xf32, #layout1>
      %4 = d2m.empty() : tensor<256x1024xf32, #layout2>
      %5 = d2m.generic {grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
          ins(%1, %3 : tensor<256x768xf32, #layout>, tensor<768x1024xf32, #layout1>)
          outs(%4 : tensor<256x1024xf32, #layout2>)  {
      ^compute0(%cb0: memref<8x24x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<24x32x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<8x32x!ttcore.tile<32x32, f32>, #l1_>):
        "d2m.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<8x24x!ttcore.tile<32x32, f32>, #l1_>, memref<24x32x!ttcore.tile<32x32, f32>, #l1_>, memref<8x32x!ttcore.tile<32x32, f32>, #l1_>) -> ()
      } : tensor<256x1024xf32, #layout2>
      %6 = d2m.empty() : tensor<256x1024xf32>
      %7 = "d2m.to_layout"(%5, %6) : (tensor<256x1024xf32, #layout2>, tensor<256x1024xf32>) -> tensor<256x1024xf32>
    ```
  }];
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::tt::d2m::D2MDialect", "mlir::linalg::LinalgDialect"];
  let constructor = "createTTIRToD2MPass()";
  let options = [
      Option<"defaultInputMemSpace", "default-input-memspace", "ttcore::MemorySpace", /*default=*/"ttcore::MemorySpace::DeviceL1", "Set default memspace for input tensors",
          [{::llvm::cl::values(
            clEnumValN(ttcore::MemorySpace::DeviceL1, "l1", "L1"),
            clEnumValN(ttcore::MemorySpace::DeviceDRAM, "dram", "DRAM")
          )}]>,
      Option<"defaultOutputMemSpace", "default-output-memspace", "ttcore::MemorySpace", /*default=*/"ttcore::MemorySpace::DeviceL1", "Set default memspace for output tensors",
          [{::llvm::cl::values(
            clEnumValN(ttcore::MemorySpace::DeviceL1, "l1", "L1"),
            clEnumValN(ttcore::MemorySpace::DeviceDRAM, "dram", "DRAM")
          )}]>,
      ListOption<"overrideDeviceShape", "override-device-shape", "int64_t",
          "Override the device shape.">,
      Option<"collapseTensorsTo2D", "collapse-tensors-2d", "bool", /*default=*/"true", "Default collapse all tensors to 2D">,
      Option<"ttnnMode", "ttnn-mode", "bool", /*default=*/"false", "Enable translation of TTNN Tensors">,
  ];
}

<<<<<<< HEAD
def ArithToD2MTileOps : Pass<"arith-to-d2m-tile-ops", "::mlir::ModuleOp"> {
  let summary = "Convert Arith to D2M Tile ops dialect.";
  let description = [{
    Take arith ops that operate over tiles and convert them to d2m tile ops.
    ```mlir
    %0 = arith.negf %arg0 : !ttcore.tile<32x32, f32>
    ```
    Becomes:
    ```mlir
    %0 = "d2m.tile_negative"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    ```
  }];
  let dependentDialects = ["mlir::tt::d2m::D2MDialect", "mlir::linalg::LinalgDialect"];
  let constructor = "createConvertArithToD2MTileOpsPass()";
}

def MathToD2MTileOps : Pass<"math-to-d2m-tile-ops", "::mlir::ModuleOp"> {
  let summary = "Convert Math to D2M Tile ops dialect.";
  let description = [{
    Take math ops that operate over tiles and convert them to d2m tile ops.
    ```mlir
    %0 = math.abs %arg0 : !ttcore.tile<32x32, f32>
    ```
    Becomes:
    ```mlir
    %0 = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    ```
  }];
  let dependentDialects = ["mlir::tt::d2m::D2MDialect", "mlir::linalg::LinalgDialect"];
  let constructor = "createConvertMathToD2MTileOpsPass()";
}

=======
>>>>>>> 671aaeffd (final feedback + reverts)
def ConvertTTIRToTTNN: Pass<"convert-ttir-to-ttnn", "::mlir::ModuleOp"> {
  let summary = "Convert TTIR dialect to TTNN dialect.";
  let constructor = "createConvertTTIRToTTNNPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::tt::ttnn::TTNNDialect"];
}

def ConvertD2MToTTMetal: Pass<"convert-d2m-to-ttmetal", "::mlir::ModuleOp"> {
  let summary = "Convert D2M dialect to TTMetal dialect.";
  let constructor = "createConvertD2MToTTMetalPass()";
  let dependentDialects = ["mlir::tt::d2m::D2MDialect", "mlir::tt::ttmetal::TTMetalDialect", "mlir::tt::ttkernel::TTKernelDialect"];
  let options = [
    Option<"mathFidelity", "math-fidelity", "mlir::tt::ttmetal::MathFidelity",
           /*default=*/"mlir::tt::ttmetal::MathFidelity::HiFi4",
           "Set the math fidelity for compute kernels",
           [{::llvm::cl::values(
             clEnumValN(mlir::tt::ttmetal::MathFidelity::LoFi, "LoFi", "LoFi"),
             clEnumValN(mlir::tt::ttmetal::MathFidelity::HiFi2, "HiFi2", "HiFi2"),
             clEnumValN(mlir::tt::ttmetal::MathFidelity::HiFi3, "HiFi3", "HiFi3"),
             clEnumValN(mlir::tt::ttmetal::MathFidelity::HiFi4, "HiFi4", "HiFi4")
           )}]>
  ];
}

def ConvertD2MToTTKernel: Pass<"convert-d2m-to-ttkernel", "::mlir::ModuleOp"> {
  let summary = "Convert D2M dialect to TTKernel dialect.";
  let constructor = "createConvertD2MToTTKernelPass()";
  let dependentDialects = ["mlir::tt::d2m::D2MDialect", "mlir::tt::ttmetal::TTMetalDialect", "mlir::tt::ttkernel::TTKernelDialect", "mlir::arith::ArithDialect"];
  let options = [
    Option<"ttnnMode", "ttnn-mode", "bool", /*default=*/"false", "Enable TTNN mode">
  ];
}

def ConvertTTNNToEmitC : Pass<"convert-ttnn-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Convert TTNN dialect to EmitC dialect.";
  let constructor = "createConvertTTNNToEmitCPass()";
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::tt::ttnn::TTNNDialect"];
}

def ConvertTTNNToEmitPy : Pass<"convert-ttnn-to-emitpy", "::mlir::ModuleOp"> {
  let summary = "Convert TTNN dialect to EmitPy dialect.";
  let constructor = "createConvertTTNNToEmitPyPass()";
  let dependentDialects = ["mlir::tt::emitpy::EmitPyDialect", "mlir::tt::ttnn::TTNNDialect"];
}

def ConvertTTKernelToEmitC : Pass<"convert-ttkernel-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Convert TTKernel dialect to EmitC dialect.";
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::func::FuncDialect",
                           "mlir::tt::ttkernel::TTKernelDialect"];
}

def ConvertSFPIToEmitC : Pass<"convert-sfpi-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Convert SFPI dialect to EmitC dialect using GCC RISC-V Tenstorrent builtins.";
  let description = [{
    This pass converts SFPI (SFPU Programming Interface) operations to EmitC
    operations that emit calls to GCC RISC-V Tenstorrent builtin functions.

    SFPI vector types (vector<4x8xf32>) are converted to __rvtt_vec_t and
    operations are mapped to corresponding __builtin_rvtt_sfp* functions.
  }];
  let constructor = "createConvertSFPIToEmitCPass()";
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::func::FuncDialect",
                           "mlir::tt::sfpi::SFPIDialect"];
}

def ConvertTTIRToLinalg: Pass<"convert-ttir-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Convert TTIR dialect to Linalg dialect.";
  let description = [{
    Conversion pass to convert TTIR ops with defined conversion pattern into linalg ops, with broadcast and collapse tensor ops as needed.
    Example:
    Input:
      func.func @add_with_broadcast(
        %arg0: tensor<32x32xf32>,
        %arg1: tensor<32x1xf32>,
        %arg2: tensor<32x32xf32>
      ) -> tensor<32x32xf32> {
        %1 = "ttir.add"(%arg0, %arg1, %arg2) : (tensor<32x32xf32>, tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        return %1 : tensor<32x32xf32>
      }
    Output:
      func.func @add_with_broadcast(
        %arg0: tensor<32x32xf32>,
        %arg1: tensor<32x1xf32>,
        %arg2: tensor<32x32xf32>
      ) -> tensor<32x32xf32> {
        %collapsed = tensor.collapse_shape %arg1 [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
        %0 = ttir.empty() : tensor<32x32xf32>
        %broadcasted = linalg.broadcast ins(%collapsed : tensor<32xf32>) outs(%0 : tensor<32x32xf32>) dimensions = [1]
        %1 = linalg.add ins(%arg0, %broadcasted : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
        return %1 : tensor<32x32xf32>
    }
  }];
  let constructor = "createConvertTTIRToLinalgPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::linalg::LinalgDialect", "mlir::tosa::TosaDialect"];
}

def ConvertTTNNToTTIR : Pass<"convert-ttnn-to-ttir", "::mlir::ModuleOp"> {
let summary = "Convert TTNN dialect to TTIR dialect.";
  let constructor = "createConvertTTNNToTTIRPass()";
  let dependentDialects = ["mlir::tt::ttnn::TTNNDialect", "mlir::tt::ttir::TTIRDialect"];
}

def ConvertD2MToTTNN: Pass<"convert-d2m-to-ttnn", "::mlir::ModuleOp">{
  let summary = "Conversion to exit D2M flow into TTNN";
  let description = [{
    Conversion pass to convert from TTIR to TTNN by constructing a TTNN Generic op using the TTIRGeneric op.
    For now assume, input into the TTIRGeneric will always be a stream layout, output is `ttnn_metal_layout_cast`.
    The input to said stream is also from a cast op. This pass also lifts any ttir.empty with attached ttnn_layout
    back into a ttnn.empty. The current version does not support intermediate tensor allocations.

    Example:
    Input IR:
      %ttnn_input_l1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #l1_memory_config}> : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #l1_layout>
      %ttnn_output_l1 = ttir.empty() : tensor<32x32xf32, #l1_layout>
      %metal_input_l1 = ttir.ttnn_metal_layout_cast %ttnn_input_l1 : tensor<32x32xf32, #l1_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>
      %metal_output_l1 = ttir.ttnn_metal_layout_cast %ttnn_output_l1 : tensor<32x32xf32, #l1_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>
      %storage = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>
      %stream  = "ttir.stream_layout" (%metal_input_l1, %storage)
            : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>,
              memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>)
            -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #ttcore.memory_space<l1>>
      ttir.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#ttir.thread<datamovement, @read_kernel>, #ttir.thread<datamovement, @write_kernel>, #ttir.thread<compute, @compute_kernel0>]}
          ins(%stream : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #ttcore.memory_space<l1>>)
          outs(%metal_output_l1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #ttcore.memory_space<l1>>)
      %output_l1 = ttir.ttnn_metal_layout_cast

    Output IR (post canonicalization):
      %1 = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>, <physical>>>}> : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout1>
      %2 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <block_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x32>, <row_major>, <physical>>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #ttnn_layout1>
      "ttnn.generic"(%1, %2) ...
  }];
  let constructor = "createConvertD2MToTTNNPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::tt::ttnn::TTNNDialect"];
}


#endif // TTMLIR_CONVERSION_PASSES
