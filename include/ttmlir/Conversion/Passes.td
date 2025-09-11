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

def TTIRToTTIRGeneric: Pass<"ttir-to-ttir-generic", "::mlir::ModuleOp"> {
  let summary = "Convert named TTIR operations to their ttir.generic form.";
  let description = [{
    This pass converts "named" ops to a nested ttir.generic/linalg.generic structure, with
    ttir.generic denoting the available degrees of parallelism across a grid of cores
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

      %0 = ttir.empty() : tensor<256x768xf32, #layout>
      %1 = "ttir.to_layout"(%arg0, %0) : (tensor<256x768xf32>, tensor<256x768xf32, #layout>) -> tensor<256x768xf32, #layout>
      %2 = ttir.empty() : tensor<768x1024xf32, #layout1>
      %3 = "ttir.to_layout"(%arg1, %2) : (tensor<768x1024xf32>, tensor<768x1024xf32, #layout1>) -> tensor<768x1024xf32, #layout1>
      %4 = ttir.empty() : tensor<256x1024xf32, #layout2>
      %5 = ttir.generic {grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#ttir.thread<compute>]}
          ins(%1, %3 : tensor<256x768xf32, #layout>, tensor<768x1024xf32, #layout1>)
          outs(%4 : tensor<256x1024xf32, #layout2>)  {
      ^compute0(%cb0: memref<8x24x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<24x32x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<8x32x!ttcore.tile<32x32, f32>, #l1_>):
        "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<8x24x!ttcore.tile<32x32, f32>, #l1_>, memref<24x32x!ttcore.tile<32x32, f32>, #l1_>, memref<8x32x!ttcore.tile<32x32, f32>, #l1_>) -> ()
      } : tensor<256x1024xf32, #layout2>
      %6 = ttir.empty() : tensor<256x1024xf32>
      %7 = "ttir.to_layout"(%5, %6) : (tensor<256x1024xf32, #layout2>, tensor<256x1024xf32>) -> tensor<256x1024xf32>
    ```
  }];
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::linalg::LinalgDialect"];
  let constructor = "createTTIRToTTIRGenericPass()";
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
  ];
}

def ConvertTTIRToTTNN: Pass<"convert-ttir-to-ttnn", "::mlir::ModuleOp"> {
  let summary = "Convert TTIR dialect to TTNN dialect.";
  let constructor = "createConvertTTIRToTTNNPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::tt::ttnn::TTNNDialect"];
}

def ConvertTTIRToTTMetal: Pass<"convert-ttir-to-ttmetal", "::mlir::ModuleOp"> {
  let summary = "Convert TTIR dialect to TTMetal dialect.";
  let constructor = "createConvertTTIRToTTMetalPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::tt::ttmetal::TTMetalDialect", "mlir::tt::ttkernel::TTKernelDialect"];
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

def ConvertTTIRToTTKernel: Pass<"convert-ttir-to-ttkernel", "::mlir::ModuleOp"> {
  let summary = "Convert TTIR dialect to TTKernel dialect.";
  let constructor = "createConvertTTIRToTTKernelPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::tt::ttmetal::TTMetalDialect", "mlir::tt::ttkernel::TTKernelDialect", "mlir::arith::ArithDialect"];
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
def ConvertTTIRToTTNNGeneric: Pass<"convert-ttir-to-ttnn-generic", "::mlir::ModuleOp">{
  let summary = "";
  let description = [{

  }];
  let constructor = "createConvertTTIRToTTNNGenericPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::tt::ttnn::TTNNDialect"];
}


#endif // TTMLIR_CONVERSION_PASSES
