// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_PIPELINES_TTNNPIPELINES_H
#define TTMLIR_DIALECT_TTNN_PIPELINES_TTNNPIPELINES_H

#include "ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTNN/Utils/MathFidelityParser.h"
#include "ttmlir/Dialect/TTNN/Utils/MemoryLayoutAnalysisParams.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace tt::tt_metal::distributed {
class MeshDevice;
} // namespace tt::tt_metal::distributed

namespace mlir::tt::ttnn {
// TTIR to TTNN Device pipeline options.
//
struct TTIRToTTNNDevicePipelineOptions
    : public PassPipelineOptions<TTIRToTTNNDevicePipelineOptions> {
  // Optimization level controls multiple optimization passes.
  // Level 0 (default): All optimizer passes disabled.
  // Level 1: All optimizer passes enabled. Memory layout analysis is disabled.
  // Moderate compile time. Level 2: All optimizer passes enabled with memory
  // layout analysis (sharding). Longest compile time. Individual options can
  // override the optimization level settings.
  Option<int> optimizationLevel{
      *this, OptionNames::optimizationLevel,
      llvm::cl::desc(
          "Optimization level: 0=all optimizer passes disabled (fastest "
          "compile), "
          "1=optimizer passes enabled except sharding (moderate compile), "
          "2=all optimizer passes including sharding enabled (longest "
          "compile)."),
      llvm::cl::init(0)};

  // Enable all optimizer passes.
  // If not explicitly set, determined by optimization_level.
  mutable Option<bool> optimizerPassEnabled{
      *this, OptionNames::optimizerPassEnabled,
      llvm::cl::desc("Determine and set max valid grid for Op execution."),
      llvm::cl::init(false)};

  // If this option is true, run a pass that checks if all ops relevant
  // to the optimizer (e.g. toLayout is ignored) have unique named locations.
  // If not, it will emit an error. This is necessary for the overrides to be
  // applied correctly.
  Option<bool> checkUniqueLocations{
      *this, "check-unique-locs",
      llvm::cl::desc("Check if all operations have unique locations."),
      llvm::cl::init(false)};

  // Option to manually insert TTNN_ToLayoutOp for specific op's operand.
  // The format is a comma separated list of op names and operand index
  // separated by ':' separator.
  //
  // Full Example: "op1=0,op2=0:1"
  //
  // This will insert one memory reconfig op responsible for resharding the
  // op1's first operand and two memory reconfig ops responsible for resharding
  // the op2's first and second operand.
  //
  // Note: This option is only valid if optimizerPassEnabled is true.
  //
  Option<llvm::StringMap<InsertMemReconfigParams>, InsertMemReconfigParser>
      insertMemReconfig{
          *this, OptionNames::insertMemReconfig,
          llvm::cl::desc(
              "Manually insert memory reconfig op for specific op's operand."),
          llvm::cl::init(llvm::StringMap<InsertMemReconfigParams>())};

  // Option to override output layout for specific operations. You can
  // override any number or combination of layout parameters. If not all are
  // overridden, the remaining ones will be inferred with all possible
  // combinations generated in LegalLayoutAnalysis. The format is a
  // comma-separated list of operation names followed by the output layout
  // parameters, separated by :. The order of parameters does not matter; the
  // parser will deduce which one is being overridden based on its value.
  //
  // op_name=grid_size:memory_space:tensor_memory_layout:memory_layout:data_type
  //
  // * grid_size=2x2
  // * memory_space: system, mmio, dram or l1
  // * tensor_memory_layout: none, interleaved, height_sharded,
  //   width_sharded or block_sharded
  // * memory_layout: row_major or tile
  // * data_type: f32, f16, bf16, bfp_f8, bfp_bf8, bfp_f4, bfp_bf4, bfp_f2,
  //   bfp_bf2, u32, u16, u8
  //
  // Full Example:
  // "op1=2x2:dram:interleaved:tile:fp32,op2=4x4:l1:block_sharded:row_major:f16"
  // Partial Example:
  // "op1=2x2:block_sharded"
  //
  //
  // Note: This option is only valid if optimizerPassEnabled is true.
  //
  Option<llvm::StringMap<OutputLayoutOverrideParams>,
         OutputLayoutOverrideParser>
      overrideOutputLayout{
          *this, OptionNames::overrideOutputLayout,
          llvm::cl::desc("Override output tensor layout for specific ops."),
          llvm::cl::init(llvm::StringMap<OutputLayoutOverrideParams>())};

  // Option to override Conv2d configuration for specific operations.
  // If not all parameters are overridden, the remaining ones will be set to
  // default (see in tt-metal). The format is a comma-separated list of
  // operation names followed by the conv2d config parameters as
  // `param_name#param_value`, separated by :. This option is only valid if
  // optimizerPassEnabled (enable-optimizer) is true.
  //
  // Full Example:
  // override-conv2d-config=conv2d_1=dtype#bf16:weights_dtype#bf16:activation#relu:deallocate_activation#false:reallocate_halo_output#true:act_block_h_override#0:act_block_w_div#1:reshard_if_not_optimal#false:override_sharding_config#false:shard_layout#block_sharded:core_grid#0:transpose_shards#true:output_layout#row_major:enable_act_double_buffer#false:enable_weights_double_buffer#false
  // Partial Example:
  // "conv2d_1=enable_weights_double_buffer#true:activation#none,conv2d_2=dtype#bf16"
  //
  // * dtype: [bf16, f32, f16, bfp_f8, bfp_bf8, bfp_f4, bfp_bf4, bfp_f2,
  // bfp_bf2, u32, u16, u8]
  // * weights_dtype: [bf16, f32, f16, bfp_f8, bfp_bf8, bfp_f4, bfp_bf4, bfp_f2,
  // bfp_bf2, u32, u16, u8]
  // * activation: [none, relu]
  // * deallocate_activation: [true, false]
  // * reallocate_halo_output: [true, false]
  // * act_block_h_override: uint32_t (multiple of 32)
  // * act_block_w_div: uint32_t
  // * reshard_if_not_optimal: [true, false]
  // * override_sharding_config: [true, false]
  // * shard_layout: [block_sharded, interleaved, height_sharded,
  // width_sharded]
  // * core_grid:
  // * transpose_shards: [true, false]
  // * output_layout: [row_major, tile]
  // * enable_act_double_buffer: [true, false]
  // * enable_weights_double_buffer: [true, false]
  //
  // For more details on parameter values see conv2d_op.hpp in tt-metal.
  //
  Option<llvm::StringMap<Conv2dConfigOverrideParams>,
         Conv2dConfigOverrideParser>
      overrideConv2dConfig{
          *this, OptionNames::overrideConv2dConfig,
          llvm::cl::desc("Override Conv2d configuration for specific ops."),
          llvm::cl::init(llvm::StringMap<Conv2dConfigOverrideParams>())};

  // Enable memory layout analysis for performant tensor layouts (sharding).
  // If not explicitly set, determined by optimization_level.
  mutable Option<bool> memoryLayoutAnalysisEnabled{
      *this, OptionNames::memoryLayoutAnalysisEnabled,
      llvm::cl::desc("Enable memory layout optimization."),
      llvm::cl::init(false)};

  // If this option is true, run L1 interleaved layout analysis.
  //
  Option<bool> l1InterleavedFallbackAnalysisEnabled{
      *this, OptionNames::l1InterleavedFallbackAnalysisEnabled,
      llvm::cl::desc("Enable DRAM to L1 interleaved fallback optimization."),
      llvm::cl::init(false)};

  // If this option is true, insert memory reconfiguration ops.
  //
  Option<bool> memReconfigEnabled{
      *this, OptionNames::memReconfigEnabled,
      llvm::cl::desc("Memory layout reconfiguration pass."),
      llvm::cl::init(true)};

  // Specify policy for memory layout analysis.
  //
  Option<MemoryLayoutAnalysisPolicyType, MemoryLayoutAnalysisPolicyTypeParser>
      memoryLayoutAnalysisPolicy{
          *this, OptionNames::memoryLayoutAnalysisPolicy,
          llvm::cl::desc("Specify policy for memory layout analysis."),
          llvm::cl::init(MemoryLayoutAnalysisPolicyType::DFSharding)};

  // Option to provide a system descriptor flatbuffer file to compile
  // against.
  //
  Option<std::string> systemDescPath{
      *this, OptionNames::systemDescPath,
      llvm::cl::desc(
          "Pass in a system descriptor flatbuffer to compile against."),
      llvm::cl::init("")};

  // Option to provide a fallback mock system descriptor arch to compile
  // against.
  //
  Option<ttcore::Arch> mockSystemDescArch{
      *this, OptionNames::mockSystemDescArch,
      llvm::cl::desc(
          "Arch name for constructing a mock system descriptor in lieu of "
          "system-desc-path."),
      llvm::cl::values(clEnumValN(ttcore::Arch::WormholeB0, "wormhole_b0",
                                  "Use mock wormhole_b0 system desc."),
                       clEnumValN(ttcore::Arch::Blackhole, "blackhole",
                                  "Use mock blackhole system desc.")),
      llvm::cl::init(ttcore::Arch::WormholeB0)};

  // Option to override maximum number of sharded layouts to be generated
  // in legal layout analysis.
  //
  Option<int64_t> maxLegalLayouts{
      *this, OptionNames::maxLegalLayouts,
      llvm::cl::desc("Override maximum number of sharded layouts for legal "
                     "layout analysis."),
      llvm::cl::init(8)};

  ListOption<int64_t> meshShape{
      *this, OptionNames::meshShape,
      llvm::cl::desc("Set the multi-device mesh shape.")};

  Option<bool> rowMajorEnabled{
      *this, "row-major-enabled",
      llvm::cl::desc(
          "Enable row major layout generation in legal layout analysis."),
      llvm::cl::init(false)};
  // Option to override maximum percent of L1 storage that can be used
  // by tensors in Optimizer analysis.
  // This is a value between 0.0 and 1.0, where 1.0 means that the entire L1
  // storage can be used by tensors.
  // The default value is 0.95.
  //
  Option<float> tensorL1UsageCap{
      *this, OptionNames::tensorL1UsageCap,
      llvm::cl::desc("Override tensor L1 usage cap in L1 Interleaved Fallback "
                     "Analysis and Memory Layout Analysis. [0.0-1.0]"),
      llvm::cl::init(0.95f)};

  // Option to enable/disable the workaround pass.
  //
  Option<bool> disableWorkarounds{
      *this, "disable-workarounds",
      llvm::cl::desc("An option to disable/enable the whole workaround pass. "
                     "If set to true, the workaround pass is disabled."),
      llvm::cl::init(false)};

  Option<bool> layoutWorkaroundsEnabled{
      *this, "enable-layout-workaround-pass",
      llvm::cl::desc("Enable layout workaround pass. Always false when "
                     "optimizer pass is enabled."),
      llvm::cl::init(true)};

  Option<bool> decompositionWorkaroundsEnabled{
      *this, "enable-decomposition-workaround-pass",
      llvm::cl::desc("Enable decomposition workaround pass."),
      llvm::cl::init(true)};

  Option<bool> implicitBroadcastFoldingEnabled{
      *this, "enable-implicit-broadcast-folding-pass",
      llvm::cl::desc("Enable implicit broadcast folding pass."),
      llvm::cl::init(true)};

  Option<bool> eraseInverseOpsEnabled{
      *this, "enable-erase-inverse-ops-pass",
      llvm::cl::desc("Enable erase inverse ops pass."), llvm::cl::init(true)};

  Option<bool> enableQuantDequantConversion{
      *this, "enable-quant-dequant-conversion-pass",
      llvm::cl::desc("Enable quant-dequant conversion pass."),
      llvm::cl::init(true)};

  Option<bool> enableFusing{*this, "enable-fusing-pass",
                            llvm::cl::desc("Enable fusing pass."),
                            llvm::cl::init(true)};

  Option<bool> enableD2MFusing{*this, "enable-d2m-fusing-pass",
                               llvm::cl::desc("Enable D2M fusing pass."),
                               llvm::cl::init(false)};

  // Enable fusing of conv2d + multiply pattern.
  // If not explicitly set, determined by optimization_level.
  mutable Option<bool> enableFusingConv2dWithMultiplyPattern{
      *this, "enable-fusing-conv2d-with-multiply-pattern",
      llvm::cl::desc("Enable Conv2dWithMultiply pattern in the fusing pass."),
      llvm::cl::init(false)};

  // Enable fusing of permute + matmul/linear pattern.
  Option<bool> enablePermuteMatmulFusion{
      *this, "enable-permute-matmul-fusion",
      llvm::cl::desc(
          "Fuse permute ops into matmul/linear transpose attributes."),
      llvm::cl::init(true)};

  Option<ttcore::TTArgumentTypeMap, ttcore::ArgumentTypeMapParser>
      argumentTypeMap{
          *this, ttcore::OptionNames::argumentTypes,
          llvm::cl::desc(
              "Map of function name to argument types. To use this option in "
              "the "
              "command line, you must provide a whitespace-free string\n\t"
              " which is a sequence of phrases in the form "
              "\"<FUNC_NAME_STR>=<ARG_TYPES>\" separated by semicolons, where "
              "<FUNC_NAME_STR>\n\t"
              " is the name of a function and <ARG_TYPES> is a sequence of "
              "argument types separated by commas. Each of which must be "
              "one\n\t"
              " of \"input\", \"parameter\" or \"constant\". \n\t"
              " Example: "
              "\"argument-types=forward=input,parameter,parameter,constant\""
              "\n\n"),
          llvm::cl::init(ttcore::TTArgumentTypeMap())};

  // TODO (azecevic): This pass is causing a lot of memory consumption and is
  // disabled by default (https://github.com/tenstorrent/tt-mlir/issues/2512).
  Option<bool> removeDeadValuesEnabled{
      *this, "enable-remove-dead-values",
      llvm::cl::desc("Enable --remove-dead-values optimization pass."),
      llvm::cl::init(false)};

  Option<bool> enableConstEval{
      *this, "enable-const-eval",
      llvm::cl::desc("Enable const-eval optimization pass."),
      llvm::cl::init(true)};

  // Enable CPU-hoisting for const-eval subgraphs.
  Option<bool> enableCPUHoistedConstEval{
      *this, "enable-cpu-hoisted-const-eval",
      llvm::cl::desc("Enable hoisting const-eval ops to CPU module."),
      llvm::cl::init(false)};

  // Enable heuristic CPU-hoisting for small-tensor ops.
  Option<bool> enableHeuristicCPUHoist{
      *this, "enable-heuristic-cpu-hoist",
      llvm::cl::desc("Enable heuristic CPU hoisting for small-tensor ops."),
      llvm::cl::init(false)};

  Option<int64_t> heuristicCPUHoistElementThreshold{
      *this, "heuristic-cpu-hoist-element-threshold",
      llvm::cl::desc("Max result element count for heuristic CPU hoisting."),
      llvm::cl::init(1024)};

  Option<int64_t> heuristicCPUHoistInputElementThreshold{
      *this, "heuristic-cpu-hoist-input-element-threshold",
      llvm::cl::desc("Max input element count for heuristic CPU hoisting."),
      llvm::cl::init(8192)};

  // Force const-eval function inputs to system memory.
  Option<bool> enableConstEvalInputsToSystemMemory{
      *this, "enable-const-eval-inputs-to-system-memory",
      llvm::cl::desc("Force const-eval function inputs to system memory."),
      llvm::cl::init(true)};

  // Force CPU-hoisted function inputs to system memory.
  Option<bool> enableCPUHoistedInputsToSystemMemory{
      *this, "enable-cpu-hoisted-inputs-to-system-memory",
      llvm::cl::desc("Force CPU-hoisted function inputs to system memory."),
      llvm::cl::init(true)};

  Option<bool> enableTrace{*this, "enable-trace",
                           llvm::cl::desc("Enable trace optimization pass."),
                           llvm::cl::init(false)};

  // Option to specify the target bit width for quantized data types.
  Option<uint32_t> quantBitWidth{
      *this, "target-bit-width",
      llvm::cl::desc(
          "Target integer bit width for quantized types (8, 16, 32, 64). "
          "Set to enable quantized data type conversion pass. "
          "Leave empty to disable the pass."),
      llvm::cl::init(32)};

  Option<bool> enableBfp8Conversion{
      *this, "enable-bfp8-conversion",
      llvm::cl::desc("Enables conversion from bfloat16 to bfp8_b."),
      llvm::cl::init(false)};

  Option<bool> experimentalBfp8Weights{
      *this, "experimental-bfp8-weights",
      llvm::cl::desc(
          "Experimental: Enables conversion of weight tensors in "
          "matrix multiplication and convolution operations to bfp8_b."),
      llvm::cl::init(false)};

  // ComputeKernelConfig options
  // Note: computeCfgMathFidelity default value is HiFi4
  // And computeCfgFp32DestAccEn default value is true.
  // This is done as part of generality effort,
  // to boost accuracy on all operations exposing compute kernel config by
  // default.
  Option<OptionalMathFidelity> computeCfgMathFidelity{
      *this, "compute-cfg-math-fidelity",
      llvm::cl::desc("Set math fidelity for all ttnn operations exposing "
                     "compute kernel config."),
      llvm::cl::values(
          clEnumValN(OptionalMathFidelity::LoFi, "lofi", "Low fidelity math"),
          clEnumValN(OptionalMathFidelity::HiFi2, "hifi2", "High fidelity 2"),
          clEnumValN(OptionalMathFidelity::HiFi3, "hifi3", "High fidelity 3"),
          clEnumValN(OptionalMathFidelity::HiFi4, "hifi4", "High fidelity 4"),
          clEnumValN(OptionalMathFidelity::Undefined, "undefined",
                     "Undefined math fidelity")),
      llvm::cl::init(OptionalMathFidelity::HiFi4)};

  Option<bool> computeCfgFp32DestAccEn{
      *this, "compute-cfg-fp32-dest-acc-en",
      llvm::cl::desc("Set fp32 destination accumulation for all ttnn "
                     "operations exposing compute kernel config."),
      llvm::cl::init(true)};

  Option<bool> ttnnPerfMetricsEnabled{
      *this, "ttnn-perf-metrics-enabled",
      llvm::cl::desc("Enable performance metrics collection."),
      llvm::cl::init(false)};

  // Optional output file path for performance metrics JSON. If not provided,
  // defaults to generate filename based on module or function name in
  // "perf_metrics" directory.
  Option<std::string> ttnnPerfMetricsOutputFile{
      *this, "ttnn-perf-metrics-output-file",
      llvm::cl::desc("Output file path for the performance metrics JSON."),
      llvm::cl::init("")};

  Option<bool> ttnnPerfMetricsVerboseOutputEnabled{
      *this, "ttnn-perf-metrics-verbose-output-enabled",
      llvm::cl::desc(
          "Enable verbose output with per-operation details in metrics."),
      llvm::cl::init(true)};

  Option<uint32_t> maxFallbackAttempts{
      *this, "max-fallback-attempts",
      llvm::cl::desc(
          "Maximum number of fallback attempts per operation in Operation "
          "Validation and Fallback pass. 0 means unlimited attempts."),
      llvm::cl::init(10000)};

  // Option to provide a pointer to an already opened device. When provided,
  // the optimizer will use this device instead of opening a new one.
  // This allows frontends to pass in an active device without closing it.
  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> devicePtr = nullptr;

  // Resolve options controlled by optimization_level.
  void resolveOptimizationLevelOptions() const {
    // Validate optimization_level is in valid range.
    if (optimizationLevel < 0 || optimizationLevel > 2) {
      llvm::reportFatalUsageError(
          "Invalid optimization_level: " + llvm::Twine(optimizationLevel) +
          ". Must be 0, 1, or 2.");
    }

    // Only apply optimization_level if user didn't explicitly set the option.
    // Use getNumOccurrences() to detect explicit user settings.
    if (optimizerPassEnabled.getNumOccurrences() == 0) {
      optimizerPassEnabled = (optimizationLevel >= 1);
    }
    if (enableFusingConv2dWithMultiplyPattern.getNumOccurrences() == 0) {
      enableFusingConv2dWithMultiplyPattern = (optimizationLevel >= 1);
    }
    if (memoryLayoutAnalysisEnabled.getNumOccurrences() == 0) {
      memoryLayoutAnalysisEnabled = (optimizationLevel >= 2);
    }
  }
};

// TTNN to EmitC Device pipeline options.
//
struct TTNNToEmitCDevicePipelineOptions
    : public PassPipelineOptions<TTNNToEmitCDevicePipelineOptions> {
  Option<bool> targetDylib{*this, "target-dylib",
                           llvm::cl::desc("Tailor passes for dylib target."),
                           llvm::cl::init(false)};

  Option<bool> tryRecoverStructure{
      *this, "try-recover-structure",
      llvm::cl::desc(
          "Enable pipelines and passes that try to recover structure of the "
          "original IR/code. Highly experimental; please file issues at "
          "https://github.com/tenstorrent/tt-mlir/issues"),
      llvm::cl::init(false)};

  Option<bool> tuplifyInputIfEmpty{
      *this, "tuplify-input-if-empty",
      llvm::cl::desc("Whether to create an empty tuple if no inputs to forward "
                     "function. This should only be used if the `target-dylib` "
                     "option is set to `true`"),
      llvm::cl::init(false)};

  Option<bool> loadInputTensorsFromDisk{
      *this, "load-input-tensors-from-disk",
      llvm::cl::desc("Load input tensors from disk using ttnn.load_tensor "
                     "instead of generating synthetic inputs with ttnn.ones"),
      llvm::cl::init(false)};

  Option<std::string> tensorLoadDirectory{
      *this, "tensor-load-directory",
      llvm::cl::desc("Directory path where input tensors are stored"),
      llvm::cl::init("")};

  Option<std::string> tensorLoadFilePrefix{
      *this, "tensor-load-file-prefix",
      llvm::cl::desc("Prefix for input tensor files"), llvm::cl::init("arg")};
};

// TTNN to EmitPy Device pipeline options.
//
struct TTNNToEmitPyDevicePipelineOptions
    : public PassPipelineOptions<TTNNToEmitPyDevicePipelineOptions> {
  Option<bool> targetModule{
      *this, "target-module",
      llvm::cl::desc("Tailor passes for Python module target. When enabled, "
                     "the entry function is named 'forward' with tuple of "
                     "tensors and device as inputs."),
      llvm::cl::init(false)};

  Option<bool> loadInputTensorsFromDisk{
      *this, "load-input-tensors-from-disk",
      llvm::cl::desc("Load input tensors from disk using ttnn.load_tensor "
                     "instead of generating synthetic inputs with ttnn.ones"),
      llvm::cl::init(false)};

  Option<std::string> tensorLoadDirectory{
      *this, "tensor-load-directory",
      llvm::cl::desc("Relative directory path where input tensors are stored"),
      llvm::cl::init("")};

  Option<std::string> tensorLoadFilePrefix{
      *this, "tensor-load-file-prefix",
      llvm::cl::desc("Prefix for input tensor files"), llvm::cl::init("arg")};

  Option<bool> tryRecoverStructure{
      *this, "try-recover-structure",
      llvm::cl::desc(
          "Enable pipelines and passes that try to recover structure of the "
          "original IR/code. Highly experimental; please file issues at "
          "https://github.com/tenstorrent/tt-mlir/issues"),
      llvm::cl::init(false)};
};

// TTIR to TTNN backend pipeline options.
//
// Inherits from TTIRToTTNNDevicePipelineOptions and
// TTIRToLLVMCPUPipelineOptions to reuse the options.
//
struct TTIRToTTNNBackendPipelineOptions
    : public TTIRToTTNNDevicePipelineOptions,
      public ttir::TTIRToLLVMCPUPipelineOptions {};

// TTIR to EmitC end-to-end pipeline options.
//
// Inherits from TTIRToTTNNDevicePipelineOptions and
// TTNNToEmitCDevicePipelineOptions to reuse the options.
//
struct TTIRToEmitCPipelineOptions : public TTIRToTTNNDevicePipelineOptions,
                                    public TTNNToEmitCDevicePipelineOptions {
  TTIRToEmitCPipelineOptions() {
    // TODO(dmilinkovic): Remove once CPU-hoisting is supported on EmitC - issue
    // #6100.
    this->enableCPUHoistedConstEval = false;
  }
};

// TTIR to EmitPy pipeline options.
//
// Inherits from TTIRToTTNNDevicePipelineOptions and
// TTNNToEmitPyDevicePipelineOptions to reuse the options.
//
struct TTIRToEmitPyPipelineOptions : public TTIRToTTNNDevicePipelineOptions,
                                     public TTNNToEmitPyDevicePipelineOptions {
};

// Recover Structure XLA/Torch pipeline options.
struct RecoverStructureXLATorchPipelineOptions
    : public PassPipelineOptions<RecoverStructureXLATorchPipelineOptions> {
  // Add any future options here if needed
};

//===----------------------------------------------------------------------===//
// End-to-end pipelines, which lower TTIR to various TTNN targets.
//===----------------------------------------------------------------------===//

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTIRToEmitCPipeline(OpPassManager &pm,
                               const TTIRToEmitCPipelineOptions &options);

void createTTIRToEmitPyPipeline(OpPassManager &pm,
                                const TTIRToEmitPyPipelineOptions &options);

void createTTNNToEmitPyPipeline(
    OpPassManager &pm, const TTNNToEmitPyDevicePipelineOptions &options);

void createRecoverStructureXLATorchPipeline(
    OpPassManager &pm, const RecoverStructureXLATorchPipelineOptions &options);

void createTTNNPipelineD2MPass(OpPassManager &pm);

void registerTTNNPipelines();
} // namespace mlir::tt::ttnn

#endif
