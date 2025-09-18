// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_PIPELINES_TTNNPIPELINES_H
#define TTMLIR_DIALECT_TTNN_PIPELINES_TTNNPIPELINES_H

#include "ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/MemoryLayoutAnalysisParams.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

#include "mlir/Pass/PassOptions.h"

namespace mlir::tt::ttnn {
// Options for the TTIR to TTNN backend pipeline.
//
struct TTIRToTTNNBackendPipelineOptions
    : public PassPipelineOptions<TTIRToTTNNBackendPipelineOptions> {
  // If this option is true, run Optimizer trying to set optimal Op
  // configuration for max performance. If this option is false, skip running
  // Optimizer pass, thus leaving all ops on default configuration.
  Option<bool> optimizerPassEnabled{
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

  // If this option is true, run memory layout analysis.
  //
  Option<bool> memoryLayoutAnalysisEnabled{
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
      llvm::cl::init(64)};

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
  // The default value is 0.8.
  //
  Option<float> tensorL1UsageCap{
      *this, OptionNames::tensorL1UsageCap,
      llvm::cl::desc("Override tensor L1 usage cap in L1 Interleaved Fallback "
                     "Analysis and Memory Layout Analysis. [0.0-1.0]"),
      llvm::cl::init(0.8f)};

  // Option to enable/disable the workaround pass.
  //
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

  Option<bool> enableFusingConv2dWithMultiplyPattern{
      *this, "enable-fusing-conv2d-with-multiply-pattern",
      llvm::cl::desc("Enable Conv2dWithMultiply pattern in the fusing pass."),
      llvm::cl::init(false)};

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
};

// TTIR to EmitC pipeline options.
// Inherit from TTIRToTTNNBackendPipelineOptions to reuse the options.
//
struct TTIRToEmitCPipelineOptions : public TTIRToTTNNBackendPipelineOptions {};

// TTIR to EmitC SO pipeline options.
// Inherit from TTIRToEmitCPipelineOptions to reuse the options.
//
struct TTIRToEmitCSOPipelineOptions : public TTIRToEmitCPipelineOptions {};

// TTIR to EmitPy pipeline options.
// Inherit from TTIRToTTNNBackendPipelineOptions to reuse the options.
//
struct TTIRToEmitPyPipelineOptions : public TTIRToTTNNBackendPipelineOptions {};

void createTTNNPipelineTTIRPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineAnalysisPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineLoweringPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineLayoutDecompositionPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTNNPipelineDeallocPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options);

void createTTIRToEmitCPipeline(OpPassManager &pm,
                               const TTIRToEmitCPipelineOptions &options);

void createTTIRToEmitCSOPipeline(OpPassManager &pm,
                                 const TTIRToEmitCSOPipelineOptions &options);

void createTTIRToEmitPyPipeline(OpPassManager &pm,
                                const TTIRToEmitPyPipelineOptions &options);

/// Registers all pipelines for the `bufferization` dialect. Currently,
/// this includes only the "ttir-to-ttnn-backend-pipeline".
void registerTTNNPipelines();
} // namespace mlir::tt::ttnn

#endif
