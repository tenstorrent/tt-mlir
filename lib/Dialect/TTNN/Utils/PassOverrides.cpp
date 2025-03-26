// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>
#include <numeric>

namespace mlir::tt::ttnn {

namespace {
std::optional<SmallVector<int64_t, 2>>
parseGrid(StringRef param, char gridSeparator, llvm::cl::Option &opt) {
  SmallVector<StringRef, 2> gridParts;
  param.split(gridParts, gridSeparator);
  if (gridParts.size() == 2) {
    int64_t gridX, gridY;
    if (gridParts[0].getAsInteger(10, gridX) ||
        gridParts[1].getAsInteger(10, gridY)) {
      opt.error("Invalid grid size: " + param);
      return std::nullopt;
    }
    return SmallVector<int64_t, 2>{gridX, gridY};
  }
  return std::nullopt;
}
} // namespace

// ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true override-output-layout=add_0=1x1,add_1=l1,add_2=block_sharded,add_3=bf16,add_4=l1:interleaved,add_5=width_sharded:tile,add_6=4x4:dram:interleaved:row_major:bf16,add_7=4x4:l1:interleaved:tile:f32"
// override-conv2d-config=op_0=dtype#bf16:weights_dtype#bf16:activation#relu:input_channels_alignment#8:deallocate_activation#true:reallocate_halo_output#true:act_block_h_override#4:act_block_w_div#4:reshard_if_not_optimal#true:override_sharding_config#true:shard_layout#interleaved:core_grid#4x4:transpose_shards#true:output_layout#row_major:enable_act_double_buffer#true:enable_weights_double_buffer#true:enable_split_reader#true
bool Conv2dConfigOverrideParser::parse(
    llvm::cl::Option &opt, StringRef argName, StringRef arg, 
    llvm::StringMap<Conv2dConfigOverrideParams> &value) {
  SmallVector<StringRef> opOverrideList;
  constexpr size_t kvPairSize = 2;
  constexpr size_t iOpName = 0;
  constexpr size_t iConv2dConfigOverrideParams = 1;
  constexpr size_t iConv2dConfigParamName = 0;
  constexpr size_t iConv2dConfigParamValue = 1;
  constexpr char opSeparator = ',';
  constexpr char opNameSeparator = '=';
  constexpr char paramSeparator = ':';
  constexpr char paramKVSeparator = '#';

  arg.split(opOverrideList, opSeparator);
  for(const StringRef override : opOverrideList) {
    SmallVector<StringRef, kvPairSize> opOverrideParts;
    override.split(opOverrideParts, opNameSeparator);
    if(opOverrideParts.size() != kvPairSize) {
      //                                         Why grid??
      opt.error("Invalid format for override grid sizes: " + override);
      return true;  
    }

    SmallVector<StringRef> conv2dParamParts;
    opOverrideParts[iConv2dConfigOverrideParams].split(conv2dParamParts,
                                                 paramSeparator);

    Conv2dConfigOverrideParams params;

    // TODO: change BoolAtrr format from 1/0 to true/false
    for(const StringRef &param : conv2dParamParts) {
      SmallVector<StringRef> param_kv;
      param.split(param_kv, paramKVSeparator);
      StringRef param_name = param_kv[iConv2dConfigParamName];
      StringRef param_value = param_kv[iConv2dConfigParamValue];

      if(param_name == "dtype") {
        auto dtype = mlir::tt::DataTypeStringToEnum(param_value);
        if (!dtype) {
          opt.error("Invalid dtype: " + param_value);
          return true;
        } 
        if(params.dtype.has_value()) {
            opt.error("Duplicate dtype: " + param_value);
            return true;
        }
        params.dtype = dtype;
      } 
      else if(param_name == "weights_dtype") {
        auto weightsDtype = mlir::tt::DataTypeStringToEnum(param_value);
        if (!weightsDtype) {
          opt.error("Invalid weights_dtype: " + param_value);
          return true;
        } 
        if(params.weightsDtype.has_value()) {
            opt.error("Duplicate weights_dtype: " + param_value);
            return true;
        }
        params.weightsDtype = weightsDtype;
      }
      //TODO: Check activation type. When to use StringRef and StringAttr.
      else if(param_name == "activation") {
        auto activation = param_value;
        if(activation.empty()) {
          opt.error("Invalid activation: " + param_value);
          return true;
        }
        if (!activation.equals_insensitive(StringRef("relu")) &&
              !activation.equals_insensitive(("none"))) {
          opt.error("Invalid activation: " + param_value);
          return true;
        } 
        if(params.activation.has_value()) {
            opt.error("Duplicate activation: " + param_value);
            return true;
        }
        params.activation = std::optional<std::string>(activation.str());
      }
      else if(param_name == "input_channels_alignment") {
        uint32_t inputChannelsAlignment; 
        if(param_value.getAsInteger<uint32_t>(10, inputChannelsAlignment)) {
          opt.error("Invalid input_channels_alignment: " + param_value);
          return true;
        }
        if(params.inputChannelsAlignment.has_value()) {
          opt.error("Duplicate input_channels_alignment: " + param_value);
          return true;
        }
        params.inputChannelsAlignment = inputChannelsAlignment;
      }  
      //TODO: 1/0 or true/false for attribute value?
      else if(param_name == "deallocate_activation") {
        bool deallocateActivation;
        // Is it safe (getAsInteger<bool>) ?
        if(param_value.getAsInteger<bool>(10, deallocateActivation)) {
          opt.error("Invalid deallocate_activation: " + param_value);
          return true;
        }
        if(params.deallocateActivation.has_value()) {
          opt.error("Duplicate deallocate_activation: " + param_value);
          return true;
        }
        params.deallocateActivation = deallocateActivation;
      }
      else if(param_name == "reallocate_halo_output") {
        bool reallocateHaloOutput;
        if(param_value.getAsInteger<bool>(10, reallocateHaloOutput)) {
          opt.error("Invalid reallocateHaloOutput: " + param_value);
          return true;
        }
        if(params.reallocateHaloOutput.has_value()) {
          opt.error("Duplicate reallocateHaloOutput: " + param_value);
          return true;
        }
        params.reallocateHaloOutput = reallocateHaloOutput;
      }
      else if(param_name == "act_block_h_override") {
        uint32_t actBlockHOverride;
        if(param_value.getAsInteger<uint32_t>(10, actBlockHOverride)) {
          opt.error("Invalid actBlockHOverride: " + param_value);
          return true;
        }
        if(params.actBlockHOverride.has_value()) {
          opt.error("Duplicate actBlockHOverride: " + param_value);
          return true;
        }
        params.actBlockHOverride = actBlockHOverride;
      }
      else if(param_name == "act_block_w_div") {
        uint32_t actBlockWDiv;
        if(param_value.getAsInteger<uint32_t>(10, actBlockWDiv)) {
          opt.error("Invalid actBlockWDiv: " + param_value);
          return true;
        }
        if(params.actBlockWDiv.has_value()) {
          opt.error("Duplicate actBlockWDiv: " + param_value);
          return true;
        }
        params.actBlockWDiv = actBlockWDiv;
      }
      else if(param_name == "reshard_if_not_optimal") {
        bool reshardIfNotOptimal;
        if(param_value.getAsInteger<bool>(10, reshardIfNotOptimal)) {
          opt.error("Invalid reshardIfNotOptimal: " + param_value);
          return true;
        }
        if(params.reshardIfNotOptimal.has_value()) {
          opt.error("Duplicate reshardIfNotOptimal: " + param_value);
          return true;
        }
        params.reshardIfNotOptimal = reshardIfNotOptimal;
      }
      else if(param_name == "override_sharding_config") {
        bool overrideShardingConfig;
        if(param_value.getAsInteger<bool>(10, overrideShardingConfig)) {
          opt.error("Invalid overrideShardingConfig: " + param_value);
          return true;
        }
        if(params.overrideShardingConfig.has_value()) {
          opt.error("Duplicate overrideShardingConfig: " + param_value);
          return true;
        }
        params.overrideShardingConfig = overrideShardingConfig;
      }
      else if(param_name == "shard_layout") {
        auto shardLayout = symbolizeTensorMemoryLayout(param_value);
        if (!shardLayout) {
          opt.error("Invalid shardLayout: " + param_value);
          return true;
        } 
        if(params.shardLayout.has_value()) {
            opt.error("Duplicate shardLayout: " + param_value);
            return true;
        }
        params.shardLayout = shardLayout;
      }
      //TODO: Check coreGrid
      else if(param_name == "core_grid") 
        continue;
      else if(param_name == "transpose_shards") {
        bool transposeShards;
        if(param_value.getAsInteger<bool>(10, transposeShards)) {
          opt.error("Invalid transposeShards: " + param_value);
          return true;
        }
        if(params.transposeShards.has_value()) {
          opt.error("Duplicate transposeShards: " + param_value);
          return true;
        }
        params.transposeShards = transposeShards;
      }
      else if(param_name == "output_layout") {
        auto outputLayout = symbolizeLayout(param_value);
        if (!outputLayout) {
          opt.error("Invalid outputLayout: " + param_value);
          return true;
        } 
        if(params.outputLayout.has_value()) {
            opt.error("Duplicate outputLayout: " + param_value);
            return true;
        }
        params.outputLayout = outputLayout;
      }
      else if(param_name == "enable_act_double_buffer") {
        bool enableActDoubleBuffer;
        if(param_value.getAsInteger<bool>(10, enableActDoubleBuffer)) {
          opt.error("Invalid enableActDoubleBuffer: " + param_value);
          return true;
        }
        if(params.enableActDoubleBuffer.has_value()) {
          opt.error("Duplicate enableActDoubleBuffer: " + param_value);
          return true;
        }
        params.enableActDoubleBuffer = enableActDoubleBuffer;
      }
      else if(param_name == "enable_weights_double_buffer") {
        bool enableWeightsDoubleBuffer;
        if(param_value.getAsInteger<bool>(10, enableWeightsDoubleBuffer)) {
          opt.error("Invalid enableWeightsDoubleBuffer: " + param_value);
          return true;
        }
        if(params.enableWeightsDoubleBuffer.has_value()) {
          opt.error("Duplicate enableWeightsDoubleBuffer: " + param_value);
          return true;
        }
        params.enableWeightsDoubleBuffer = enableWeightsDoubleBuffer;
      }
      else if(param_name == "enable_split_reader") {
        bool enableSplitReader;
        if(param_value.getAsInteger<bool>(10, enableSplitReader)) {
          opt.error("Invalid enableSplitReader: " + param_value);
          return true;
        }
        if(params.enableSplitReader.has_value()) {
          opt.error("Duplicate enableSplitReader: " + param_value);
          return true;
        }
        params.enableSplitReader = enableSplitReader;
      }
      else if(param_name == "enable_subblock_padding") {
        bool enableSubblockPadding;
        if(param_value.getAsInteger<bool>(10, enableSubblockPadding)) {
          opt.error("Invalid enableSubblockPadding: " + param_value);
          return true;
        }
        if(params.enableSubblockPadding.has_value()) {
          opt.error("Duplicate enableSubblockPadding: " + param_value);
          return true;
        }
        params.enableSubblockPadding = enableSubblockPadding;
      }
      else {
        opt.error("Invalid Conv2dConfig parameter: " + param);
        return true;
      }
    }

    value[opOverrideParts[iOpName]] = params;
  }
  return false;
}

std::string Conv2dConfigOverrideParser::toString(
    const llvm::StringMap<Conv2dConfigOverrideParams> &value) {
  std::string res;
  size_t count = 0;
  for (const auto &entry : value) {
    res += std::string(entry.getKey()) + "=";
    const Conv2dConfigOverrideParams &params = entry.getValue();

    std::vector<std::string> parts;

    if(params.dtype.has_value()) {
      parts.push_back("dtype#" + DataTypeEnumToString(params.dtype.value()).str());
    }
    if(params.weightsDtype.has_value()) {
      parts.push_back("weights_dtype#" + DataTypeEnumToString(params.weightsDtype.value()).str());
    }
    if(params.activation.has_value()) {
      parts.push_back("activation#" + params.activation.value());
    }
    if(params.inputChannelsAlignment.has_value()) {
      parts.push_back("input_channels_alignment#" + std::to_string(params.inputChannelsAlignment.value()));
    }
    if(params.deallocateActivation.has_value()) {
      parts.push_back("deallocate_activation#" + std::to_string(params.deallocateActivation.value()));
    }
    if(params.reallocateHaloOutput.has_value()) {
      parts.push_back("reallocate_halo_output#" + std::to_string(params.reallocateHaloOutput.value()));
    }
    if(params.actBlockHOverride.has_value()) {
      parts.push_back("act_block_h_override#" + std::to_string(params.actBlockHOverride.value()));
    }
    if(params.actBlockWDiv.has_value()) {
      parts.push_back("act_block_w_div#" + std::to_string(params.actBlockWDiv.value()));
    }
    if(params.reshardIfNotOptimal.has_value()) {
      parts.push_back("reshard_if_not_optimal#" + std::to_string(params.reshardIfNotOptimal.value()));
    }
    if(params.overrideShardingConfig.has_value()) {
      parts.push_back("override_sharding_config#" + std::to_string(params.overrideShardingConfig.value()));
    }
    if(params.shardLayout.has_value()) {
      parts.push_back("shard_layout#" + stringifyTensorMemoryLayout(params.shardLayout.value()).str());
    }
    if(params.transposeShards.has_value()) {
      parts.push_back("transpose_shards#" + std::to_string(params.transposeShards.value()));
    }
    if(params.outputLayout.has_value()) {
      parts.push_back("output_layout#" + stringifyLayout(params.outputLayout.value()).str());
    }
    if(params.enableActDoubleBuffer.has_value()) {
      parts.push_back("enable_act_double_buffer#" + std::to_string(params.enableActDoubleBuffer.value()));
    }
    if(params.enableWeightsDoubleBuffer.has_value()) {
      parts.push_back("enable_weights_double_buffer#" + std::to_string(params.enableWeightsDoubleBuffer.value()));
    }
    if(params.enableSplitReader.has_value()) {
      parts.push_back("enable_split_reader#" + std::to_string(params.enableSplitReader.value()));
    }
    if(params.enableSubblockPadding.has_value()) {
      parts.push_back("enable_subblock_padding#" + std::to_string(params.enableSubblockPadding.value()));
    }

    res += std::accumulate(parts.begin(), parts.end(), std::string(),
                           [](const std::string &a, const std::string &b) {
                             return a.empty() ? b : a + ":" + b;
                           });

    if(++count < value.size()) {
      res += ",";
    }
  }
  return res;
}

void Conv2dConfigOverrideParser::print(
    llvm::raw_ostream &os,
    const llvm::StringMap<Conv2dConfigOverrideParams> &value) {
  os << "override-conv2d-config=";
  os << Conv2dConfigOverrideParser::toString(value);
  os << "\n";
}

bool OutputLayoutOverrideParser::parse(
    llvm::cl::Option &opt, StringRef argName, StringRef arg,
    llvm::StringMap<OutputLayoutOverrideParams> &value) {
  SmallVector<StringRef> opOverrideList;
  constexpr size_t kvPairSize = 2;
  constexpr size_t iOpName = 0;
  constexpr size_t iLayoutOverrideParams = 1;
  constexpr char opSeparator = ',';
  constexpr char opNameSeparator = '=';
  constexpr char paramSeparator = ':';
  constexpr char gridSeparator = 'x';

  arg.split(opOverrideList, opSeparator);
  for (const StringRef override : opOverrideList) {
    SmallVector<StringRef, kvPairSize> opOverrideParts;
    override.split(opOverrideParts, opNameSeparator);
    if (opOverrideParts.size() != kvPairSize) {
      opt.error("Invalid format for override grid sizes: " + override);
      return true;
    }

    SmallVector<StringRef> layoutParamParts;
    opOverrideParts[iLayoutOverrideParams].split(layoutParamParts,
                                                 paramSeparator);

    OutputLayoutOverrideParams params;

    for (const StringRef &param : layoutParamParts) {
      if (auto grid = parseGrid(param, gridSeparator, opt)) {
        if (params.grid.has_value()) {
          opt.error("Multiple grid parameters provided: " + param);
          return true;
        }
        params.grid = grid;
      } else if (auto bufferType = symbolizeBufferType(param)) {
        if (params.bufferType.has_value()) {
          opt.error("Multiple buffer type parameters provided: " + param);
          return true;
        }
        params.bufferType = bufferType;
      } else if (auto tensorMemoryLayout = symbolizeTensorMemoryLayout(param)) {
        if (params.tensorMemoryLayout.has_value()) {
          opt.error("Multiple tensor memory layout parameters provided: " +
                    param);
          return true;
        }
        params.tensorMemoryLayout = tensorMemoryLayout;
      } else if (auto memoryLayout = mlir::tt::ttnn::symbolizeLayout(param)) {
        if (params.memoryLayout.has_value()) {
          opt.error("Multiple memory layout parameters provided: " + param);
          return true;
        }
        params.memoryLayout = memoryLayout;
      } else if (auto dataType = mlir::tt::DataTypeStringToEnum(param)) {
        if (params.dataType.has_value()) {
          opt.error("Multiple data type parameters provided: " + param);
          return true;
        }
        params.dataType = dataType;
      } else {
        opt.error("Invalid layout parameter: " + param);
        return true;
      }
    }

    value[opOverrideParts[iOpName]] = params;
  }
  return false;
}

std::string OutputLayoutOverrideParser::toString(
    const llvm::StringMap<OutputLayoutOverrideParams> &value) {
  std::string res;
  size_t count = 0;
  for (const auto &entry : value) {
    res += std::string(entry.getKey()) + "=";
    const OutputLayoutOverrideParams &params = entry.getValue();

    std::vector<std::string> parts;

    // Collect grid values
    if (params.grid.has_value()) {
      std::string gridStr;
      for (size_t i = 0; i < params.grid.value().size(); ++i) {
        gridStr += std::to_string(params.grid.value()[i]);
        if (i < params.grid.value().size() - 1) {
          gridStr += "x";
        }
      }
      parts.push_back(gridStr);
    }
    // Collect memory space and memory layout
    if (params.bufferType.has_value()) {
      parts.push_back(std::string(
          mlir::tt::ttnn::stringifyBufferType(params.bufferType.value())));
    }
    if (params.tensorMemoryLayout.has_value()) {
      parts.push_back(std::string(mlir::tt::ttnn::stringifyTensorMemoryLayout(
          params.tensorMemoryLayout.value())));
    }
    if (params.memoryLayout.has_value()) {
      parts.push_back(std::string(
          mlir::tt::ttnn::stringifyLayout(params.memoryLayout.value())));
    }
    if (params.dataType.has_value()) {
      parts.push_back(
          std::string(mlir::tt::DataTypeEnumToString(params.dataType.value())));
    }

    // Join parts with ":"
    res += std::accumulate(parts.begin(), parts.end(), std::string(),
                           [](const std::string &a, const std::string &b) {
                             return a.empty() ? b : a + ":" + b;
                           });

    if (++count < value.size()) {
      res += ",";
    }
  }
  return res;
}

void OutputLayoutOverrideParser::print(
    llvm::raw_ostream &os,
    const llvm::StringMap<OutputLayoutOverrideParams> &value) {
  os << "override-output-layout=";
  os << OutputLayoutOverrideParser::toString(value);
  os << "\n";
}

bool InputLayoutOverrideParser::parse(
    llvm::cl::Option &opt, StringRef argName, StringRef arg,
    llvm::StringMap<InputLayoutOverrideParams> &value) {
  SmallVector<StringRef> opOverrideList;
  constexpr size_t kvPairSize = 2;
  constexpr size_t iOpName = 0;
  constexpr size_t iOperands = 1;
  constexpr char opSeparator = ',';
  constexpr char opNameSeparator = '=';
  constexpr char opParamSeparator = ':';

  arg.split(opOverrideList, opSeparator);
  for (const StringRef override : opOverrideList) {
    SmallVector<StringRef, kvPairSize> opOverrideParts;
    override.split(opOverrideParts, opNameSeparator);
    if (opOverrideParts.size() != kvPairSize) {
      opt.error("Invalid format for input layouts override: " + override);
      return true;
    }

    SmallVector<int64_t> operandIndexes;
    SmallVector<StringRef> operandIndexParts;

    // Parse operand indexes.
    opOverrideParts[iOperands].split(operandIndexParts, opParamSeparator);
    for (const StringRef operandIndexPart : operandIndexParts) {
      int64_t operandIndexValue;
      if (operandIndexPart.getAsInteger(10 /*Radix*/, operandIndexValue)) {
        opt.error("Invalid operand index: " + operandIndexPart);
        return true;
      }
      operandIndexes.push_back(operandIndexValue);
    }

    // Set parsed op overrides.
    value[opOverrideParts[iOpName]] =
        InputLayoutOverrideParams{std::move(operandIndexes)};
  }
  return false;
}

std::string InputLayoutOverrideParser::toString(
    const llvm::StringMap<InputLayoutOverrideParams> &value) {
  std::string res;
  size_t count = 0;
  for (const auto &entry : value) {
    res += std::string(entry.getKey()) + "=";
    const InputLayoutOverrideParams &params = entry.getValue();
    for (int64_t operandIdx : params.operandIdxes) {
      res += std::to_string(operandIdx) + ":";
    }
    // Remove the last colon.
    res.pop_back();
    if (++count < value.size()) {
      res += ",";
    }
  }
  return res;
}

void InputLayoutOverrideParser::print(
    llvm::raw_ostream &os,
    const llvm::StringMap<InputLayoutOverrideParams> &value) {
  os << "insert-memreconfig=";
  os << InputLayoutOverrideParser::toString(value);
  os << "\n";
}

} // namespace mlir::tt::ttnn
