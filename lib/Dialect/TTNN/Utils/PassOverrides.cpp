// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "llvm/ADT/SmallVector.h"

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

// Parse bool ("true" or "false") from string. If the string is not "true" or
// "false", returns true to signify the error.
//
bool parseBool(StringRef param, bool &result) {
  if (param.equals_insensitive("true")) {
    result = true;
    return false;
  }
  if (param.equals_insensitive("false")) {
    result = false;
    return false;
  }
  return true;
}
} // namespace

// Full Example:
// conv2d_1=dtype#bf16:weights_dtype#bf16:activation#relu:input_channels_alignment#32:deallocate_activation#false:reallocate_halo_output#true:act_block_h_override#0:act_block_w_div#1:reshard_if_not_optimal#false:override_sharding_config#false:shard_layout#block_sharded:core_grid#0:transpose_shards#true:output_layout#row_major:enable_act_double_buffer#false:enable_weights_double_buffer#false:enable_split_reader#false:enable_subblock_padding#false
// Partial Example:
// conv2d_1=enable_weights_double_buffer#true:activation#none,conv2d_2=dtype#bf16
bool Conv2dConfigOverrideParser::parse(
    llvm::cl::Option &opt, StringRef argName, StringRef arg,
    llvm::StringMap<Conv2dConfigOverrideParams> &value) {
  SmallVector<StringRef> opOverrideList;
  constexpr size_t kvPairSize = 2;
  constexpr size_t iOpName = 0;
  constexpr size_t iConv2dConfigOverrideParams = 1;
  constexpr char opSeparator = ',';
  constexpr char opNameSeparator = '=';
  constexpr char paramSeparator = ':';
  constexpr char paramNameValueSeparator = '#';

  arg.split(opOverrideList, opSeparator);
  for (StringRef opOverrides : opOverrideList) {
    SmallVector<StringRef, kvPairSize> opOverrideParts;
    opOverrides.split(opOverrideParts, opNameSeparator);
    if (opOverrideParts.size() != kvPairSize) {
      opt.error("Invalid override format " + opOverrides);
      return true;
    }

    SmallVector<StringRef> conv2dConfigParams;
    opOverrideParts[iConv2dConfigOverrideParams].split(conv2dConfigParams,
                                                       paramSeparator);

    Conv2dConfigOverrideParams params;

    for (StringRef param : conv2dConfigParams) {
      auto [paramName, paramValue] = param.split(paramNameValueSeparator);

      if (paramName == "dtype") {
        auto dtype = mlir::tt::DataTypeStringToEnum(paramValue);
        if (!dtype) {
          opt.error("Invalid dtype: " + paramValue);
          return true;
        }
        if (params.dtype.has_value()) {
          opt.error("Duplicate dtype: " + paramValue);
          return true;
        }
        params.dtype = dtype;
      } else if (paramName == "weights_dtype") {
        auto weightsDtype = mlir::tt::DataTypeStringToEnum(paramValue);
        if (!weightsDtype) {
          opt.error("Invalid weights_dtype: " + paramValue);
          return true;
        }
        if (params.weightsDtype.has_value()) {
          opt.error("Duplicate weights_dtype: " + paramValue);
          return true;
        }
        params.weightsDtype = weightsDtype;
      } else if (paramName == "activation") {
        auto activation = paramValue;
        if (!activation.equals_insensitive("relu") &&
            !activation.equals_insensitive("none")) {
          opt.error("Invalid activation: " + paramValue);
          return true;
        }
        if (params.activation.has_value()) {
          opt.error("Duplicate activation: " + paramValue);
          return true;
        }
        params.activation = activation.str();
      } else if (paramName == "input_channels_alignment") {
        uint32_t inputChannelsAlignment;
        if (paramValue.getAsInteger<uint32_t>(10, inputChannelsAlignment)) {
          opt.error("Invalid input_channels_alignment: " + paramValue);
          return true;
        }
        if (params.inputChannelsAlignment.has_value()) {
          opt.error("Duplicate input_channels_alignment: " + paramValue);
          return true;
        }
        params.inputChannelsAlignment = inputChannelsAlignment;
      } else if (paramName == "deallocate_activation") {
        bool deallocateActivation;
        if (parseBool(paramValue, deallocateActivation)) {
          opt.error("Invalid deallocate_activation: " + paramValue);
          return true;
        }
        if (params.deallocateActivation.has_value()) {
          opt.error("Duplicate deallocate_activation: " + paramValue);
          return true;
        }
        params.deallocateActivation = deallocateActivation;
      } else if (paramName == "reallocate_halo_output") {
        bool reallocateHaloOutput;
        if (parseBool(paramValue, reallocateHaloOutput)) {
          opt.error("Invalid reallocate_halo_output: " + paramValue);
          return true;
        }
        if (params.reallocateHaloOutput.has_value()) {
          opt.error("Duplicate reallocate_halo_output: " + paramValue);
          return true;
        }
        params.reallocateHaloOutput = reallocateHaloOutput;
      } else if (paramName == "act_block_h_override") {
        uint32_t actBlockHOverride;
        if (paramValue.getAsInteger<uint32_t>(10, actBlockHOverride)) {
          opt.error("Invalid actBlockHOverride: " + paramValue);
          return true;
        }
        if (params.actBlockHOverride.has_value()) {
          opt.error("Duplicate actBlockHOverride: " + paramValue);
          return true;
        }
        params.actBlockHOverride = actBlockHOverride;
      } else if (paramName == "act_block_w_div") {
        uint32_t actBlockWDiv;
        if (paramValue.getAsInteger<uint32_t>(10, actBlockWDiv)) {
          opt.error("Invalid actBlockWDiv: " + paramValue);
          return true;
        }
        if (params.actBlockWDiv.has_value()) {
          opt.error("Duplicate actBlockWDiv: " + paramValue);
          return true;
        }
        params.actBlockWDiv = actBlockWDiv;
      } else if (paramName == "reshard_if_not_optimal") {
        bool reshardIfNotOptimal;
        if (parseBool(paramValue, reshardIfNotOptimal)) {
          opt.error("Invalid reshard_if_not_optimal: " + paramValue);
          return true;
        }
        if (params.reshardIfNotOptimal.has_value()) {
          opt.error("Duplicate reshard_if_not_optimal: " + paramValue);
          return true;
        }
        params.reshardIfNotOptimal = reshardIfNotOptimal;
      } else if (paramName == "override_sharding_config") {
        bool overrideShardingConfig;
        if (parseBool(paramValue, overrideShardingConfig)) {
          opt.error("Invalid override_sharding_config: " + paramValue);
          return true;
        }
        if (params.overrideShardingConfig.has_value()) {
          opt.error("Duplicate override_sharding_config: " + paramValue);
          return true;
        }
        params.overrideShardingConfig = overrideShardingConfig;
      } else if (paramName == "shard_layout") {
        auto shardLayout = symbolizeTensorMemoryLayout(paramValue);
        if (!shardLayout) {
          opt.error("Invalid shardLayout: " + paramValue);
          return true;
        }
        if (params.shardLayout.has_value()) {
          opt.error("Duplicate shardLayout: " + paramValue);
          return true;
        }
        params.shardLayout = shardLayout;
      }
      // TODO(vkovacevic): Parse core_grid
      else if (paramName == "core_grid") {
        assert(false && "overriding core_grid is not supported yet");
        continue;
      } else if (paramName == "transpose_shards") {
        bool transposeShards;
        if (parseBool(paramValue, transposeShards)) {
          opt.error("Invalid transpose_shards: " + paramValue);
          return true;
        }
        if (params.transposeShards.has_value()) {
          opt.error("Duplicate transpose_shards: " + paramValue);
          return true;
        }
        params.transposeShards = transposeShards;
      } else if (paramName == "output_layout") {
        auto outputLayout = symbolizeLayout(paramValue);
        if (!outputLayout) {
          opt.error("Invalid outputLayout: " + paramValue);
          return true;
        }
        if (params.outputLayout.has_value()) {
          opt.error("Duplicate outputLayout: " + paramValue);
          return true;
        }
        params.outputLayout = outputLayout;
      } else if (paramName == "enable_act_double_buffer") {
        bool enableActDoubleBuffer;
        if (parseBool(paramValue, enableActDoubleBuffer)) {
          opt.error("Invalid enable_act_double_buffer: " + paramValue);
          return true;
        }
        if (params.enableActDoubleBuffer.has_value()) {
          opt.error("Duplicate enable_act_double_buffer: " + paramValue);
          return true;
        }
        params.enableActDoubleBuffer = enableActDoubleBuffer;
      } else if (paramName == "enable_weights_double_buffer") {
        bool enableWeightsDoubleBuffer;
        if (parseBool(paramValue, enableWeightsDoubleBuffer)) {
          opt.error("Invalid enable_weights_double_buffer: " + paramValue);
          return true;
        }
        if (params.enableWeightsDoubleBuffer.has_value()) {
          opt.error("Duplicate enable_weights_double_buffer: " + paramValue);
          return true;
        }
        params.enableWeightsDoubleBuffer = enableWeightsDoubleBuffer;
      } else if (paramName == "enable_split_reader") {
        bool enableSplitReader;
        if (parseBool(paramValue, enableSplitReader)) {
          opt.error("Invalid enable_split_reader: " + paramValue);
          return true;
        }
        if (params.enableSplitReader.has_value()) {
          opt.error("Duplicate enable_split_reader: " + paramValue);
          return true;
        }
        params.enableSplitReader = enableSplitReader;
      } else if (paramName == "enable_subblock_padding") {
        bool enableSubblockPadding;
        if (parseBool(paramValue, enableSubblockPadding)) {
          opt.error("Invalid enable_subblock_padding: " + paramValue);
          return true;
        }
        if (params.enableSubblockPadding.has_value()) {
          opt.error("Duplicate enable_subblock_padding: " + paramValue);
          return true;
        }
        params.enableSubblockPadding = enableSubblockPadding;
      } else {
        opt.error("Invalid override parameter: " + paramName);
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

    if (params.dtype.has_value()) {
      parts.push_back("dtype#" +
                      DataTypeEnumToString(params.dtype.value()).str());
    }
    if (params.weightsDtype.has_value()) {
      parts.push_back("weights_dtype#" +
                      DataTypeEnumToString(params.weightsDtype.value()).str());
    }
    if (params.activation.has_value()) {
      parts.push_back("activation#" + params.activation.value());
    }
    if (params.inputChannelsAlignment.has_value()) {
      parts.push_back("input_channels_alignment#" +
                      std::to_string(params.inputChannelsAlignment.value()));
    }
    if (params.deallocateActivation.has_value()) {
      parts.push_back("deallocate_activation#" +
                      std::to_string(params.deallocateActivation.value()));
    }
    if (params.reallocateHaloOutput.has_value()) {
      parts.push_back("reallocate_halo_output#" +
                      std::to_string(params.reallocateHaloOutput.value()));
    }
    if (params.actBlockHOverride.has_value()) {
      parts.push_back("act_block_h_override#" +
                      std::to_string(params.actBlockHOverride.value()));
    }
    if (params.actBlockWDiv.has_value()) {
      parts.push_back("act_block_w_div#" +
                      std::to_string(params.actBlockWDiv.value()));
    }
    if (params.reshardIfNotOptimal.has_value()) {
      parts.push_back("reshard_if_not_optimal#" +
                      std::to_string(params.reshardIfNotOptimal.value()));
    }
    if (params.overrideShardingConfig.has_value()) {
      parts.push_back("override_sharding_config#" +
                      std::to_string(params.overrideShardingConfig.value()));
    }
    if (params.shardLayout.has_value()) {
      parts.push_back(
          "shard_layout#" +
          stringifyTensorMemoryLayout(params.shardLayout.value()).str());
    }
    if (params.transposeShards.has_value()) {
      parts.push_back("transpose_shards#" +
                      std::to_string(params.transposeShards.value()));
    }
    if (params.outputLayout.has_value()) {
      parts.push_back("output_layout#" +
                      stringifyLayout(params.outputLayout.value()).str());
    }
    if (params.enableActDoubleBuffer.has_value()) {
      parts.push_back("enable_act_double_buffer#" +
                      std::to_string(params.enableActDoubleBuffer.value()));
    }
    if (params.enableWeightsDoubleBuffer.has_value()) {
      parts.push_back("enable_weights_double_buffer#" +
                      std::to_string(params.enableWeightsDoubleBuffer.value()));
    }
    if (params.enableSplitReader.has_value()) {
      parts.push_back("enable_split_reader#" +
                      std::to_string(params.enableSplitReader.value()));
    }
    if (params.enableSubblockPadding.has_value()) {
      parts.push_back("enable_subblock_padding#" +
                      std::to_string(params.enableSubblockPadding.value()));
    }

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

void Conv2dConfigOverrideParser::print(
    llvm::raw_ostream &os,
    const llvm::StringMap<Conv2dConfigOverrideParams> &value) {
  os << OptionNames::overrideConv2dConfig << "=";
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
  for (StringRef opOverrides : opOverrideList) {
    SmallVector<StringRef, kvPairSize> opOverrideParts;
    opOverrides.split(opOverrideParts, opNameSeparator);
    if (opOverrideParts.size() != kvPairSize) {
      opt.error("Invalid format for override: " + opOverrides);
      return true;
    }

    SmallVector<StringRef> layoutParamParts;
    opOverrideParts[iLayoutOverrideParams].split(layoutParamParts,
                                                 paramSeparator);

    OutputLayoutOverrideParams params;

    for (StringRef param : layoutParamParts) {
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
  os << OptionNames::overrideOutputLayout << "=";
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
  for (StringRef opOverrides : opOverrideList) {
    SmallVector<StringRef, kvPairSize> opOverrideParts;
    opOverrides.split(opOverrideParts, opNameSeparator);
    if (opOverrideParts.size() != kvPairSize) {
      opt.error("Invalid format for input layouts override: " + opOverrides);
      return true;
    }

    SmallVector<int64_t> operandIndexes;
    SmallVector<StringRef> operandIndexParts;

    // Parse operand indexes.
    opOverrideParts[iOperands].split(operandIndexParts, opParamSeparator);
    for (StringRef operandIndexPart : operandIndexParts) {
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
  os << OptionNames::overrideInputLayout << "=";
  os << InputLayoutOverrideParser::toString(value);
  os << "\n";
}

} // namespace mlir::tt::ttnn
