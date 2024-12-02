// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

namespace mlir::tt::ttnn {

bool OutputLayoutOverrideParser::parse(
    llvm::cl::Option &opt, StringRef argName, StringRef arg,
    llvm::StringMap<OutputLayoutOverrideParams> &value) {
  SmallVector<StringRef> opOverrideList;
  constexpr size_t kMaxGridSize = 2;
  constexpr size_t kvPairSize = 2;
  constexpr size_t kMaxLayoutOverrideParams = 5;
  constexpr size_t iOpName = 0;
  constexpr size_t iLayoutOverrideParams = 1;
  constexpr size_t iGrid = 0;
  constexpr size_t iMemorySpace = 1;
  constexpr size_t iTensorMemoryLayout = 2;
  constexpr size_t iMemoryLayout = 3;
  constexpr size_t iDataType = 4;
  constexpr char opSeparator = ',';
  constexpr char opNameSeparator = '=';
  constexpr char paramSepataor = ':';
  constexpr char gridSeparator = 'x';

  arg.split(opOverrideList, opSeparator);
  for (const StringRef override : opOverrideList) {
    SmallVector<StringRef, kvPairSize> opOverrideParts;
    override.split(opOverrideParts, opNameSeparator);
    if (opOverrideParts.size() != kvPairSize) {
      opt.error("Invalid format for override grid sizes: " + override);
      return true;
    }

    SmallVector<StringRef, kMaxLayoutOverrideParams> layoutParamParts;
    // Split into layout parameters.
    opOverrideParts[iLayoutOverrideParams].split(layoutParamParts,
                                                 paramSepataor);
    if (layoutParamParts.size() != kMaxLayoutOverrideParams) {
      opt.error("Invalid number of layout parameters: " +
                std::to_string(layoutParamParts.size()));
      return true;
    }

    // Parse grid.
    SmallVector<int64_t, kMaxGridSize> grid;
    SmallVector<StringRef, kMaxGridSize> gridParts;
    layoutParamParts[iGrid].split(gridParts, gridSeparator);
    for (const StringRef gridPart : gridParts) {
      int64_t gridValue;
      if (gridPart.getAsInteger(10 /*Radix*/, gridValue)) {
        opt.error("Invalid grid size: " + gridPart);
        return true;
      }
      grid.push_back(gridValue);
    }

    // Parse memory space.
    std::optional<BufferType> bufferType =
        symbolizeBufferType(layoutParamParts[iMemorySpace]);
    if (!bufferType.has_value()) {
      opt.error("Invalid memory space: " + layoutParamParts[iMemorySpace]);
      return true;
    }

    // Parse tensor memory layout.
    std::optional<TensorMemoryLayout> tensorMemoryLayout =
        symbolizeTensorMemoryLayout(layoutParamParts[iTensorMemoryLayout]);
    if (!tensorMemoryLayout.has_value()) {
      opt.error("Invalid tensor memory layout: " +
                layoutParamParts[iTensorMemoryLayout]);
      return true;
    }

    // Parse memory layout.
    std::optional<tt::ttnn::Layout> memoryLayout =
        mlir::tt::ttnn::symbolizeLayout(layoutParamParts[iMemoryLayout]);
    if (!memoryLayout.has_value()) {
      opt.error("Invalid memory layout: " + layoutParamParts[iMemoryLayout]);
      return true;
    }

    // Parse data type.
    std::optional<tt::DataType> dataType =
        mlir::tt::DataTypeStringToEnum(layoutParamParts[iDataType]);
    if (!dataType.has_value()) {
      opt.error("Invalid data type: " + layoutParamParts[iDataType]);
      return true;
    }

    // Set parsed op overrides.
    value[opOverrideParts[iOpName]] = OutputLayoutOverrideParams{
        std::move(grid), bufferType.value(), tensorMemoryLayout.value(),
        memoryLayout.value(), dataType.value()};
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
    // Print grid values
    for (size_t i = 0; i < params.grid.size(); ++i) {
      res += std::to_string(params.grid[i]);
      if (i < params.grid.size() - 1) {
        res += "x";
      }
    }
    // Print memory space and memory layout
    res += ":" +
           std::string(mlir::tt::ttnn::stringifyBufferType(params.bufferType));
    res += ":" + std::string(mlir::tt::ttnn::stringifyTensorMemoryLayout(
                     params.tensorMemoryLayout));
    res +=
        ":" + std::string(mlir::tt::ttnn::stringifyLayout(params.memoryLayout));
    res += ":" + std::string(mlir::tt::DataTypeEnumToString(params.dataType));
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
