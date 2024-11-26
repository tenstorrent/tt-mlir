// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"

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

void OutputLayoutOverrideParser::print(
    llvm::raw_ostream &os,
    const llvm::StringMap<OutputLayoutOverrideParams> &value) {
  os << "override-output-layout=";
  size_t count = 0;
  for (const auto &entry : value) {
    os << entry.getKey() << "=";
    const OutputLayoutOverrideParams &params = entry.getValue();
    // Print grid values
    for (size_t i = 0; i < params.grid.size(); ++i) {
      os << params.grid[i];
      if (i < params.grid.size() - 1) {
        os << "x";
      }
    }
    // Print memory space and memory layout
    os << ":" << mlir::tt::ttnn::stringifyBufferType(params.bufferType);
    os << ":"
       << mlir::tt::ttnn::stringifyTensorMemoryLayout(
              params.tensorMemoryLayout);
    os << ":" << mlir::tt::ttnn::stringifyLayout(params.memoryLayout);
    os << ":" << mlir::tt::DataTypeEnumToString(params.dataType);
    if (++count < value.size()) {
      os << ",";
    }
  }
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

void InputLayoutOverrideParser::print(
    llvm::raw_ostream &os,
    const llvm::StringMap<InputLayoutOverrideParams> &value) {
  os << "insert-memreconfig=";
  size_t count = 0;
  for (const auto &entry : value) {
    os << entry.getKey() << "=";
    const InputLayoutOverrideParams &params = entry.getValue();
    for (int64_t operandIdx : params.operandIdxes) {
      os << operandIdx
         << (operandIdx < static_cast<int64_t>(params.operandIdxes.size()) - 1
                 ? ':'
                 : char());
    }
    if (++count < value.size()) {
      os << ",";
    }
  }
  os << "\n";
}



  void OptimizerOverridesHandler::setOptimizerPass(bool value) { enableOptimizerPass = value; }
 
  void OptimizerOverridesHandler::setMemoryConfig(bool value) { enableMemoryConfig = value; }
  void OptimizerOverridesHandler::setMemoryLayoutAnalysis(bool value) { enableMemoryLayoutAnalysis = value; }
  void OptimizerOverridesHandler::setEnableMemoryLayoutAnalysisPolicy(bool value) { enableMemoryLayoutAnalysisPolicy = value; }
  void OptimizerOverridesHandler::setMemoryLayoutAnalysisPolicy(MemoryLayoutAnalysisPolicyType value) { memoryLayoutAnalysisPolicy = value; }

  void OptimizerOverridesHandler::setInputLayoutOverrides(llvm::StringMap<InputLayoutOverrideParams> &value) { inputLayoutOverrides = value; }
  void OptimizerOverridesHandler::setOutputLayoutOverrides(llvm::StringMap<OutputLayoutOverrideParams> &value) { outputLayoutOverrides = value; }

  void OptimizerOverridesHandler::setSystemDescPath(std::string value) { systemDescPath = value; }
  void OptimizerOverridesHandler::setMaxLegalLayouts(int64_t value) { maxLegalLayouts = value; }
  void OptimizerOverridesHandler::setMeshShape(std::vector<int64_t> value) { meshShape = value; }

  bool OptimizerOverridesHandler::getOptimizerPass() const { return enableOptimizerPass; }

  bool OptimizerOverridesHandler::getMemoryConfig() const { return enableMemoryConfig; }
  bool OptimizerOverridesHandler::getMemoryLayoutAnalysis() const { return enableMemoryLayoutAnalysis; }
  bool OptimizerOverridesHandler::getEnableMemoryLayoutAnalysisPolicy() const { return enableMemoryLayoutAnalysisPolicy; }
  MemoryLayoutAnalysisPolicyType OptimizerOverridesHandler::getMemoryLayoutAnalysisPolicy() const { return memoryLayoutAnalysisPolicy; }

  std::string OptimizerOverridesHandler::getSystemDescPath() const { return systemDescPath; }
  int64_t OptimizerOverridesHandler::getMaxLegalLayouts() const { return maxLegalLayouts; }
  std::vector<int64_t> OptimizerOverridesHandler::getMeshShape() const { return meshShape; }

  llvm::StringMap<InputLayoutOverrideParams> OptimizerOverridesHandler::getInputLayoutOverrides() const { return inputLayoutOverrides; }
  llvm::StringMap<OutputLayoutOverrideParams> OptimizerOverridesHandler::getOutputLayoutOverrides() const { return outputLayoutOverrides; }

  std::string OptimizerOverridesHandler::toString() const {
    
    std::string options = "";
    
    if (enableOptimizerPass) {
      options += "enable-optimizer=true ";
    }

    if (enableMemoryConfig) {
      options += "memreconfig-enabled=true ";
    }

    if (enableMemoryLayoutAnalysis) {
      options += "memory-layout-analysis-enabled=true ";
    }

    if (enableMemoryLayoutAnalysisPolicy) {
      options += "memory-layout-analysis-policy=";
      switch (memoryLayoutAnalysisPolicy)
      {
        case MemoryLayoutAnalysisPolicyType::DFSharding:
          options += "DFSharding ";
          break;
        case MemoryLayoutAnalysisPolicyType::L1Interleaved:
          options += "L1Interleaved ";
          break;
      }
    }

    // Create input layout overrides.
    //  Example: insert-memreconfig=input0=0:1,input1=0,input2=0:1:2
    if (inputLayoutOverrides.size() > 0) {
      options += "insert-memreconfig=";
      // Read operation name and operand indexes.
      for (const auto &entry : inputLayoutOverrides) {
        // Concatenate operation name.
        options += std::string(entry.getKey()) + "=";
        // Concatenate operand indexes.
        const InputLayoutOverrideParams &params = entry.getValue();
        // Concatenate operand indexes.
        for (int64_t operandIdx : params.operandIdxes) {
          options += std::to_string(operandIdx) + ":";
        }
        // Remove the last colon.
        options.pop_back();
        options += ",";
      }
      // Remove the last comma.
      options.pop_back();
      // Add a space for the next option.
      options += " ";
    }

    // Create output layout overrides.
    //  Example: override-output-layout=op1=2x2:dram:interleaved:tile:fp32,op2=4x4:l1:block_sharded:row_major:fp16
    //  Example: override-output-layout=add_1_2=1x1:dram:interleaved:row_major:f32"
    if (outputLayoutOverrides.size() > 0) {
      options += "override-output-layout=";
      // Read operation name and output layout parameters.
      for (const auto &entry : outputLayoutOverrides) {
        // Concatenate operation name.
        options += std::string(entry.getKey()) + "=";
        const OutputLayoutOverrideParams &params = entry.getValue();
        // Concatenate grid size.
        for (int64_t gridValue : params.grid) {
          options += std::to_string(gridValue) + "x";
        }
        // Remove the last x.
        options.pop_back();
        // Concatenate buffer type
        // enum class BufferType : uint32_t {
        //   DRAM = 0,
        //   L1 = 1,
        //   SystemMemory = 2,
        //   L1Small = 3,
        //   Trace = 4,
        // };
        options += ":";
        switch (params.bufferType)
        {
          case mlir::tt::ttnn::BufferType::DRAM:
            options += "dram";
            break;
          case mlir::tt::ttnn::BufferType::L1:
            options += "l1";
            break;
          case mlir::tt::ttnn::BufferType::SystemMemory:
            options += "system_memory";
            break;
          case mlir::tt::ttnn::BufferType::L1Small:
            options += "l1_small";
            break;
          case mlir::tt::ttnn::BufferType::Trace:
            options += "trace";
            break;
        }

        // Concatenate tensor memory layout.
        // enum class TensorMemoryLayout : uint32_t {
        //   None = 0,
        //   Interleaved = 1,
        //   SingleBank = 2,
        //   HeightSharded = 3,
        //   WidthSharded = 4,
        //   BlockSharded = 5,
        // };
        options += ":";
        switch (params.tensorMemoryLayout)
        {
          case mlir::tt::ttnn::TensorMemoryLayout::None:
            options += "none";
            break;
          case mlir::tt::ttnn::TensorMemoryLayout::Interleaved:
            options += "interleaved";
            break;
          case mlir::tt::ttnn::TensorMemoryLayout::SingleBank:
            options += "single_bank";
            break;
          case mlir::tt::ttnn::TensorMemoryLayout::HeightSharded:
            options += "height_sharded";
            break;
          case mlir::tt::ttnn::TensorMemoryLayout::WidthSharded:
            options += "width_sharded";
            break;
          case mlir::tt::ttnn::TensorMemoryLayout::BlockSharded:
            options += "block_sharded";
            break;
        }

        // Concatenate memory layout.
        // TTNN Layout
        // enum class Layout : uint32_t {
        //   RowMajor = 0,
        //   Tile = 1,
        //   Invalid = 2,
        // };
        options += ":";
        switch (params.memoryLayout)
        {
          case mlir::tt::ttnn::Layout::RowMajor:
            options += "row_major";
            break;
          case mlir::tt::ttnn::Layout::Tile:
            options += "tile";
            break;
          case mlir::tt::ttnn::Layout::Invalid:
            options += "invalid";
            break;
        }

        // Concatenate data type.
        // enum class DataType : uint32_t {
        //   Float32 = 0,
        //   Float16 = 1,
        //   BFloat16 = 2,
        //   BFP_Float8 = 3,
        //   BFP_BFloat8 = 4,
        //   BFP_Float4 = 5,
        //   BFP_BFloat4 = 6,
        //   BFP_Float2 = 7,
        //   BFP_BFloat2 = 8,
        //   UInt32 = 9,
        //   UInt16 = 10,
        //   UInt8 = 11,
        // };
        options += ":";
        switch (params.dataType)
        {
          case mlir::tt::DataType::Float32:
            options += "f32";
            break;
          case mlir::tt::DataType::Float16:
            options += "f16";
            break;
          case mlir::tt::DataType::BFloat16:
            options += "bf16";
            break;
          case mlir::tt::DataType::BFP_Float8:
            options += "bfp_f8";
            break;
          case mlir::tt::DataType::BFP_BFloat8:
            options += "bfp_bf8";
            break;
          case mlir::tt::DataType::BFP_Float4:
            options += "bfp_f4";
            break;
          case mlir::tt::DataType::BFP_BFloat4:
            options += "bfp_bf4";
            break;
          case mlir::tt::DataType::BFP_Float2:
            options += "bfp_f2";
            break;
          case mlir::tt::DataType::BFP_BFloat2:
            options += "bfp_bf2";
            break;
          case mlir::tt::DataType::UInt32:
            options += "u32";
            break;
          case mlir::tt::DataType::UInt16:
            options += "u16";
            break;
          case mlir::tt::DataType::UInt8:
            options += "u8";
            break;
        }

        options += ",";
      }
      // Remove the last comma.
      options.pop_back();
      // Add a space for the next option.
      options += " ";
    }

    if (systemDescPath.size() > 0) {
      options += "system-desc-path=" + systemDescPath + " ";
    }

    if (maxLegalLayouts > 0) {
      options += "max-legal-layouts=" + std::to_string(maxLegalLayouts) + " ";
    }

    if (meshShape.size() > 0) {
      options += "mesh-shape=";
      for (int64_t meshShapeValue : meshShape) {
        options += std::to_string(meshShapeValue) + ",";
      }
      // Remove the last comma.
      options.pop_back();
    }

    if (options[options.size() - 1] == ' ') {
      options.pop_back();
    }

    return options;

  }

  void OptimizerOverridesHandler::addInputLayoutOverride(StringRef opName, InputLayoutOverrideParams params) { inputLayoutOverrides[opName] = params; }
  void OptimizerOverridesHandler::addInputLayoutOverride(StringRef opName, SmallVector<int64_t> operandIdxes) { inputLayoutOverrides[opName] = InputLayoutOverrideParams{std::move(operandIdxes)}; }
  void OptimizerOverridesHandler::addOutputLayoutOverride(StringRef opName, OutputLayoutOverrideParams params) { outputLayoutOverrides[opName] = params; }
  void OptimizerOverridesHandler::addOutputLayoutOverride(StringRef opName, SmallVector<int64_t> grid, BufferType bufferType, 
                                                          TensorMemoryLayout tensorMemoryLayout, tt::ttnn::Layout memoryLayout, tt::DataType dataType) { 
                                                            outputLayoutOverrides[opName] = OutputLayoutOverrideParams{std::move(grid), bufferType, tensorMemoryLayout, memoryLayout, dataType}; }


} // namespace mlir::tt::ttnn
