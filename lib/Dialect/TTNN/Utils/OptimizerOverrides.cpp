// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"

namespace mlir::tt::ttnn {

void OptimizerOverridesHandler::setEnableOptimizer(bool value) {
  enableOptimizer = value;
}

void OptimizerOverridesHandler::setMemoryReconfig(bool value) {
  enableMemoryReconfig = value;
}
void OptimizerOverridesHandler::setEnableMemoryLayoutAnalysis(bool value) {
  enableMemoryLayoutAnalysis = value;
}
void OptimizerOverridesHandler::setEnableMemoryLayoutAnalysisPolicy(
    bool value) {
  enableMemoryLayoutAnalysisPolicy = value;
}
void OptimizerOverridesHandler::setMemoryLayoutAnalysisPolicy(
    MemoryLayoutAnalysisPolicyType value) {
  enableMemoryLayoutAnalysisPolicy = true;
  memoryLayoutAnalysisPolicy = value;
}

void OptimizerOverridesHandler::setInsertMemReconfig(
    llvm::StringMap<InsertMemReconfigParams> &value) {
  insertMemReconfig = value;
}
void OptimizerOverridesHandler::setOutputLayoutOverrides(
    llvm::StringMap<OutputLayoutOverrideParams> &value) {
  outputLayoutOverrides = value;
}

void OptimizerOverridesHandler::setSystemDescPath(std::string value) {
  systemDescPath = value;
}
void OptimizerOverridesHandler::setMaxLegalLayouts(int64_t value) {
  maxLegalLayouts = value;
}
void OptimizerOverridesHandler::setMeshShape(std::vector<int64_t> value) {
  meshShape = value;
}

void OptimizerOverridesHandler::setConv2dConfigOverrides(
    llvm::StringMap<Conv2dConfigOverrideParams> &value) {
  conv2dConfigOverrides = value;
}

bool OptimizerOverridesHandler::getEnableOptimizer() const {
  return enableOptimizer;
}

bool OptimizerOverridesHandler::getMemoryReconfig() const {
  return enableMemoryReconfig;
}
bool OptimizerOverridesHandler::getEnableMemoryLayoutAnalysis() const {
  return enableMemoryLayoutAnalysis;
}
bool OptimizerOverridesHandler::getEnableMemoryLayoutAnalysisPolicy() const {
  return enableMemoryLayoutAnalysisPolicy;
}
MemoryLayoutAnalysisPolicyType
OptimizerOverridesHandler::getMemoryLayoutAnalysisPolicy() const {
  return memoryLayoutAnalysisPolicy;
}

std::string OptimizerOverridesHandler::getSystemDescPath() const {
  return systemDescPath;
}
int64_t OptimizerOverridesHandler::getMaxLegalLayouts() const {
  return maxLegalLayouts;
}
std::vector<int64_t> OptimizerOverridesHandler::getMeshShape() const {
  return meshShape;
}

llvm::StringMap<InsertMemReconfigParams>
OptimizerOverridesHandler::getInsertMemReconfig() const {
  return insertMemReconfig;
}
llvm::StringMap<OutputLayoutOverrideParams>
OptimizerOverridesHandler::getOutputLayoutOverrides() const {
  return outputLayoutOverrides;
}

llvm::StringMap<Conv2dConfigOverrideParams>
OptimizerOverridesHandler::getConv2dConfigOverrides() const {
  return conv2dConfigOverrides;
}

std::unordered_map<std::string, InsertMemReconfigParams>
OptimizerOverridesHandler::getInsertMemReconfigNanobindWrapper() const {
  std::unordered_map<std::string, InsertMemReconfigParams>
      insertMemReconfigWrapper;
  for (auto &entry : insertMemReconfig) {
    insertMemReconfigWrapper[entry.getKey().str()] = entry.getValue();
  }
  return insertMemReconfigWrapper;
}

std::unordered_map<std::string, OutputLayoutOverrideParams>
OptimizerOverridesHandler::getOutputLayoutOverridesNanobindWrapper() const {
  std::unordered_map<std::string, OutputLayoutOverrideParams>
      outputLayoutOverridesWrapper;
  for (auto &entry : outputLayoutOverrides) {
    outputLayoutOverridesWrapper[entry.getKey().str()] = entry.getValue();
  }
  return outputLayoutOverridesWrapper;
}

std::unordered_map<std::string, Conv2dConfigOverrideParams>
OptimizerOverridesHandler::getConv2dConfigOverridesNanobindWrapper() const {
  std::unordered_map<std::string, Conv2dConfigOverrideParams>
      conv2dConfigOverridesWrapper;
  for (auto &entry : conv2dConfigOverrides) {
    conv2dConfigOverridesWrapper[entry.getKey().str()] = entry.getValue();
  }
  return conv2dConfigOverridesWrapper;
}

std::string OptimizerOverridesHandler::toString() const {

  std::string options = "";

  if (enableOptimizer) {
    options += OptionNames::optimizerPassEnabled.str() + "=true ";
  }

  if (enableMemoryReconfig) {
    options += OptionNames::memReconfigEnabled.str() + "=true ";
  }

  if (enableMemoryLayoutAnalysis) {
    options += OptionNames::memoryLayoutAnalysisEnabled.str() + "=true ";
  }

  if (enableMemoryLayoutAnalysisPolicy) {
    options += OptionNames::memoryLayoutAnalysisPolicy.str() + "=" +
               MemoryLayoutAnalysisPolicyTypeParser::toString(
                   memoryLayoutAnalysisPolicy) +
               " ";
  }

  // Create input layout overrides.
  //  Example:
  //    insert-memreconfig=input0=0:1,input1=0,input2=0:1:2
  if (insertMemReconfig.size() > 0) {
    options += OptionNames::insertMemReconfig.str() + "=" +
               InsertMemReconfigParser::toString(insertMemReconfig) + " ";
  }

  // Create output layout overrides.
  //  Example:
  //    override-output-layout=op1=2x2:dram:interleaved:tile:fp32,op2=4x4:l1:block_sharded:row_major:fp16
  //  Example:
  //    override-output-layout=add_1_2=1x1:dram:interleaved:row_major:f32"
  if (outputLayoutOverrides.size() > 0) {
    options += OptionNames::overrideOutputLayout.str() + "=" +
               OutputLayoutOverrideParser::toString(outputLayoutOverrides) +
               " ";
  }

  if (conv2dConfigOverrides.size() > 0) {
    options += OptionNames::overrideConv2dConfig.str() + "=" +
               Conv2dConfigOverrideParser::toString(conv2dConfigOverrides) +
               " ";
  }

  if (systemDescPath.size() > 0) {
    options += OptionNames::systemDescPath.str() + "=" + systemDescPath + " ";
  }

  if (maxLegalLayouts > 0) {
    options += OptionNames::maxLegalLayouts.str() + "=" +
               std::to_string(maxLegalLayouts) + " ";
  }

  if (meshShape.size() > 0) {
    options += OptionNames::meshShape.str() + "=";
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

void OptimizerOverridesHandler::addInsertMemReconfig(
    StringRef opName, InsertMemReconfigParams params) {
  insertMemReconfig[opName] = params;
}
void OptimizerOverridesHandler::addInsertMemReconfig(
    StringRef opName, SmallVector<int64_t> &operandIdxes) {
  insertMemReconfig[opName] = InsertMemReconfigParams{std::move(operandIdxes)};
}
void OptimizerOverridesHandler::addOutputLayoutOverride(
    StringRef opName, OutputLayoutOverrideParams params) {
  outputLayoutOverrides[opName] = params;
}
void OptimizerOverridesHandler::addOutputLayoutOverride(
    StringRef opName, SmallVector<int64_t> &grid, BufferType bufferType,
    TensorMemoryLayout tensorMemoryLayout, tt::ttnn::Layout memoryLayout,
    tt::DataType dataType) {
  outputLayoutOverrides[opName] = OutputLayoutOverrideParams{
      std::move(grid), bufferType, tensorMemoryLayout, memoryLayout, dataType};
}

void OptimizerOverridesHandler::addConv2dConfigOverride(
    StringRef opName, Conv2dConfigOverrideParams params) {
  conv2dConfigOverrides[opName] = params;
}

void OptimizerOverridesHandler::addInsertMemReconfigNanobindWrapper(
    std::string opName, std::vector<int64_t> &operandIdxes) {
  StringRef opNameStringRef(opName);
  SmallVector<int64_t> operandIdxesSmallVector(operandIdxes.begin(),
                                               operandIdxes.end());
  addInsertMemReconfig(opNameStringRef, operandIdxesSmallVector);
}

void OptimizerOverridesHandler::addOutputLayoutOverrideNanobindWrapper(
    std::string opName, OutputLayoutOverrideParams overrideParams) {
  StringRef opNameStringRef(opName);
  addOutputLayoutOverride(opNameStringRef, overrideParams);
}

void OptimizerOverridesHandler::addConv2dConfigOverrideNanobindWrapper(
    std::string opName, Conv2dConfigOverrideParams overrideParams) {
  StringRef opNameStringRef(opName);
  addConv2dConfigOverride(opNameStringRef, overrideParams);
}

} // namespace mlir::tt::ttnn
