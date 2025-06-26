// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZEROVERRIDES_H
#define TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZEROVERRIDES_H

#include <iostream>
#include <string>
#include <unordered_map>

#include "ttmlir/Dialect/TTNN/Utils/MemoryLayoutAnalysisParams.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

namespace mlir::tt::ttnn {

class OptimizerOverridesHandler {
public:
  OptimizerOverridesHandler() {};
  ~OptimizerOverridesHandler() {};

  // Enable/disable the optimizer passes.
  void setEnableOptimizer(bool);
  // Enable/disable the memory configurations.
  void setMemoryReconfig(bool);
  void setEnableMemoryLayoutAnalysis(bool);
  void setEnableMemoryLayoutAnalysisPolicy(bool);
  void setMemoryLayoutAnalysisPolicy(MemoryLayoutAnalysisPolicyType);
  void setInsertMemReconfig(llvm::StringMap<InsertMemReconfigParams> &);
  void setOutputLayoutOverrides(llvm::StringMap<OutputLayoutOverrideParams> &);
  void setConv2dConfigOverrides(llvm::StringMap<Conv2dConfigOverrideParams> &);
  // Add system descriptor path.
  void setSystemDescPath(std::string);
  // Set the maximum number of legal layouts for grid analysis.
  void setMaxLegalLayouts(int64_t);
  void setMeshShape(std::vector<int64_t>);

  // Get the current state of the optimizer passes.
  bool getEnableOptimizer() const;
  // Get the current state of the memory configurations.
  bool getMemoryReconfig() const;
  bool getEnableMemoryLayoutAnalysis() const;
  bool getEnableMemoryLayoutAnalysisPolicy() const;
  MemoryLayoutAnalysisPolicyType getMemoryLayoutAnalysisPolicy() const;
  llvm::StringMap<InsertMemReconfigParams> getInsertMemReconfig() const;
  llvm::StringMap<OutputLayoutOverrideParams> getOutputLayoutOverrides() const;
  llvm::StringMap<Conv2dConfigOverrideParams> getConv2dConfigOverrides() const;
  std::string getSystemDescPath() const;
  // Get the current maximum number of legal layouts for grid analysis.
  int64_t getMaxLegalLayouts() const;
  std::vector<int64_t> getMeshShape() const;

  std::string toString() const;

  // Fill override maps.
  // This is used from tt-forge frontend where we define and compile the models.
  void addInsertMemReconfig(StringRef, InsertMemReconfigParams);
  void addInsertMemReconfig(StringRef, SmallVector<int64_t> &);
  void addOutputLayoutOverride(StringRef, OutputLayoutOverrideParams);
  void addOutputLayoutOverride(StringRef, SmallVector<int64_t> &, BufferType,
                               TensorMemoryLayout, tt::ttnn::Layout,
                               tt::DataType);
  void addConv2dConfigOverride(StringRef, Conv2dConfigOverrideParams);

  std::unordered_map<std::string, InsertMemReconfigParams>
  getInsertMemReconfigNanobindWrapper() const;
  std::unordered_map<std::string, OutputLayoutOverrideParams>
  getOutputLayoutOverridesNanobindWrapper() const;
  std::unordered_map<std::string, Conv2dConfigOverrideParams>
  getConv2dConfigOverridesNanobindWrapper() const;

  void addInsertMemReconfigNanobindWrapper(std::string, std::vector<int64_t> &);
  void addOutputLayoutOverrideNanobindWrapper(std::string,
                                              OutputLayoutOverrideParams);
  void addConv2dConfigOverrideNanobindWrapper(std::string,
                                              Conv2dConfigOverrideParams);

private:
  // Flags for enabling/disabling the optimizer passes
  bool enableOptimizer = false;

  // Flags for enabling/disabling the memory configurations
  bool enableMemoryReconfig = true;
  bool enableMemoryLayoutAnalysis = false;

  llvm::StringMap<InsertMemReconfigParams> insertMemReconfig;

  llvm::StringMap<OutputLayoutOverrideParams> outputLayoutOverrides;

  llvm::StringMap<Conv2dConfigOverrideParams> conv2dConfigOverrides;

  bool enableMemoryLayoutAnalysisPolicy = false;
  MemoryLayoutAnalysisPolicyType memoryLayoutAnalysisPolicy;

  // System descriptor path
  std::string systemDescPath;

  // Maximum number of legal layouts for grid analysis
  int64_t maxLegalLayouts = 0;

  std::vector<int64_t> meshShape;

}; // class OptimizerOverridesHandler

} // namespace mlir::tt::ttnn

#endif
