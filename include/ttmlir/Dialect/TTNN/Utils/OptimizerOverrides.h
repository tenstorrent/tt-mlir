// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZEROVERRIDES_H
#define TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZEROVERRIDES_H

#include "ttmlir/Dialect/TT/Utils/MemoryLayoutAnalysisParams.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

namespace mlir::tt::ttnn {

class OptimizerOverridesHandler {
public:
  OptimizerOverridesHandler() {};
  ~OptimizerOverridesHandler() {};

  // Setters for the overrides
  // These are used to enable/disable the optimizer passes
  void setEnableOptimizer(bool);
  // These are used to enable/disable the memory configurations
  void setMemoryReconfig(bool);
  void setEnableMemoryLayoutAnalysis(bool);
  void setEnableMemoryLayoutAnalysisPolicy(bool);
  void setMemoryLayoutAnalysisPolicy(MemoryLayoutAnalysisPolicyType);
  // These are used to set the input/output layout overrides
  void setInputLayoutOverrides(llvm::StringMap<InputLayoutOverrideParams> &);
  void setOutputLayoutOverrides(llvm::StringMap<OutputLayoutOverrideParams> &);
  // These are used to add system descriptor path
  void setSystemDescPath(std::string);
  // These are used to set the maximum number of legal layouts for grid analysis
  void setMaxLegalLayouts(int64_t);
  // These are used to set the mesh shape
  void setMeshShape(std::vector<int64_t>);

  // Getters for the overrides
  // These are used to get the current state of the optimizer passes
  bool getEnableOptimizer() const;
  // These are used to get the current state of the memory configurations
  bool getMemoryReconfig() const;
  bool getEnableMemoryLayoutAnalysis() const;
  bool getEnableMemoryLayoutAnalysisPolicy() const;
  MemoryLayoutAnalysisPolicyType getMemoryLayoutAnalysisPolicy() const;
  // These are used to get the current input/output layout overrides
  llvm::StringMap<InputLayoutOverrideParams> getInputLayoutOverrides() const;
  llvm::StringMap<OutputLayoutOverrideParams> getOutputLayoutOverrides() const;
  // These are used to get the current system descriptor path
  std::string getSystemDescPath() const;
  // These are used to get the current maximum number of legal layouts for grid
  // analysis
  int64_t getMaxLegalLayouts() const;
  // These are used to get the current mesh shape
  std::vector<int64_t> getMeshShape() const;

  // Method that converts the overrides to a string
  std::string toString() const;

  // Fill input/output layout overrides maps.
  // This is used from tt-forge frontend where we define and compile the models.
  void addInputLayoutOverride(StringRef, InputLayoutOverrideParams);
  void addInputLayoutOverride(StringRef, SmallVector<int64_t> &);
  void addOutputLayoutOverride(StringRef, OutputLayoutOverrideParams);
  void addOutputLayoutOverride(StringRef, SmallVector<int64_t> &, BufferType,
                               TensorMemoryLayout, tt::ttnn::Layout,
                               tt::DataType);

private:
  // Options for the TTIR to TTNN backend pipeline,
  // we use them to extract the names and the deafulat values.
  TTIRToTTNNBackendPipelineOptions pipelineOptions;

  // Flags for enabling/disabling the optimizer passes
  bool enableOptimizer = false;

  // Flags for enabling/disabling the memory configurations
  bool enableMemoryReconfig = true;
  bool enableMemoryLayoutAnalysis = false;

  // Input layout overrides
  llvm::StringMap<InputLayoutOverrideParams> inputLayoutOverrides;

  // Output layout overrides
  llvm::StringMap<OutputLayoutOverrideParams> outputLayoutOverrides;

  // Memory layout analysis policy
  bool enableMemoryLayoutAnalysisPolicy = false;
  MemoryLayoutAnalysisPolicyType memoryLayoutAnalysisPolicy;

  // System descriptor path
  std::string systemDescPath;

  // Maximum number of legal layouts for grid analysis
  int64_t maxLegalLayouts = 0;

  // Mesh shape
  std::vector<int64_t> meshShape;

}; // class OptimizerOverridesHandler

} // namespace mlir::tt::ttnn

#endif
