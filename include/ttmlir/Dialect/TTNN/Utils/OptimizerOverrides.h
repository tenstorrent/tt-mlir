// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZEROVERRIDES_H
#define TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZEROVERRIDES_H


#include <llvm/Support/CommandLine.h>

// #include "mlir/Pass/PassOptions.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
// #include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TT/Utils/MemoryLayoutAnalysisParams.h"

namespace mlir::tt::ttnn {

struct OutputLayoutOverrideParams {
  
  SmallVector<int64_t, 2> grid;
  BufferType bufferType;
  TensorMemoryLayout tensorMemoryLayout; // INTERLEAVED / SHARDED etc...
  Layout memoryLayout;                   // ROW_MAJOR / TILE
  mlir::tt::DataType dataType;

  bool operator==(const OutputLayoutOverrideParams rhs) const {
    return grid[0] == rhs.grid[0] &&
           grid[1] == rhs.grid[1] &&
           bufferType == rhs.bufferType &&
           tensorMemoryLayout == rhs.tensorMemoryLayout &&
           memoryLayout == rhs.memoryLayout &&
           dataType == rhs.dataType;
  }

  bool operator!=(const OutputLayoutOverrideParams &rhs) const {
    return !(*this == rhs);
  }

};

struct InputLayoutOverrideParams {

  SmallVector<int64_t> operandIdxes;

  bool operator==(const InputLayoutOverrideParams &rhs) const {
    if (operandIdxes.size() != rhs.operandIdxes.size())
      return false;
    for (std::size_t i = 0; i < operandIdxes.size(); i++) {
      if (operandIdxes[i] != rhs.operandIdxes[i])
        return false;
    }
    return true;
  }

  bool operator!=(const InputLayoutOverrideParams &rhs) const {
    return !(*this == rhs);
  }

};

struct OutputLayoutOverrideParser
    : public llvm::cl::parser<llvm::StringMap<OutputLayoutOverrideParams>> {
public:
  OutputLayoutOverrideParser(llvm::cl::Option &opt)
      : llvm::cl::parser<llvm::StringMap<OutputLayoutOverrideParams>>(opt) {}

  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             llvm::StringMap<OutputLayoutOverrideParams> &value);

  static void print(llvm::raw_ostream &os,
                    const llvm::StringMap<OutputLayoutOverrideParams> &value);
};

struct InputLayoutOverrideParser
    : public llvm::cl::parser<llvm::StringMap<InputLayoutOverrideParams>> {
public:
  InputLayoutOverrideParser(llvm::cl::Option &opt)
      : llvm::cl::parser<llvm::StringMap<InputLayoutOverrideParams>>(opt) {}

  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             llvm::StringMap<InputLayoutOverrideParams> &value);

  static void print(llvm::raw_ostream &os,
                    const llvm::StringMap<InputLayoutOverrideParams> &value);
};


class OptimizerOverridesHandler {
public:

  OptimizerOverridesHandler() {};
  ~OptimizerOverridesHandler() {};

  // Setters for the overrides
  // These are used to enable/disable the optimizer passes
  void setOptimizerPass(bool);
  // These are used to enable/disable the memory configurations
  void setMemoryConfig(bool);
  void setMemoryLayoutAnalysis(bool);
  void setMemoryLayoutAnalysisPolicy(MemoryLayoutAnalysisPolicyType);
  // These are used to set the input/output layout overrides
  void setInputLayoutOverrides(llvm::StringMap<InputLayoutOverrideParams>&);
  void setOutputLayoutOverrides(llvm::StringMap<OutputLayoutOverrideParams>&);
  // These are used to add system descriptor path
  void setSystemDescPath(std::string);
  // These are used to set the maximum number of legal layouts for grid analysis
  void setMaxLegalLayouts(int64_t);
  // These are used to set the mesh shape
  void setMeshShape(std::vector<int64_t>);

  // Getters for the overrides
  // These are used to get the current state of the optimizer passes
  bool getOptimizerPass() const;
  // These are used to get the current state of the memory configurations
  bool getMemoryConfig() const;
  bool getMemoryLayoutAnalysis() const;
  MemoryLayoutAnalysisPolicyType getMemoryLayoutAnalysisPolicy() const;
  // These are used to get the current input/output layout overrides
  llvm::StringMap<InputLayoutOverrideParams> getInputLayoutOverrides() const;
  llvm::StringMap<OutputLayoutOverrideParams> getOutputLayoutOverrides() const;
  // These are used to get the current system descriptor path
  std::string getSystemDescPath() const;
  // These are used to get the current maximum number of legal layouts for grid analysis
  int64_t getMaxLegalLayouts() const;
  // These are used to get the current mesh shape
  std::vector<int64_t> getMeshShape() const;

  // Method that converts the overrides to a string
  std::string toString() const;

  // Fill input/output layout overrides maps. 
  // This is used from tt-forge frontend where we define and compile the models.
  void addInputLayoutOverride(StringRef, InputLayoutOverrideParams);
  void addInputLayoutOverride(StringRef, SmallVector<int64_t>);
  void addOutputLayoutOverride(StringRef, OutputLayoutOverrideParams);
  void addOutputLayoutOverride(StringRef, SmallVector<int64_t>, BufferType, TensorMemoryLayout, tt::ttnn::Layout, tt::DataType);

private:

  // Flags for enabling/disabling the optimizer passes
  bool enableOptimizerPass = true;

  // Flags for enabling/disabling the memory configurations
  bool enableMemoryConfig = true;
  bool enableMemoryLayoutAnalysis = true;

  // Input layout overrides
  llvm::StringMap<InputLayoutOverrideParams> inputLayoutOverrides;

  // Output layout overrides
  llvm::StringMap<OutputLayoutOverrideParams> outputLayoutOverrides;

  // Memory layout analysis policy
  MemoryLayoutAnalysisPolicyType memoryLayoutAnalysisPolicy;

  // System descriptor path
  std::string systemDescPath;

  // Maximum number of legal layouts for grid analysis
  int64_t maxLegalLayouts;

  // Mesh shape
  std::vector<int64_t> meshShape;

};  // class OptimizerOverridesHandler


} // namespace mlir::tt::ttnn

#endif
