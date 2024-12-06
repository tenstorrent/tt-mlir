// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_PASSOVERRIDES_H
#define TTMLIR_DIALECT_TTNN_UTILS_PASSOVERRIDES_H

#include <llvm/Support/CommandLine.h>

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

namespace mlir::tt::ttnn {

struct OptionNames {

  static const std::string optimizerPassEnabled;
  static const std::string overrideInputLayout;
  static const std::string overrideOutputLayout;
  static const std::string memoryLayoutAnalysisEnabled;
  static const std::string memReconfigEnabled;
  static const std::string memoryLayoutAnalysisPolicy;
  static const std::string systemDescPath;
  static const std::string maxLegalLayouts;
  static const std::string meshShape;
};

struct OutputLayoutOverrideParams {

  SmallVector<int64_t, 2> grid;
  BufferType bufferType;
  TensorMemoryLayout tensorMemoryLayout; // INTERLEAVED / SHARDED etc...
  Layout memoryLayout;                   // ROW_MAJOR / TILE
  mlir::tt::DataType dataType;

  bool operator==(const OutputLayoutOverrideParams rhs) const {
    return grid[0] == rhs.grid[0] && grid[1] == rhs.grid[1] &&
           bufferType == rhs.bufferType &&
           tensorMemoryLayout == rhs.tensorMemoryLayout &&
           memoryLayout == rhs.memoryLayout && dataType == rhs.dataType;
  }

  bool operator!=(const OutputLayoutOverrideParams &rhs) const {
    return !(*this == rhs);
  }
};

struct InputLayoutOverrideParams {

  SmallVector<int64_t> operandIdxes;

  bool operator==(const InputLayoutOverrideParams &rhs) const {
    if (operandIdxes.size() != rhs.operandIdxes.size()) {
      return false;
    }
    for (std::size_t i = 0; i < operandIdxes.size(); i++) {
      if (operandIdxes[i] != rhs.operandIdxes[i]) {
        return false;
      }
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

  static std::string
  toString(const llvm::StringMap<OutputLayoutOverrideParams> &);

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

  static std::string
  toString(const llvm::StringMap<InputLayoutOverrideParams> &);

  static void print(llvm::raw_ostream &os,
                    const llvm::StringMap<InputLayoutOverrideParams> &value);
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_UTILS_PASSOVERRIDES_H
