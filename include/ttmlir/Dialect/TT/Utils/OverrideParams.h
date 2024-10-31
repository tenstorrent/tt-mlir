// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_UTILS_OVERRIDEPARAMS_H
#define TTMLIR_DIALECT_TT_UTILS_OVERRIDEPARAMS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include <llvm/Support/CommandLine.h>

namespace mlir::tt {

struct InputLayoutOverrideParams {
  SmallVector<int64_t> operandIdxes;
};

struct InputLayoutOverrideParser
    : public llvm::cl::parser<llvm::StringMap<InputLayoutOverrideParams>> {
public:
  InputLayoutOverrideParser(llvm::cl::Option &opt)
      : llvm::cl::parser<llvm::StringMap<InputLayoutOverrideParams>>(opt) {}

  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
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

  static void print(llvm::raw_ostream &os,
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
};

struct OutputLayoutOverrideParams {
  SmallVector<int64_t, 2> grid;
  MemorySpace memorySpace;
  TensorMemoryLayout memoryLayout;
};

struct OutputLayoutOverrideParser
    : public llvm::cl::parser<llvm::StringMap<OutputLayoutOverrideParams>> {
public:
  OutputLayoutOverrideParser(llvm::cl::Option &opt)
      : llvm::cl::parser<llvm::StringMap<OutputLayoutOverrideParams>>(opt) {}

  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             llvm::StringMap<OutputLayoutOverrideParams> &value) {
    SmallVector<StringRef> opOverrideList;
    constexpr size_t kMaxGridSize = 2;
    constexpr size_t kvPairSize = 2;
    constexpr size_t kMaxLayoutOverrideParams = 3;
    constexpr size_t iOpName = 0;
    constexpr size_t iLayoutOverrideParams = 1;
    constexpr size_t iGrid = 0;
    constexpr size_t iMemorySpace = 1;
    constexpr size_t iMemoryLayout = 2;
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
      std::optional<mlir::tt::MemorySpace> memorySpace =
          mlir::tt::symbolizeMemorySpace(layoutParamParts[iMemorySpace]);
      if (!memorySpace.has_value()) {
        opt.error("Invalid memory space: " + layoutParamParts[iMemorySpace]);
        return true;
      }

      // Parse tensor memory layout.
      std::optional<mlir::tt::TensorMemoryLayout> memoryLayout =
          mlir::tt::symbolizeTensorMemoryLayout(
              layoutParamParts[iMemoryLayout]);
      if (!memoryLayout.has_value()) {
        opt.error("Invalid tensor memory layout: " +
                  layoutParamParts[iMemoryLayout]);
        return true;
      }

      // Set parsed op overrides.
      value[opOverrideParts[iOpName]] = OutputLayoutOverrideParams{
          grid, memorySpace.value(), memoryLayout.value()};
    }
    return false;
  }

  static void print(llvm::raw_ostream &os,
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
      os << ":" << mlir::tt::stringifyMemorySpace(params.memorySpace);
      os << ":" << mlir::tt::stringifyTensorMemoryLayout(params.memoryLayout);
      if (++count < value.size()) {
        os << ",";
      }
    }
    os << "\n";
  }
};

} // namespace mlir::tt

#endif
