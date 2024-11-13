// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZEROVERRIDES_H
#define TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZEROVERRIDES_H

#include <llvm/Support/CommandLine.h>

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

struct OutputLayoutOverrideParams {
  SmallVector<int64_t, 2> grid;
  tt::MemorySpace memorySpace;
  tt::TensorMemoryLayout tensorMemoryLayout; // INTERLEAVED / SHARDED etc...
  tt::ttnn::Layout memoryLayout;             // ROW_MAJOR / TILE
  tt::DataType dataType;
};

struct InputLayoutOverrideParams {
  SmallVector<int64_t> operandIdxes;
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

} // namespace mlir::tt::ttnn

#endif
