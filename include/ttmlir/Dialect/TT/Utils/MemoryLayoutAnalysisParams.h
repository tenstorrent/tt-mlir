// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_UTILS_MEMORYLAYOUTANALYSISPARAMS_H
#define TTMLIR_DIALECT_TT_UTILS_MEMORYLAYOUTANALYSISPARAMS_H

#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/CommandLine.h>

namespace mlir::tt {

enum class MemoryLayoutAnalysisPolicyType { DFSharding, L1Interleaved };

struct MemoryLayoutAnalysisPolicyTypeParser
    : public llvm::cl::parser<MemoryLayoutAnalysisPolicyType> {
public:
  MemoryLayoutAnalysisPolicyTypeParser(llvm::cl::Option &opt)
      : llvm::cl::parser<MemoryLayoutAnalysisPolicyType>(opt) {}

  bool parse(llvm::cl::Option &opt, llvm::StringRef argName,
             llvm::StringRef arg, MemoryLayoutAnalysisPolicyType &value) {
    value = llvm::StringSwitch<MemoryLayoutAnalysisPolicyType>(arg)
                .Case("DFSharding", MemoryLayoutAnalysisPolicyType::DFSharding)
                .Case("L1Interleaved",
                      MemoryLayoutAnalysisPolicyType::L1Interleaved);
    return false;
  }

  static void print(llvm::raw_ostream &os,
                    const MemoryLayoutAnalysisPolicyType &value) {
    llvm::StringRef policy;
    switch (value) {
    case MemoryLayoutAnalysisPolicyType::DFSharding:
      policy = "DFSharding";
      break;
    case MemoryLayoutAnalysisPolicyType::L1Interleaved:
      policy = "L1Interleaved";
      break;
    }
    os << "memory-layout-analysis-policy=" << policy << "\n";
  }
};

} // namespace mlir::tt

#endif // TTMLIR_DIALECT_TT_UTILS_MEMORYLAYOUTANALYSISPARAMS_H
