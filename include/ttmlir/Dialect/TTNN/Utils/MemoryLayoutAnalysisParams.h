// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_MEMORYLAYOUTANALYSISPARAMS_H
#define TTMLIR_DIALECT_TTNN_UTILS_MEMORYLAYOUTANALYSISPARAMS_H

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

  static std::string toString(const MemoryLayoutAnalysisPolicyType &value) {
    std::string res;
    switch (value) {
    case MemoryLayoutAnalysisPolicyType::DFSharding:
      res += "DFSharding";
      break;
    case MemoryLayoutAnalysisPolicyType::L1Interleaved:
      res += "L1Interleaved";
      break;
    }
    return res;
  }

  static void print(llvm::raw_ostream &os,
                    const MemoryLayoutAnalysisPolicyType &value) {
    os << "memory-layout-analysis-policy="
       << MemoryLayoutAnalysisPolicyTypeParser::toString(value) << "\n";
  }
};

} // namespace mlir::tt

#endif // TTMLIR_DIALECT_TTNN_UTILS_MEMORYLAYOUTANALYSISPARAMS_H
