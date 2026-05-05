// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_BFPDTYPEPARSER_H
#define TTMLIR_DIALECT_TTNN_UTILS_BFPDTYPEPARSER_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "llvm/Support/CommandLine.h"
#include <optional>

namespace mlir::tt::ttnn {

// Maps a BFPDtype pass option to its corresponding ttcore::DataType.
inline ttcore::DataType bfpDtypeToDataType(BFPDtype dtype) {
  switch (dtype) {
  case BFPDtype::BFP_BFloat8:
    return ttcore::DataType::BFP_BFloat8;
  case BFPDtype::BFP_BFloat4:
    return ttcore::DataType::BFP_BFloat4;
  default:
    llvm_unreachable("Invalid BFPDtype for conversion");
  }
}

} // namespace mlir::tt::ttnn

namespace llvm::cl {

template <>
class parser<mlir::tt::ttnn::BFPDtype>
    : public basic_parser<mlir::tt::ttnn::BFPDtype> {
public:
  parser(Option &opt) : basic_parser<mlir::tt::ttnn::BFPDtype>(opt) {}

  void addLiteralOption(StringRef name, int value, StringRef helpStr) {}

  bool parse(Option &opt, StringRef argName, StringRef arg,
             mlir::tt::ttnn::BFPDtype &value) {
    std::optional<mlir::tt::ttnn::BFPDtype> result =
        mlir::tt::ttnn::symbolizeBFPDtype(arg);
    if (result.has_value()) {
      value = *result;
      return false;
    }
    return opt.error("Invalid value '" + arg.str() +
                     "' for BFP dtype. Valid values are: none, bfp_bf8, "
                     "bfp_bf4");
  }

  void print(raw_ostream &os, const mlir::tt::ttnn::BFPDtype &value) {
    os << mlir::tt::ttnn::stringifyBFPDtype(value);
  }

  void
  printOptionDiff(const Option &opt,
                  const OptionValue<mlir::tt::ttnn::BFPDtype> &value,
                  const OptionValue<mlir::tt::ttnn::BFPDtype> &defaultValue,
                  size_t globalWidth) const {
    printOptionName(opt, globalWidth);
    std::string defaultStr =
        mlir::tt::ttnn::stringifyBFPDtype(defaultValue.getValue()).str();
    std::string valueStr =
        mlir::tt::ttnn::stringifyBFPDtype(value.getValue()).str();
    outs() << "= " << valueStr;
    if (defaultValue.getValue() != value.getValue()) {
      outs() << " (default: " << defaultStr << ")";
    }
    outs() << "\n";
  }
};

} // namespace llvm::cl

#endif // TTMLIR_DIALECT_TTNN_UTILS_BFPDTYPEPARSER_H
