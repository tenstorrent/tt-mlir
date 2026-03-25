// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_WEIGHTDTYPEPARSER_H
#define TTMLIR_DIALECT_TTNN_UTILS_WEIGHTDTYPEPARSER_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "llvm/Support/CommandLine.h"
#include <optional>

namespace llvm::cl {

template <>
class parser<mlir::tt::ttnn::WeightDtype>
    : public basic_parser<mlir::tt::ttnn::WeightDtype> {
public:
  parser(Option &opt) : basic_parser<mlir::tt::ttnn::WeightDtype>(opt) {}

  void addLiteralOption(StringRef name, int value, StringRef helpStr) {}

  bool parse(Option &opt, StringRef argName, StringRef arg,
             mlir::tt::ttnn::WeightDtype &value) {
    std::optional<mlir::tt::ttnn::WeightDtype> result =
        mlir::tt::ttnn::symbolizeWeightDtype(arg);
    if (result.has_value()) {
      value = *result;
      return false;
    }
    return opt.error("Invalid value '" + arg.str() +
                     "' for weight dtype. Valid values are: none, bfp_bf8, "
                     "bfp_bf4");
  }

  void print(raw_ostream &os, const mlir::tt::ttnn::WeightDtype &value) {
    os << mlir::tt::ttnn::stringifyWeightDtype(value);
  }

  void
  printOptionDiff(const Option &opt,
                  const OptionValue<mlir::tt::ttnn::WeightDtype> &value,
                  const OptionValue<mlir::tt::ttnn::WeightDtype> &defaultValue,
                  size_t globalWidth) const {
    printOptionName(opt, globalWidth);
    std::string defaultStr =
        mlir::tt::ttnn::stringifyWeightDtype(defaultValue.getValue()).str();
    std::string valueStr =
        mlir::tt::ttnn::stringifyWeightDtype(value.getValue()).str();
    outs() << "= " << valueStr;
    if (defaultValue.getValue() != value.getValue()) {
      outs() << " (default: " << defaultStr << ")";
    }
    outs() << "\n";
  }
};

} // namespace llvm::cl

#endif // TTMLIR_DIALECT_TTNN_UTILS_WEIGHTDTYPEPARSER_H
