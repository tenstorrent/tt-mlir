// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_ACCURACYMODEPARSER_H
#define TTMLIR_DIALECT_TTNN_UTILS_ACCURACYMODEPARSER_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "llvm/Support/CommandLine.h"
#include <optional>

namespace llvm::cl {

template <>
class parser<mlir::tt::ttnn::AccuracyMode>
    : public basic_parser<mlir::tt::ttnn::AccuracyMode> {
public:
  parser(Option &opt) : basic_parser<mlir::tt::ttnn::AccuracyMode>(opt) {}

  void addLiteralOption(StringRef name, int value, StringRef helpStr) {}

  bool parse(Option &opt, StringRef argName, StringRef arg,
             mlir::tt::ttnn::AccuracyMode &value) {
    std::optional<mlir::tt::ttnn::AccuracyMode> result =
        mlir::tt::ttnn::symbolizeAccuracyMode(arg);
    if (result.has_value()) {
      value = *result;
      return false;
    }
    return opt.error("Invalid value '" + arg.str() +
                     "' for accuracy mode. Valid values are: none, accuracy, "
                     "performance");
  }

  void print(raw_ostream &os, const mlir::tt::ttnn::AccuracyMode &value) {
    os << mlir::tt::ttnn::stringifyAccuracyMode(value);
  }

  void
  printOptionDiff(const Option &opt,
                  const OptionValue<mlir::tt::ttnn::AccuracyMode> &value,
                  const OptionValue<mlir::tt::ttnn::AccuracyMode> &defaultValue,
                  size_t globalWidth) const {
    printOptionName(opt, globalWidth);
    std::string defaultStr =
        mlir::tt::ttnn::stringifyAccuracyMode(defaultValue.getValue()).str();
    std::string valueStr =
        mlir::tt::ttnn::stringifyAccuracyMode(value.getValue()).str();
    outs() << "= " << valueStr;
    if (defaultValue.getValue() != value.getValue()) {
      outs() << " (default: " << defaultStr << ")";
    }
    outs() << "\n";
  }
};

} // namespace llvm::cl

#endif // TTMLIR_DIALECT_TTNN_UTILS_ACCURACYMODEPARSER_H
