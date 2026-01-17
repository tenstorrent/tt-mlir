// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_MATHFIDELITYPARSER_H
#define TTMLIR_DIALECT_TTNN_UTILS_MATHFIDELITYPARSER_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "llvm/Support/CommandLine.h"
#include <optional>

namespace llvm::cl {

// Template specialization of llvm::cl::parser for
// mlir::tt::ttnn::OptionalMathFidelity
// This enables command-line parsing of optional math fidelity options for MLIR
// passes
template <>
class parser<mlir::tt::ttnn::OptionalMathFidelity>
    : public basic_parser<mlir::tt::ttnn::OptionalMathFidelity> {
public:
  parser(Option &opt)
      : basic_parser<mlir::tt::ttnn::OptionalMathFidelity>(opt) {}

  // Called by clEnumValN to register enum values
  void addLiteralOption(StringRef name, int value, StringRef helpStr) {
    // This method is called during option setup but we don't need to store
    // anything as we use symbolizeOptionalMathFidelity() directly in parse()
  }

  bool parse(Option &opt, StringRef argName, StringRef arg,
             mlir::tt::ttnn::OptionalMathFidelity &value) {
    // Try to symbolize the string using the generated function
    std::optional<mlir::tt::ttnn::OptionalMathFidelity> result =
        mlir::tt::ttnn::symbolizeOptionalMathFidelity(arg);
    if (result.has_value()) {
      value = *result;
      return false; // Success
    }

    // If symbolization failed, return error
    return opt.error("Invalid value '" + arg.str() +
                     "' for math-fidelity. Valid values are: lofi, hifi2, "
                     "hifi3, hifi4, undefined");
  }

  void print(raw_ostream &os,
             const mlir::tt::ttnn::OptionalMathFidelity &value) {
    os << mlir::tt::ttnn::stringifyOptionalMathFidelity(value);
  }

  void printOptionDiff(
      const Option &opt,
      const OptionValue<mlir::tt::ttnn::OptionalMathFidelity> &value,
      const OptionValue<mlir::tt::ttnn::OptionalMathFidelity> &defaultValue,
      size_t globalWidth) const {
    printOptionName(opt, globalWidth);
    std::string defaultStr =
        mlir::tt::ttnn::stringifyOptionalMathFidelity(defaultValue.getValue())
            .str();
    std::string valueStr =
        mlir::tt::ttnn::stringifyOptionalMathFidelity(value.getValue()).str();
    outs() << "= " << valueStr;
    if (defaultValue.getValue() != value.getValue()) {
      outs() << " (default: " << defaultStr << ")";
    }
    outs() << "\n";
  }
};

// Template specialization of llvm::cl::parser for
// std::optional<mlir::tt::ttnn::MathFidelity> This enables command-line parsing
// of math fidelity options for MLIR passes
template <>
class parser<std::optional<mlir::tt::ttnn::MathFidelity>>
    : public basic_parser<std::optional<mlir::tt::ttnn::MathFidelity>> {
public:
  parser(Option &opt)
      : basic_parser<std::optional<mlir::tt::ttnn::MathFidelity>>(opt) {}

  bool parse(Option &opt, StringRef argName, StringRef arg,
             std::optional<mlir::tt::ttnn::MathFidelity> &value) {
    // Try to symbolize the string using the generated function
    std::optional<mlir::tt::ttnn::MathFidelity> result =
        mlir::tt::ttnn::symbolizeMathFidelity(arg);
    if (result.has_value()) {
      value = result;
      return false; // Success
    }

    // If symbolization failed, return error
    return opt.error("Invalid value '" + arg.str() +
                     "' for math-fidelity. Valid values are: lofi, hifi2, "
                     "hifi3, hifi4");
  }

  void print(raw_ostream &os,
             const std::optional<mlir::tt::ttnn::MathFidelity> &value) {
    if (!value.has_value()) {
      os << "none";
    } else {
      os << mlir::tt::ttnn::stringifyMathFidelity(*value);
    }
  }

  void printOptionDiff(
      const Option &opt,
      const OptionValue<std::optional<mlir::tt::ttnn::MathFidelity>> &value,
      const OptionValue<std::optional<mlir::tt::ttnn::MathFidelity>>
          &defaultValue,
      size_t globalWidth) const {
    printOptionName(opt, globalWidth);
    std::string defaultStr =
        defaultValue.getValue().has_value()
            ? mlir::tt::ttnn::stringifyMathFidelity(*defaultValue.getValue())
                  .str()
            : "none";
    std::string valueStr =
        value.getValue().has_value()
            ? mlir::tt::ttnn::stringifyMathFidelity(*value.getValue()).str()
            : "none";
    outs() << "= " << valueStr;
    if (defaultValue.getValue() != value.getValue()) {
      outs() << " (default: " << defaultStr << ")";
    }
    outs() << "\n";
  }
};

} // namespace llvm::cl

#endif // TTMLIR_DIALECT_TTNN_UTILS_MATHFIDELITYPARSER_H
