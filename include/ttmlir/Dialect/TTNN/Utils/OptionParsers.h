// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_OPTIONPARSERS_H
#define TTMLIR_DIALECT_TTNN_UTILS_OPTIONPARSERS_H

#include "mlir/Pass/PassOptions.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include <optional>

namespace llvm::cl {

// Custom parser for std::optional<mlir::tt::ttnn::MathFidelity>
// Supports clEnumValN values and special "unset"/"none" for std::nullopt
template <>
class parser<std::optional<mlir::tt::ttnn::MathFidelity>>
    : public basic_parser<std::optional<mlir::tt::ttnn::MathFidelity>> {
public:
  using Enum = mlir::tt::ttnn::MathFidelity;
  using DataType = std::optional<Enum>;

  parser(Option &O) : basic_parser<DataType>(O) {}

  struct OptionInfo {
    // Use const char* instead of StringRef to avoid lifetime issues
    // clEnumValN passes string literals which have static storage duration
    const char *Name;
    DataType V;
    const char *HelpStr;
  };

  // Called by clEnumValN - store enum value wrapped in optional
  void addLiteralOption(StringRef Name, int Value, StringRef HelpStr) {
    // String literals from clEnumValN have static storage, safe to store as
    // const char*
    Values.push_back(
        {Name.data(), DataType(static_cast<Enum>(Value)), HelpStr.data()});
  }

  // Parse the command line argument
  bool parse(Option &O, StringRef ArgName, StringRef Arg, DataType &Val) {
    // Check for special "unset"/"none" tokens
    if (Arg == "unset" || Arg == "none") {
      Val = std::nullopt;
      return false; // success
    }

    // Try to match registered enum values (from clEnumValN)
    for (const auto &V : Values) {
      if (Arg == V.Name) {
        Val = V.V;
        return false; // success
      }
    }

    // If Values is empty (no clEnumValN used), fall back to enum's symbolize
    // function
    if (Values.empty()) {
      auto enumVal = mlir::tt::ttnn::symbolizeMathFidelity(Arg);
      if (enumVal.has_value()) {
        Val = DataType(*enumVal);
        return false; // success
      }
    }

    // Not found
    return O.error(
        "Invalid value '" + Arg +
        "' for option. Valid values: lofi, hifi2, hifi3, hifi4, unset, none");
  }

  // Provide option names for help
  void getExtraOptionNames(SmallVectorImpl<StringRef> &OptionNames) {
    for (const auto &V : Values) {
      OptionNames.push_back(V.Name);
    }
    OptionNames.push_back("unset");
    OptionNames.push_back("none");
  }

  // Required for help printing
  void printOptionDiff(const Option &O, DataType V, const OptVal &Default,
                       size_t GlobalWidth) const {
    printOptionName(O, GlobalWidth);
    outs() << "= ";
    if (V.has_value()) {
      // Find and print enum name
      for (const auto &Val : Values) {
        if (Val.V == V) {
          outs() << Val.Name;
          break;
        }
      }
    } else {
      outs() << "unset";
    }
    outs() << " (default)\n";
  }

private:
  SmallVector<OptionInfo, 8> Values;
};

} // namespace llvm::cl

#endif // TTMLIR_DIALECT_TTNN_UTILS_OPTIONPARSERS_H
