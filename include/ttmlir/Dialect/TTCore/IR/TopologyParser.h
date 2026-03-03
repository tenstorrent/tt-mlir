// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_IR_TOPOLOGYPARSER_H
#define TTMLIR_DIALECT_TTCORE_IR_TOPOLOGYPARSER_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "llvm/Support/CommandLine.h"

namespace llvm::cl {

// Template specialization of llvm::cl::parser for mlir::tt::ttcore::Topology.
// This enables command-line parsing of topology values ("ring", "linear",
// "disabled") in MLIR pass pipeline option strings, where clEnumValN alone
// is not sufficient (MLIR's OptionParser does not call addLiteralOption).
template <>
class parser<mlir::tt::ttcore::Topology>
    : public basic_parser<mlir::tt::ttcore::Topology> {
public:
  parser(Option &opt) : basic_parser<mlir::tt::ttcore::Topology>(opt) {}

  // Called during option construction via clEnumValN / cl::values, but we
  // use symbolizeTopology() in parse() so no storage is needed here.
  void addLiteralOption(StringRef, int, StringRef) {}

  bool parse(Option &opt, StringRef argName, StringRef arg,
             mlir::tt::ttcore::Topology &value) {
    if (auto result = mlir::tt::ttcore::symbolizeTopology(arg)) {
      value = *result;
      return false;
    }
    return opt.error("Invalid value '" + arg.str() +
                     "' for topology. Valid values are: ring, linear, "
                     "mesh, torus, disabled");
  }

  void print(raw_ostream &os, const mlir::tt::ttcore::Topology &value) {
    os << mlir::tt::ttcore::stringifyTopology(value);
  }

  void printOptionDiff(const Option &opt,
                       const OptionValue<mlir::tt::ttcore::Topology> &value,
                       const OptionValue<mlir::tt::ttcore::Topology> &def,
                       size_t globalWidth) const {
    printOptionName(opt, globalWidth);
    outs() << "= " << mlir::tt::ttcore::stringifyTopology(value.getValue());
    if (def.getValue() != value.getValue()) {
      outs() << " (default: "
             << mlir::tt::ttcore::stringifyTopology(def.getValue()) << ")";
    }
    outs() << "\n";
  }
};

} // namespace llvm::cl

#endif // TTMLIR_DIALECT_TTCORE_IR_TOPOLOGYPARSER_H
