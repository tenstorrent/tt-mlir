// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNUNIQUELOCATIONS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNUniqueLocations
    : public impl::TTNNUniqueLocationsBase<TTNNUniqueLocations> {

public:
  TTNNUniqueLocations() = default;

  // Check if all operations have unique locations.
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    llvm::StringMap<Operation *> locationMap;
    bool hasNonUniqueLocations = false;

    moduleOp.walk([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      if (!isa<RankedTensorType>(op->getResult(0).getType())) {
        return;
      }

      if (not isa<NameLoc>(op->getLoc())) {
        return;
      }

      if (mlir::isa<ToLayoutOp>(op)) {
        return;
      }

      llvm::StringRef locStr = mlir::cast<NameLoc>(op->getLoc()).getName();
      auto it = locationMap.find(locStr);
      if (!locationMap.try_emplace(locStr, op).second) {
        hasNonUniqueLocations = true;

        op->emitError() << "Operation '" << op->getName()
                        << "' has a non-unique location: '" << locStr
                        << "'. Previously seen in operation '"
                        << it->second->getName() << "'";
      }
    });

    if (hasNonUniqueLocations) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
