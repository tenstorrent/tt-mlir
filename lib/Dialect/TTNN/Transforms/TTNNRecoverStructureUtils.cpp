// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/TTNNRecoverStructureUtils.h"

#include "ttmlir/Support/Logger.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttnn {

std::string locationToStr(const mlir::Location &loc) {
  std::string locStr;
  llvm::raw_string_ostream(locStr) << loc;

  // Remove the loc(" and ") characters
  if (locStr.find("loc(\"") == 0) {
    locStr = locStr.substr(5);
  }
  if (locStr.find("\")") == locStr.size() - 2) {
    locStr = locStr.substr(0, locStr.size() - 2);
  }

  return locStr;
}

PyLoc::PyLoc(Operation *op) {
  this->op = op;

  // Skip unknown locs.
  if (isa<UnknownLoc>(op->getLoc())) {
    this->isValid = false;
    return;
  }

  // Non TTNN dialect ops are not target.
  if (!isa<ttnn::TTNNDialect>(op->getDialect())) {
    this->isValid = false;
    return;
  }

  // Allow NameLocs only.
  if (!isa<NameLoc>(op->getLoc())) {
    // TODO (svuckovic): should we fail the pass for this?
    this->isValid = false;
    return;
  }

  // Get location without "loc(" and ")" characters.
  std::string locStr = locationToStr(op->getLoc());

  TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tParsing loc: {}",
               locStr);

  // Split locStr by "|" character.
  // For example, given:
  //   "7|Tail[tail]|ReLU[tail.relu]|/localdev/.../test.py:106|forward|107|aten__relu"
  // Return:
  //   ["7", "Tail[tail]", "ReLU[tail.relu]", "/localdev/.../test.py:106",
  //   "forward", "107", "aten__relu"]
  llvm::SmallVector<llvm::StringRef> locParts;
  llvm::StringRef(locStr).split(locParts, "|", -1, false);

  bool parseErrorReturned =
      llvm::StringRef(locParts[0].str()).getAsInteger(10, this->opIndex);
  if (parseErrorReturned) {
    emitError(op->getLoc())
        << "Failed to parse op index from location: " << locStr;
    this->isValid = false;
    return;
  }

  // Validate that we have at least 4 parts (funcPath, funcName, opLineNum,
  // opName)
  size_t numParts = locParts.size();
  if (numParts < 4) {
    this->isValid = false;
    return;
  }

  // Fill in fields from back of locParts.
  this->opName = locParts[numParts - 1].str();
  parseErrorReturned = locParts[numParts - 2].getAsInteger(10, this->opLineNum);
  if (parseErrorReturned) {
    emitError(op->getLoc())
        << "Failed to parse op line number from location: " << locStr;
    this->isValid = false;
    return;
  }
  this->funcName = locParts[numParts - 3].str();
  this->funcPath = locParts[numParts - 4].str();
  this->modules = llvm::SmallVector<Module>();
  for (size_t i = 1; i < numParts - 4; i++) {
    // Split each module into class and name.
    // For example, given:
    //   "Tail[tail]"
    // Return:
    //   ["Tail", "tail"]
    llvm::SmallVector<llvm::StringRef, 2> moduleParts;
    locParts[i].split(moduleParts, "[", -1, false);
    this->modules.push_back(
        Module{/* moduleClass= */ moduleParts[0].str(),
               // Remove trailing "]" from module name.
               /* moduleName= */ moduleParts[1].str().substr(
                   0, moduleParts[1].str().size() - 1)});
  }
  this->isValid = true;
}

void FuncGroup::generateFuncName() {
  // clang-format off
  //
  // If we were to list module classes for each op, like this:
  //   - modules: ResNetModel ResNetEncoder ResNetStage ResNetBasicLayer Sequential ResNetConvLayer Conv2d
  //   - modules: ResNetModel ResNetEncoder ResNetStage ResNetBasicLayer Sequential ResNetConvLayer BatchNorm2d
  //   - modules: ResNetModel ResNetEncoder ResNetStage ResNetBasicLayer Sequential ResNetConvLayer ReLU
  // we want to find the LCA of them, which in this case is "ResNetConvLayer"
  //
  // clang-format on
  //
  TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "Func name: {}",
               funcName);
  for ([[maybe_unused]] const OpPyLoc &opPyLoc : opPyLocs) {
    TTMLIR_DEBUG(
        ttmlir::LogComponent::RecoverStructure, "\t- modules: {}",
        llvm::join(llvm::map_range(opPyLoc.pyLoc.modules,
                                   [](const auto &m) { return m.toString(); }),
                   ", "));
    for ([[maybe_unused]] const PyLoc::Module &module : opPyLoc.pyLoc.modules) {
      TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\t\t- {}",
                   module.moduleClass);
    }
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "");
  }

  if (opPyLocs.empty()) {
    this->funcName = "forward";
    if (index >= 0) {
      this->funcName += "_" + std::to_string(index);
    }
    return;
  }

  // First find the smallest length module class list
  size_t minLength = std::numeric_limits<size_t>::max();
  for (const OpPyLoc &opPyLoc : opPyLocs) {
    if (opPyLoc.pyLoc.modules.size() < minLength) {
      minLength = opPyLoc.pyLoc.modules.size();
    }
  }

  // Then go through the module class list from behind and find the first
  // one that all the ops share
  for (int i = minLength - 1; i >= 0; i--) {
    bool allMatch = true;
    std::string moduleClass = opPyLocs[0].pyLoc.modules[i].moduleClass;
    for (const OpPyLoc &opPyLoc : opPyLocs) {
      if (opPyLoc.pyLoc.modules[i].moduleClass != moduleClass) {
        allMatch = false;
        break;
      }
    }

    if (allMatch) {
      this->funcName = moduleClass;
      if (index >= 0) {
        this->funcName += "_" + std::to_string(index);
      }
      return;
    }
  }
}

bool CompareOpPyLoc::operator()(const OpPyLoc &a, const OpPyLoc &b) const {
  // C++ priority_queue convention: return true means b has higher priority
  constexpr bool A_HAS_PRIORITY = false;
  constexpr bool B_HAS_PRIORITY = true;

  bool comparatorDebug = false;

  bool isASameGroup = a.pyLoc.funcPath == *currentFuncName;
  bool isBSameGroup = b.pyLoc.funcPath == *currentFuncName;

  if (comparatorDebug) {
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "Comparing {} and {}",
                 a.op->getName(), b.op->getName());
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "\ta.pyLoc.funcPath: {}", a.pyLoc.funcPath);
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "\tb.pyLoc.funcPath: {}", b.pyLoc.funcPath);
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "\t*currentFuncName: {}", *currentFuncName);
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "\ta.distanceFromRoot: {}", a.distanceFromRoot);
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "\tb.distanceFromRoot: {}", b.distanceFromRoot);
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tisASameGroup: {}",
                 isASameGroup);
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure, "\tisBSameGroup: {}",
                 isBSameGroup);
  }

  bool result;
  std::string reason;

  // First break by group
  //
  if (isASameGroup && !isBSameGroup) {
    result = A_HAS_PRIORITY;
    reason = "a is same group, b is different (a has higher priority)";
  } else if (!isASameGroup && isBSameGroup) {
    result = B_HAS_PRIORITY;
    reason = "b is same group, a is different (b has higher priority)";
  } else if (isASameGroup == isBSameGroup) {
    // Now break by deallocate ops
    //
    bool aIsDeallocate = isa<ttnn::DeallocateOp>(a.op);
    bool bIsDeallocate = isa<ttnn::DeallocateOp>(b.op);
    if (aIsDeallocate && !bIsDeallocate) {
      result = B_HAS_PRIORITY;
      reason = "a is deallocate, b is not (b has higher priority)";
    } else if (!aIsDeallocate && bIsDeallocate) {
      result = A_HAS_PRIORITY;
      reason = "b is deallocate, a is not (a has higher priority)";
    } else {
      // Now break by distance to root
      result = a.distanceFromRoot > b.distanceFromRoot;
      reason = result ? "a has greater distance (b wins)"
                      : "b has greater distance (a wins)";
    }
  } else {
    // This should never happen
    llvm::errs()
        << "Both ops are in the same group and are not ttnn.deallocate ops";
    exit(1);
  }

  if (comparatorDebug) {
    TTMLIR_DEBUG(ttmlir::LogComponent::RecoverStructure,
                 "\tWinner (closer to front): {} ({})",
                 (result ? b.op->getName() : a.op->getName()), reason);
  }

  return result;
}

} // namespace mlir::tt::ttnn
