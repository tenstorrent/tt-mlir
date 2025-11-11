// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNPRETTIFYFORCODEGEN
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNPrettifyForCodegen
    : public impl::TTNNPrettifyForCodegenBase<TTNNPrettifyForCodegen> {

private:
  // @dataclass
  // class LocationModuleCodegen:
  //     module_class: str
  //     module_name: str

  // @dataclass
  // class LocationCodegen:
  //     modules: list[LocationModuleCodegen]
  //     func_path: str
  //     func_name: str
  //     op_line_num: int
  //     op_name: str
  struct PyLoc {
    struct Module {
      std::string moduleClass;
      std::string moduleName;
    };

    llvm::SmallVector<Module> modules;
    std::string funcPath;
    std::string funcName;
    int opLineNum;
    std::string opName;

    Operation *op;
    bool isValid;

    PyLoc(Operation *op) {
      this->op = op;

      // Get location without "loc(" and ")" characters.
      std::string locStr = locationToStr(op->getLoc());

      // Split locStr by "|" character.
      // For example, given:
      //   "Tail[tail]|ReLU[tail.relu]|/localdev/.../test.py:106|forward|107|aten__relu"
      // Return:
      //   ["Tail[tail]", "ReLU[tail.relu]", "/localdev/.../test.py:106",
      //   "forward", "107", "aten__relu"]
      llvm::SmallVector<llvm::StringRef, 5> locParts;
      llvm::StringRef(locStr).split(locParts, "|", -1, false);

      // Validate that we have at least 4 parts (funcPath, funcName, opLineNum,
      // opName)
      size_t n = locParts.size();
      if (n < 4) {
        this->isValid = false;
        return;
      }

      // Fill in fields from back of locParts.
      this->opName = locParts[n - 1].str();
      this->opLineNum = std::stoi(locParts[n - 2].str());
      this->funcName = locParts[n - 3].str();
      this->funcPath = locParts[n - 4].str();
      this->modules = llvm::SmallVector<Module>();
      for (size_t i = 0; i < n - 4; i++) {
        // Split each module into class and name.
        // For example, given:
        //   "Tail[tail]"
        // Return:
        //   ["Tail", "tail"]
        llvm::SmallVector<llvm::StringRef, 2> moduleParts;
        locParts[i].split(moduleParts, "[", -1, false);
        std::cout << locStr << "\n";
        this->modules.push_back(
            Module{/* moduleClass= */ moduleParts[0].str(),
                   // Remove trailing "]" from module name.
                   /* moduleName= */ moduleParts[1].str().substr(
                       0, moduleParts[1].str().size() - 1)});
      }
      this->isValid = true;
    }
  };

  void printFnInfo(func::FuncOp funcOp) {
    llvm::outs() << "Fn: " << funcOp.getName() << "\n";
    std::cout << "  Inputs count: "
              << funcOp.getFunctionType().getInputs().size() << "\n";
    llvm::outs() << "  Results count: "
                 << funcOp.getFunctionType().getResults().size() << "\n";
    llvm::outs() << "  Is private: " << funcOp.isPrivate() << "\n";
    llvm::outs() << "  Is const-eval: "
                 << ttmlir::utils::isConstEvalFunc(funcOp) << "\n";
    llvm::outs() << "  Is candidate: " << isCandidateFn(funcOp) << "\n";
  }

  bool isCandidateFn(func::FuncOp funcOp) {
    return !funcOp.isPrivate() && !ttmlir::utils::isConstEvalFunc(funcOp) &&
           !funcOp.getFunctionType().getInputs().empty();
  }

  bool isCandidateOp(Operation *op) {
    // Check if ttnn op
    return isa<ttnn::TTNNDialect>(op->getDialect());
  }

  static std::string locationToStr(const mlir::Location &loc) {
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

  SmallVector<func::FuncOp> findCandidateFns(ModuleOp moduleOp) {
    SmallVector<func::FuncOp, 1> candidateFns;
    moduleOp->walk([&](func::FuncOp funcOp) {
      // printFnInfo(funcOp);
      if (isCandidateFn(funcOp)) {
        candidateFns.push_back(funcOp);
      }
    });
    return candidateFns;
  }

  SmallVector<PyLoc> parseOpsAndGatherLocations(func::FuncOp funcOp) {
    SmallVector<PyLoc> locations;

    funcOp.walk([&](Operation *op) {
      // llvm::outs() << "Op: " << op->getName() << "\n";
      if (!isCandidateOp(op)) {
        return WalkResult::advance();
      }

      PyLoc pyLoc(op);
      if (pyLoc.isValid) {
        locations.push_back(pyLoc);
      }
      return WalkResult::advance();
    });

    return locations;
  }

  void printPyLocs(func::FuncOp candidateFn, SmallVector<PyLoc> locations) {
    for (PyLoc &pyLoc : locations) {
      llvm::outs() << "PyLoc: " << pyLoc.op->getName() << "\n";
      llvm::outs() << "  Loc: " << pyLoc.op->getLoc() << "\n";
      llvm::outs() << "  Func path: " << pyLoc.funcPath << "\n";
      llvm::outs() << "  Func name: " << pyLoc.funcName << "\n";
      llvm::outs() << "  Op line num: " << pyLoc.opLineNum << "\n";
      llvm::outs() << "  Op name: " << pyLoc.opName << "\n";
      llvm::outs() << "  Modules: " << pyLoc.modules.size() << "\n";
      for (PyLoc::Module &module : pyLoc.modules) {
        llvm::outs() << "    Module: " << module.moduleClass << "["
                     << module.moduleName << "]\n";
      }
    }
  }

  std::string validateLocations(SmallVector<PyLoc> locations) {
    llvm::StringMap<PyLoc> funcPathMap;
    for (PyLoc &pyLoc : locations) {
      auto [it, success] = funcPathMap.try_emplace(pyLoc.funcPath, pyLoc);
      if (!success && it->second.funcName != pyLoc.funcName) {
        return "Found different func names at the same path: " +
               pyLoc.funcPath + " (first func name: " + it->second.funcName +
               ", second func name: " + pyLoc.funcName + ")";
      }
    }
    return "";
  }

  // Group operations by their funcPath.
  // Returns a map: funcPath -> (funcName, vector of PyLocs)
  llvm::StringMap<std::pair<std::string, SmallVector<PyLoc>>>
  groupOperationsByFunction(SmallVector<PyLoc> &locations) {
    llvm::StringMap<std::pair<std::string, SmallVector<PyLoc>>> funcGroups;

    for (PyLoc &pyLoc : locations) {
      auto it = funcGroups.find(pyLoc.funcPath);
      if (it == funcGroups.end()) {
        // First time seeing this funcPath
        SmallVector<PyLoc> ops;
        ops.push_back(pyLoc);
        funcGroups[pyLoc.funcPath] = {pyLoc.funcName, std::move(ops)};
      } else {
        // Add to existing group
        it->second.second.push_back(pyLoc);
      }
    }

    return funcGroups;
  }

  void printFunctionGroups(
      const llvm::StringMap<std::pair<std::string, SmallVector<PyLoc>>>
          &funcGroups) {
    llvm::outs() << "\n=== Function Groups ===\n";
    for (const auto &[funcPath, funcInfo] : funcGroups) {
      const std::string &funcName = funcInfo.first;
      const SmallVector<PyLoc> &locations = funcInfo.second;

      llvm::outs() << "\nFunction: " << funcName << "\n";
      llvm::outs() << "  Path: " << funcPath << "\n";
      llvm::outs() << "  Operations: " << locations.size() << "\n";
      for (const PyLoc &pyLoc : locations) {
        llvm::outs() << "    - " << pyLoc.op->getName() << " (line "
                     << pyLoc.opLineNum << ")\n";
      }
    }
    llvm::outs() << "\n";
  }

  // Information about data flow for a function group
  struct FunctionBoundaryInfo {
    std::string funcName;
    std::string funcPath;
    SmallVector<PyLoc> locations;

    // Values that flow INTO this function (used but not defined here)
    SmallVector<Value> inputValues;

    // Values that flow OUT of this function (defined here, used elsewhere)
    SmallVector<Value> outputValues;

    // Values that are internal (defined and used only within this function)
    SmallVector<Value> internalValues;
  };

  // Analyze data flow for each function group
  llvm::StringMap<FunctionBoundaryInfo> analyzeFunctionBoundaries(
      const llvm::StringMap<std::pair<std::string, SmallVector<PyLoc>>>
          &funcGroups) {
    llvm::StringMap<FunctionBoundaryInfo> boundaryInfos;

    for (const auto &[funcPath, funcInfo] : funcGroups) {
      const std::string &funcName = funcInfo.first;
      const SmallVector<PyLoc> &locations = funcInfo.second;

      FunctionBoundaryInfo info;
      info.funcName = funcName;
      info.funcPath = funcPath.str();
      info.locations = locations;

      // Build a set of operations in this group for fast lookup
      llvm::DenseSet<Operation *> opsInGroup;
      for (const PyLoc &pyLoc : locations) {
        opsInGroup.insert(pyLoc.op);
      }

      // Track all values defined within this group
      llvm::DenseSet<Value> valuesDefinedInGroup;
      for (const PyLoc &pyLoc : locations) {
        for (Value result : pyLoc.op->getResults()) {
          valuesDefinedInGroup.insert(result);
        }
      }

      // Analyze each operation's operands and results
      llvm::DenseSet<Value> inputValuesSet;
      llvm::DenseSet<Value> outputValuesSet;
      llvm::DenseSet<Value> internalValuesSet;

      for (const PyLoc &pyLoc : locations) {
        // Check operands (inputs to the op)
        for (Value operand : pyLoc.op->getOperands()) {
          // If this value is not defined in this group, it's an input
          if (!valuesDefinedInGroup.contains(operand)) {
            inputValuesSet.insert(operand);
          }
        }

        // Check results (outputs from the op)
        for (Value result : pyLoc.op->getResults()) {
          bool usedOutside = false;
          bool usedInside = false;

          // Check if this result is used outside this group
          for (Operation *user : result.getUsers()) {
            if (opsInGroup.contains(user)) {
              usedInside = true;
            } else {
              usedOutside = true;
            }
          }

          if (usedOutside) {
            outputValuesSet.insert(result);
          } else if (usedInside) {
            internalValuesSet.insert(result);
          }
          // Note: if a value is not used at all, we don't track it
        }
      }

      // Convert sets to vectors
      info.inputValues.assign(inputValuesSet.begin(), inputValuesSet.end());
      info.outputValues.assign(outputValuesSet.begin(), outputValuesSet.end());
      info.internalValues.assign(internalValuesSet.begin(),
                                 internalValuesSet.end());

      boundaryInfos[funcPath] = std::move(info);
    }

    return boundaryInfos;
  }

  void printFunctionBoundaries(
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos) {
    llvm::outs() << "\n=== Function Boundary Analysis ===\n";
    for (const auto &[funcPath, info] : boundaryInfos) {
      llvm::outs() << "\nFunction: " << info.funcName << "\n";
      llvm::outs() << "  Path: " << info.funcPath << "\n";
      llvm::outs() << "  Input values: " << info.inputValues.size() << "\n";
      for (Value input : info.inputValues) {
        llvm::outs() << "    - " << input << " (type: " << input.getType()
                     << ")\n";
      }
      llvm::outs() << "  Output values: " << info.outputValues.size() << "\n";
      for (Value output : info.outputValues) {
        llvm::outs() << "    - " << output << " (type: " << output.getType()
                     << ")\n";
      }
      llvm::outs() << "  Internal values: " << info.internalValues.size()
                   << "\n";
    }
    llvm::outs() << "\n";
  }

  // Create new function declarations based on boundary information
  llvm::StringMap<func::FuncOp> createNewFunctions(
      IRRewriter &rewriter, func::FuncOp candidateFn,
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos) {
    llvm::StringMap<func::FuncOp> newFunctions;

    // Track function name counts to generate unique names
    llvm::StringMap<unsigned> funcNameCounts;

    for (const auto &[funcPath, info] : boundaryInfos) {
      // Generate unique function name with suffix
      unsigned count = funcNameCounts[info.funcName]++;
      std::string uniqueName = info.funcName + "_" + std::to_string(count);

      // Collect input types from input values
      SmallVector<Type> inputTypes;
      for (Value input : info.inputValues) {
        inputTypes.push_back(input.getType());
      }

      // Collect output types from output values
      SmallVector<Type> outputTypes;
      for (Value output : info.outputValues) {
        outputTypes.push_back(output.getType());
      }

      // Create function type
      FunctionType funcType =
          FunctionType::get(rewriter.getContext(), inputTypes, outputTypes);

      // Set insertion point to after the candidate function in the same parent
      // module
      rewriter.setInsertionPointAfter(candidateFn);

      // Create the new function
      func::FuncOp newFunc = rewriter.create<func::FuncOp>(
          candidateFn.getLoc(), uniqueName, funcType);

      // Mark as private for now (we'll make the entry point public later)
      newFunc.setPrivate();

      // Store the function
      newFunctions[funcPath] = newFunc;
    }

    return newFunctions;
  }

  void printNewFunctions(const llvm::StringMap<func::FuncOp> &newFunctions) {
    llvm::outs() << "\n=== Created Functions ===\n";
    for (const auto &entry : newFunctions) {
      llvm::StringRef funcPath = entry.getKey();
      func::FuncOp funcOp = entry.getValue();

      llvm::outs() << "\nFunction: " << funcOp.getName() << "\n";
      llvm::outs() << "  Path: " << funcPath << "\n";
      llvm::outs() << "  Signature: " << funcOp.getFunctionType() << "\n";
      llvm::outs() << "  Input types: "
                   << funcOp.getFunctionType().getInputs().size() << "\n";
      for (Type inputType : funcOp.getFunctionType().getInputs()) {
        llvm::outs() << "    - " << inputType << "\n";
      }
      llvm::outs() << "  Output types: "
                   << funcOp.getFunctionType().getResults().size() << "\n";
      for (Type outputType : funcOp.getFunctionType().getResults()) {
        llvm::outs() << "    - " << outputType << "\n";
      }
    }
    llvm::outs() << "\n";
  }

  // Populate function bodies by cloning operations from the original function
  void populateFunctionBodies(
      IRRewriter &rewriter,
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos,
      llvm::StringMap<func::FuncOp> &newFunctions) {

    // For each function, we need to:
    // 1. Create entry block with arguments
    // 2. Clone operations in order
    // 3. Map old values to new values
    // 4. Add return statement

    for (const auto &entry : newFunctions) {
      llvm::StringRef funcPath = entry.getKey();
      func::FuncOp funcOp = entry.getValue();
      const FunctionBoundaryInfo &info = boundaryInfos.lookup(funcPath);

      // Create entry block
      Block *entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // Create value mapping: old values -> new values
      IRMapping valueMapping;

      // Map input values to block arguments
      for (size_t i = 0; i < info.inputValues.size(); i++) {
        valueMapping.map(info.inputValues[i], entryBlock->getArgument(i));
      }

      // Clone operations in order
      for (const PyLoc &pyLoc : info.locations) {
        Operation *oldOp = pyLoc.op;
        Operation *newOp = rewriter.clone(*oldOp, valueMapping);

        // Update the value mapping with the results of the cloned operation
        for (size_t i = 0; i < oldOp->getNumResults(); i++) {
          valueMapping.map(oldOp->getResult(i), newOp->getResult(i));
        }
      }

      // Collect output values (now mapped to new function's values)
      SmallVector<Value> returnValues;
      for (Value oldOutput : info.outputValues) {
        Value newOutput = valueMapping.lookup(oldOutput);
        returnValues.push_back(newOutput);
      }

      // Add return statement
      rewriter.create<func::ReturnOp>(funcOp.getLoc(), returnValues);
    }
  }

  // Replace operations in the original function with calls to new functions
  void replaceWithFunctionCalls(
      IRRewriter &rewriter, func::FuncOp candidateFn,
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos,
      const llvm::StringMap<func::FuncOp> &newFunctions) {

    // We need to process functions in the order they appear in the original IR
    // Build a map from operations to their function info
    llvm::DenseMap<Operation *,
                   std::pair<llvm::StringRef, const FunctionBoundaryInfo *>>
        opToFuncInfo;

    for (const auto &entry : boundaryInfos) {
      llvm::StringRef funcPath = entry.getKey();
      const FunctionBoundaryInfo &info = entry.getValue();
      for (const PyLoc &pyLoc : info.locations) {
        opToFuncInfo[pyLoc.op] = {funcPath, &info};
      }
    }

    // Track which operations have been processed
    llvm::DenseSet<Operation *> processedOps;

    // Mapping from old values to new values (call results)
    IRMapping valueMapping;

    // Set insertion point to the beginning of the candidate function
    rewriter.setInsertionPointToStart(&candidateFn.getBody().front());

    // Walk through the original function in order
    SmallVector<Operation *> opsToReplace;
    candidateFn.walk([&](Operation *op) {
      if (opToFuncInfo.count(op)) {
        opsToReplace.push_back(op);
      }
    });

    // Process operations in order, grouping consecutive ops from the same
    // function
    llvm::StringRef currentFuncPath;
    SmallVector<Operation *> currentGroupOps;

    auto processGroup = [&]() {
      if (currentGroupOps.empty()) {
        return;
      }

      const auto &[funcPath, info] = opToFuncInfo[currentGroupOps.front()];
      func::FuncOp targetFunc = newFunctions.lookup(funcPath);

      // Prepare operands for the call (input values)
      SmallVector<Value> callOperands;
      for (Value inputValue : info->inputValues) {
        // If this input was produced by a previous call, use the mapped value
        if (valueMapping.contains(inputValue)) {
          callOperands.push_back(valueMapping.lookup(inputValue));
        } else {
          callOperands.push_back(inputValue);
        }
      }

      // Set insertion point to after the last operation we're about to replace
      Operation *lastOp = currentGroupOps.back();
      rewriter.setInsertionPointAfter(lastOp);

      // Create the call
      func::CallOp callOp = rewriter.create<func::CallOp>(
          lastOp->getLoc(), targetFunc, callOperands);

      // Map output values to call results
      for (size_t i = 0; i < info->outputValues.size(); i++) {
        valueMapping.map(info->outputValues[i], callOp.getResult(i));
      }

      // Replace uses of output values with call results
      // We need to get a mutable reference to the FunctionBoundaryInfo
      FunctionBoundaryInfo &mutableInfo =
          const_cast<FunctionBoundaryInfo &>(*info);
      for (size_t i = 0; i < mutableInfo.outputValues.size(); i++) {
        mutableInfo.outputValues[i].replaceAllUsesWith(callOp.getResult(i));
      }

      // Mark operations for deletion
      for (Operation *op : currentGroupOps) {
        processedOps.insert(op);
      }

      currentGroupOps.clear();
    };

    for (Operation *op : opsToReplace) {
      auto [funcPath, info] = opToFuncInfo[op];

      if (funcPath != currentFuncPath) {
        // Process previous group
        processGroup();
        currentFuncPath = funcPath;
      }

      currentGroupOps.push_back(op);
    }

    // Process final group
    processGroup();

    // Erase replaced operations in reverse order
    for (auto it = opsToReplace.rbegin(); it != opsToReplace.rend(); ++it) {
      if (processedOps.contains(*it)) {
        rewriter.eraseOp(*it);
      }
    }
  }

public:
  using impl::TTNNPrettifyForCodegenBase<
      TTNNPrettifyForCodegen>::TTNNPrettifyForCodegenBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Need a couple passes
    // 1. Find candidate fns
    // 2. Parse and gather IR locations (remember ops)
    // -----
    // 3. Build functree
    // 4. Analyze functree
    // 5. Move from original IR to functree-aligned IR

    // Open questions:
    // - What do we do with const-eval fns?
    // - What makes a fn a candidate fn?

    // For simplicity, supporting only one for now, but can support multiple.
    SmallVector<func::FuncOp> candidateFns = findCandidateFns(moduleOp);
    assert(candidateFns.size() == 1 &&
           "Only one candidate fn is supported now");

    func::FuncOp candidateFn = candidateFns.front();

    llvm::SmallVector<PyLoc> locations =
        parseOpsAndGatherLocations(candidateFn);

    // Debug prints
    printPyLocs(candidateFn, locations);

    // Validate locations
    std::string errorMessage = validateLocations(locations);
    if (!errorMessage.empty()) {
      llvm::outs() << "PrettifyForCodegen error: " << errorMessage << "\n";
      signalPassFailure();
    }

    // Group operations by function
    auto funcGroups = groupOperationsByFunction(locations);
    printFunctionGroups(funcGroups);

    // Analyze function boundaries and data flow
    auto boundaryInfos = analyzeFunctionBoundaries(funcGroups);
    printFunctionBoundaries(boundaryInfos);

    // Create new functions based on boundary information
    auto newFunctions =
        createNewFunctions(rewriter, candidateFn, boundaryInfos);
    printNewFunctions(newFunctions);

    // Populate function bodies with operations
    populateFunctionBodies(rewriter, boundaryInfos, newFunctions);

    // Replace operations in original function with calls to new functions
    replaceWithFunctionCalls(rewriter, candidateFn, boundaryInfos,
                             newFunctions);
  }
};
} // namespace mlir::tt::ttnn
