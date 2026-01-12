// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPRETTIFYUTILS_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPRETTIFYUTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttnn {

// Forward declarations
struct PyLoc;
struct OpPyLoc;
struct FuncGroup;
struct FunctionBoundaryInfo;
struct CompareOpPyLoc;

// Utility function to convert MLIR Location to string format
std::string locationToStr(const mlir::Location &loc);

// This structure is used to store the location information of an operation in
// a more user-friendly format.
//
struct PyLoc {
  // This structure is used to store the original module information.
  //
  struct Module {
    std::string moduleClass;
    std::string moduleName;

    std::string toString() const {
      return moduleClass + "[" + moduleName + "]";
    }
  };

  // Op location is parsed into these fields.
  //
  int opIndex;
  llvm::SmallVector<Module> modules;
  std::string funcPath;
  std::string funcName;
  int opLineNum;
  std::string opName;

  Operation *op;
  bool isValid;

  // Constructor that parses location from operation
  PyLoc(Operation *op);
};

// This structure is used to store the operation, its PyLoc information, and
// other metadata relevant for the "prettification" algorithm.
//
struct OpPyLoc {
  Operation *op;
  PyLoc pyLoc;
  int distanceFromRoot;
};

// This structure is used to store the operations and their PyLocs in a group
// of operations that belong to the same module/function.
//
struct FuncGroup {
  // What the function will be named in the new IR.
  //
  std::string funcName;

  int index = -1;

  // The operations and their PyLocs in the group.
  //
  llvm::SmallVector<OpPyLoc> opPyLocs;

  // Generate name based on modules of the ops in the group.
  //
  void generateFuncName();
};

// Information about data flow for a function group.
//
struct FunctionBoundaryInfo {
  std::string funcName;
  std::string funcPath;
  llvm::SmallVector<OpPyLoc> opPyLocs;

  // Values that flow INTO this function (used but not defined here).
  //
  llvm::SmallVector<Value> inputValues;

  // Values that flow OUT of this function (defined here, used elsewhere).
  //
  llvm::SmallVector<Value> outputValues;

  // Values that are internal (defined and used only within this function).
  //
  llvm::SmallVector<Value> internalValues;
};

// This comparator is used to compare two OpPyLoc objects and determine which
// one should be "worked on" next. Using it helps provide a topological sort
// that is useful in handling forked paths in graph such that ops chosen first
// are more likely to belong to the original module.
//
struct CompareOpPyLoc {
  std::string *currentFuncName;

  CompareOpPyLoc(std::string *currentFuncName)
      : currentFuncName(currentFuncName) {}

  bool operator()(const OpPyLoc &a, const OpPyLoc &b) const;
};

} // namespace mlir::tt::ttnn
#endif
