// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/Python/PythonEmitter.h"

#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <regex>
#include <stack>
#include <unordered_set>

using namespace mlir;
using namespace mlir::tt::emitpy;
using llvm::formatv;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras which typically
/// do not handle error propagation.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end) {
    return success();
  }
  if (failed(eachFn(*begin))) {
    return failure();
  }
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin))) {
      return failure();
    }
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &container,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(container.begin(), container.end(), eachFn,
                             betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container container,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(container.begin(), container.end(), eachFn,
                             [&]() { os << ", "; });
}

namespace {
/// Emitter that uses dialect specific emitters to emit Python code.
struct PythonEmitter {
  explicit PythonEmitter(raw_ostream &os) : os(os) {
    valueInScopeCount.push(llvm::StringMap<int64_t>{});
  }

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' or returns failure.
  LogicalResult emitOperation(Operation &op);

  /// TODO: implement this for type hints
  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type);

  /// Emits an assignment for a variable which has been declared previously or
  /// returns failure.
  LogicalResult emitVariableAssignment(OpResult result, Operation &op);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits multiple variables separated with comma for multi-valued
  /// operation;
  /// - emits single variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Emits value as an operand of an operation.
  LogicalResult emitOperand(Value value, std::string name);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value value, std::string name);

  // Return the textual representation of a subscript operation.
  std::string getSubscriptName(SubscriptOp op);

  /// Insert the op result into the value cache.
  void cacheDeferredOpResult(Value value, StringRef str);

  /// RAII helper function to manage entering/exiting Python scopes.
  struct Scope {
    Scope(PythonEmitter &emitter)
        : valueMapperScope(emitter.valueMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
    }
    ~Scope() { emitter.valueInScopeCount.pop(); }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    PythonEmitter &emitter;
  };

  /// Returns wether the Value is assigned to a Python variable in the scope.
  bool hasValueInScope(Value value) { return valueMapper.count(value); };

  void resetInitializedInput() { initialized_input = false; }
  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Map from value to name of Python variable that contains the name.
  ValueMapper valueMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<llvm::StringMap<int64_t>> valueInScopeCount;

  /// Whether the input has been initialized.
  bool initialized_input = false;
};
} // namespace

/// Determine whether op result should be emitted in a deferred way.
static bool hasDeferredEmission(Operation *op) {
  return isa_and_nonnull<LiteralOp>(op);
}

StringRef PythonEmitter::getOrCreateName(Value value, std::string name) {
  if (!valueMapper.count(value)) {
    if (!initialized_input) {
      initialized_input = true;
      valueMapper.insert(value, name);
    } else {
      valueMapper.insert(
          value, formatv("{0}_{1}", name, ++valueInScopeCount.top()[name]));
    }
  }
  return *valueMapper.begin(value);
}

std::string PythonEmitter::getSubscriptName(SubscriptOp op) {
  std::string name;
  llvm::raw_string_ostream ss(name);
  auto index = op.getIndex();
  ss << "[" << getOrCreateName(index, "subscript") << "]";
  return name;
}

void PythonEmitter::cacheDeferredOpResult(Value value, StringRef str) {
  if (!valueMapper.count(value)) {
    valueMapper.insert(value, str.str());
  }
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    CallOpaqueOp callOpaqueOp) {
  raw_indented_ostream &os = emitter.ostream();
  Operation &op = *callOpaqueOp.getOperation();
  StringRef callee = callOpaqueOp.getCallee();
  if (failed(emitter.emitAssignPrefix(op))) {
    return failure();
  }

  auto emitArgs = [&](const auto &pair) -> LogicalResult {
    auto [attr, keywordArg] = pair;
    auto keywordArgStr = mlir::cast<StringAttr>(keywordArg).getValue();
    keywordArgStr != "" ? os << keywordArgStr << "=" : os << keywordArgStr;

    if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
      if (iAttr.getType().isIndex()) {
        int64_t idx = iAttr.getInt();
        Value operand = op.getOperand(idx);
        if (!emitter.hasValueInScope(operand)) {
          return op.emitOpError("operand ")
                 << idx << "'s value not defined in scope";
        }
        if (failed(emitter.emitOperand(operand, "print_arg"))) {
          return failure();
        }
        return success();
      }
    }
    if (failed(emitter.emitAttribute(op.getLoc(), attr))) {
      return failure();
    }
    return success();
  };

  if (callee == "util_create_list") {
    os << "[";
    if (failed(emitter.emitOperands(op))) {
      return failure();
    }
    os << "]";
  } else {
    os << callee << "(";
    LogicalResult emittedArgs = success();
    if (callOpaqueOp.getArgs()) {
      auto args =
          llvm::zip(*callOpaqueOp.getArgs(), *callOpaqueOp.getKeywordArgs());
      emittedArgs = interleaveCommaWithError(args, os, emitArgs);
    } else {
      emittedArgs = emitter.emitOperands(op);
    }
    if (failed(emittedArgs)) {
      return failure();
    }
    os << ")";
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, ImportOp importOp) {
  raw_indented_ostream &os = emitter.ostream();
  if (importOp.getImportAll()) {
    os << "from " << importOp.getModuleName() << " import *";
  } else if (importOp.getMembersToImport()) {
    os << "from " << importOp.getModuleName() << " import ";

    auto members = *importOp.getMembersToImport();
    os << mlir::cast<StringAttr>(members[0]).getValue();
    auto member_aliases = *importOp.getMemberAliases();
    auto member_alias = mlir::cast<StringAttr>(member_aliases[0]).getValue();
    if (!member_alias.empty()) {
      os << " as " << member_alias;
    }

    for (size_t i = 1; i < members.size(); i++) {
      os << ", " << mlir::cast<StringAttr>(members[i]).getValue();
      member_alias = mlir::cast<StringAttr>(member_aliases[i]).getValue();
      if (!member_alias.empty()) {
        os << " as " << member_alias;
      }
    }
  } else {
    os << "import " << importOp.getModuleName();
    if (importOp.getModuleAlias()) {
      os << " as " << importOp.getModuleAlias();
    }
  }
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    func::CallOp callOp) {
  Operation &op = *callOp.getOperation();
  StringRef callee = callOp.getCallee();
  raw_indented_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(op))) {
    return failure();
  }
  os << callee;
  os << "(";
  if (failed(emitter.emitOperands(op))) {
    return failure();
  }
  os << ")";
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, ModuleOp moduleOp) {
  PythonEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult printFunctionArgs(PythonEmitter &emitter, Operation &op,
                                       Region::BlockArgListType arguments) {
  raw_indented_ostream &os = emitter.ostream();
  emitter.resetInitializedInput();
  return interleaveCommaWithError(arguments, os, [&](BlockArgument arg) {
    return emitter.emitOperand(arg, "inputs");
  });
}

static LogicalResult printFunctionBody(PythonEmitter &emitter, Operation &op,
                                       Region::BlockListType &blocks) {
  raw_indented_ostream &os = emitter.ostream();
  os.indent();
  if (blocks.size() > 1) {
    return op.emitOpError(
        "PythonEmitter's printFunctionBody currently only supports "
        "single-block functions directly "
        "or relies on prior passes to structure control flow.");
  }

  for (Operation &op : blocks.front().getOperations()) {
    if (failed(emitter.emitOperation(op))) {
      return failure();
    }
  }
  os.unindent();
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    func::FuncOp functionOp) {
  PythonEmitter::Scope scope(emitter);
  Operation &op = *functionOp.getOperation();
  raw_indented_ostream &os = emitter.ostream();
  StringRef callee = functionOp.getName();
  os << "def";
  os << " " << callee;
  os << "(";
  if (failed(printFunctionArgs(emitter, op, functionOp.getArguments()))) {
    return failure();
  }
  os << "): \n";
  if (failed(printFunctionBody(emitter, op, functionOp.getBlocks()))) {
    return failure();
  }

  if (callee == "main") {
    os << "\n";
    os << "if __name__ == \'__main__\':\n";
    os << "  main()\n";
  }
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << "return";
  switch (returnOp.getNumOperands()) {
  case 0:
    break;
  case 1: {
    os << " ";
    if (failed(emitter.emitOperand(returnOp.getOperand(0), "return_arg"))) {
      return failure();
    }
    break;
  }
  default: {
    os << " ";
    if (failed(emitter.emitOperands(*returnOp.getOperation()))) {
      return failure();
    }
  }
  }
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    SubscriptOp subscriptOp) {
  if (failed(emitter.emitAssignPrefix(*subscriptOp))) {
    return failure();
  }

  raw_indented_ostream &os = emitter.ostream();
  if (failed(emitter.emitOperand(subscriptOp.getOperand(0), "subscript_arg"))) {
    return failure();
  }

  os << emitter.getSubscriptName(subscriptOp);

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, AssignOp assignOp) {
  if (failed(emitter.emitAssignPrefix(*assignOp))) {
    return failure();
  }

  return emitter.emitOperands(*assignOp);
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    ConstantOp constantOp) {
  Attribute value = constantOp.getValue();

  if (failed(emitter.emitAssignPrefix(*constantOp))) {
    return failure();
  }

  return emitter.emitAttribute(constantOp->getLoc(), value);
}

LogicalResult PythonEmitter::emitOperation(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // EmitPy ops.
          .Case<CallOpaqueOp, ImportOp, AssignOp, ConstantOp, SubscriptOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<LiteralOp>([&](auto op) {
            cacheDeferredOpResult(op.getResult(), op.getValue());
            return success();
          })
          // Func ops.
          .Case<func::CallOp, func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    return failure();
  }

  if (hasDeferredEmission(&op)) {
    return success();
  }

  os << "\n";

  return success();
}

LogicalResult PythonEmitter::emitOperand(Value value, std::string name) {
  os << getOrCreateName(value, name);

  return success();
}

LogicalResult PythonEmitter::emitOperands(Operation &op) {
  return interleaveCommaWithError(op.getOperands(), os, [&](Value value) {
    if (failed(emitOperand(value, "op_arg"))) {
      return failure();
    }

    return success();
  });
}

LogicalResult PythonEmitter::emitAttribute(Location loc, Attribute attr) {
  auto printInt = [&](APInt attr) {
    SmallString<128> strValue;
    attr.toString(strValue, 10, true);
    return strValue;
  };
  // Print integer attributes.
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    os << printInt(iAttr.getValue());
    return success();
  }

  // Print opaque attributes.
  if (auto oAttr = dyn_cast<mlir::tt::emitpy::OpaqueAttr>(attr)) {
    os << oAttr.getValue();
    return success();
  }

  return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult PythonEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (failed(emitVariableAssignment(result, op))) {
      return failure();
    }
    break;
  }
  default: {
    interleaveComma(op.getResults(), os, [&](Value result) {
      os << getOrCreateName(result, "prefix");
    });
    os << " = ";
  }
  }
  return success();
}

bool isValidPythonIdentifier(const std::string &name) {
  static const std::regex pattern(R"(^[A-Za-z_][A-Za-z0-9_]*$)");
  static const std::unordered_set<std::string> keywords = {
      "False",  "None",   "True",    "and",      "as",       "assert", "async",
      "await",  "break",  "class",   "continue", "def",      "del",    "elif",
      "else",   "except", "finally", "for",      "from",     "global", "if",
      "import", "in",     "is",      "lambda",   "nonlocal", "not",    "or",
      "pass",   "raise",  "return",  "try",      "while",    "with",   "yield"};

  return std::regex_match(name, pattern) &&
         (keywords.find(name) == keywords.end());
}

LogicalResult PythonEmitter::emitVariableAssignment(OpResult result,
                                                    Operation &op) {
  std::string name = "var";

  std::string ssaName;
  llvm::raw_string_ostream stream(ssaName);
  mlir::OpPrintingFlags flags;
  op.getResult(0).printAsOperand(stream, flags);
  stream.flush();
  size_t dotPos = ssaName.find(".result");
  if (dotPos != std::string::npos) {
    name = ssaName.substr(1, dotPos - 1);
  } else {
    name = ssaName.substr(1);
  }
  if (!isValidPythonIdentifier(name)) {
    name = "var";
  }

  if (auto calleeAttr = op.getAttr("callee")) {
    std::string calleeName;
    llvm::raw_string_ostream stream(calleeName);
    calleeAttr.print(stream, false);
    stream.flush();
    calleeName = calleeName.substr(1);
    if (name == "var" && isValidPythonIdentifier(calleeName)) {
      name = calleeName;
    }
  }

  if (op.getNumOperands() > 0 && name == "var") {
    name = getOrCreateName(op.getOperand(0), name).str();
  }
  if (!isValidPythonIdentifier(name)) {
    name = "var";
  }
  os << getOrCreateName(result, name) << " = ";
  return success();
}

LogicalResult mlir::tt::emitpy::translateToPython(Operation *op,
                                                  raw_ostream &os) {
  PythonEmitter emitter(os);
  return emitter.emitOperation(*op);
}
