// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/Python/PythonEmitter.h"

#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <regex>
#include <stack>
#include <string>
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

// Forward declaration
struct PythonEmitter;

/// Helper class for building inline Python expressions
class ExpressionBuilder {
public:
  ExpressionBuilder(PythonEmitter &emitter) {}

  /// Register a block argument with its corresponding name
  void mapBlockArgument(Value blockArg, StringRef name) {
    blockArgNames[blockArg] = name.str();
  }

  /// Build an expression string for a value recursively
  FailureOr<std::string> buildExpression(Value value);

private:
  /// Build expression for specific operation types
  FailureOr<std::string> buildCallOpaqueExpr(CallOpaqueOp op);
  FailureOr<std::string> buildLiteralExpr(LiteralOp op);
  FailureOr<std::string> buildSubscriptExpr(SubscriptOp op);
  FailureOr<std::string> buildGetAttrExpr(GetAttrOp op);

  /// Map from Value to its expression string representation
  DenseMap<Value, std::string> expressionCache;

  /// Block arguments mapped to their names
  DenseMap<Value, std::string> blockArgNames;
};

/// Emitter that uses dialect specific emitters to emit Python code.
struct PythonEmitter {
  explicit PythonEmitter(raw_ostream &os, std::string &fileId)
      : os(os), fileId(fileId) {
    valueInScopeCount.push(0);
    usedNames.push(std::set<std::string>());
  }

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' or returns failure.
  LogicalResult emitOperation(Operation &op);

  /// TODO: implement this for type hints
  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type);

  /// Emits an assignment for a variable or
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

  /// Emits a global statement.
  LogicalResult emitGlobalStatement(GlobalStatementOp globalStatementOp);

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Emits value as an operand of an operation.
  LogicalResult emitOperand(Value value, std::string name);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value value, std::string name);

  /// Return the textual representation of a subscript operation.
  std::string getSubscriptName(SubscriptOp op);

  /// Register a value with a name so it can be referenced later.
  void registerDeferredValue(Value value, StringRef str);

  /// Decides whether the file should be emitted. If fileId is set, only
  /// the ops of the file with the matching id are emitted.
  bool shouldEmitFile(FileOp fileOp);

  /// Returns true if we're emitting multiple files (fileId is empty).
  bool isEmittingMultipleFiles() const { return fileId.empty(); }

  /// Emits a comment label for the file to separate outputs.
  void emitFileLabel(FileOp fileOp);

  /// RAII helper function to manage entering/exiting Python scopes.
  struct Scope {
    Scope(PythonEmitter &emitter)
        : valueMapperScope(emitter.valueMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.usedNames.push(emitter.usedNames.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.usedNames.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    PythonEmitter &emitter;
  };

  /// Returns whether the Value is assigned to a Python variable in the scope.
  bool hasValueInScope(Value value) { return valueMapper.count(value); };

  bool isInClassScope() const { return classDepth > 0; }

  /// RAII helper function to manage entering/exiting Python class scopes.
  struct ClassScope {
    ClassScope(PythonEmitter &emitter) : emitter(&emitter) {
      emitter.classDepth++;
    }
    ~ClassScope() { emitter->classDepth--; }

  private:
    // Non-owning pointer: ClassScope is a stack guard created inside
    // printOperation(PythonEmitter&, ClassOp) and destroyed before that call
    // returns. The PythonEmitter object is owned by translateToPython and
    // outlives all nested ClassScope instances, so this pointer cannot dangle.
    PythonEmitter *emitter;
  };

  /// Reserve a name to prevent collisions.
  void reserveName(const std::string &name) { usedNames.top().insert(name); }

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Map from value to name of Python variable that contains the name.
  ValueMapper valueMapper;

  /// The number of values in the current scope per variable name. This is used
  /// to declare the names of values in a scope.
  std::stack<int64_t> valueInScopeCount;

  /// The set of names used in the current scope.
  std::stack<std::set<std::string>> usedNames;

  int classDepth = 0;

  /// The id of the current file. Only files with this id are emitted. If empty,
  /// all files are emitted as one.
  std::string fileId;
};
} // namespace

// Implementation of ExpressionBuilder methods
FailureOr<std::string> ExpressionBuilder::buildExpression(Value value) {
  // Check if already cached
  if (expressionCache.count(value)) {
    return expressionCache[value];
  }

  // Check if it's a block argument
  if (blockArgNames.count(value)) {
    return blockArgNames[value];
  }

  // Otherwise, it must be defined by an operation
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return failure();
  }

  // Build expression based on operation type
  FailureOr<std::string> result;
  if (auto callOp = dyn_cast<CallOpaqueOp>(defOp)) {
    result = buildCallOpaqueExpr(callOp);
  } else if (auto literalOp = dyn_cast<LiteralOp>(defOp)) {
    result = buildLiteralExpr(literalOp);
  } else if (auto subscriptOp = dyn_cast<SubscriptOp>(defOp)) {
    result = buildSubscriptExpr(subscriptOp);
  } else if (auto getAttrOp = dyn_cast<GetAttrOp>(defOp)) {
    result = buildGetAttrExpr(getAttrOp);
  } else {
    return defOp->emitOpError(
        "operation type not supported in inline expression");
  }

  if (succeeded(result)) {
    expressionCache[value] = *result;
  }
  return result;
}

FailureOr<std::string> ExpressionBuilder::buildCallOpaqueExpr(CallOpaqueOp op) {
  std::string expr;
  llvm::raw_string_ostream os(expr);

  os << op.getCallee() << "(";

  bool first = true;
  for (Value operand : op.getOperands()) {
    if (!first) {
      os << ", ";
    }
    first = false;

    auto operandExpr = buildExpression(operand);
    if (failed(operandExpr)) {
      return failure();
    }
    os << *operandExpr;
  }

  os << ")";
  return expr;
}

FailureOr<std::string> ExpressionBuilder::buildLiteralExpr(LiteralOp op) {
  return std::string(op.getValue());
}

FailureOr<std::string> ExpressionBuilder::buildSubscriptExpr(SubscriptOp op) {
  std::string expr;
  llvm::raw_string_ostream os(expr);

  auto valueExpr = buildExpression(op.getValue());
  if (failed(valueExpr)) {
    return failure();
  }

  auto indexExpr = buildExpression(op.getIndex());
  if (failed(indexExpr)) {
    return failure();
  }

  os << *valueExpr << "[" << *indexExpr << "]";
  return expr;
}

FailureOr<std::string> ExpressionBuilder::buildGetAttrExpr(GetAttrOp op) {
  std::string expr;
  llvm::raw_string_ostream os(expr);

  auto objectExpr = buildExpression(op.getObject());
  if (failed(objectExpr)) {
    return failure();
  }

  os << *objectExpr << "." << op.getAttrName();
  return expr;
}

/// Determine whether op result should be emitted in a deferred way.
static bool hasDeferredEmission(Operation *op) {
  // ExpressionOp with inline mode should also be deferred
  if (auto exprOp = dyn_cast_or_null<ExpressionOp>(op)) {
    return !exprOp.getDoNotInline();
  }
  return isa_and_nonnull<LiteralOp>(op);
}

StringRef PythonEmitter::getOrCreateName(Value value, std::string name) {
  if (!valueMapper.count(value)) {
    while (usedNames.top().count(name)) {
      name = name + "_" + std::to_string(valueInScopeCount.top()++);
    }
    valueMapper.insert(value, name);
    usedNames.top().insert(name);
  }
  return *valueMapper.begin(value);
}

std::string PythonEmitter::getSubscriptName(SubscriptOp op) {
  std::string name;
  llvm::raw_string_ostream ss(name);
  auto index = op.getIndex();
  std::string indexName = "index_" + std::to_string(valueInScopeCount.top()++);
  ss << "[" << getOrCreateName(index, indexName) << "]";
  return name;
}

void PythonEmitter::registerDeferredValue(Value value, StringRef str) {
  if (!valueMapper.count(value)) {
    valueMapper.insert(value, str.str());
  }
}

bool PythonEmitter::shouldEmitFile(FileOp fileOp) {
  return fileId == fileOp.getId() || isEmittingMultipleFiles();
}

void PythonEmitter::emitFileLabel(FileOp fileOp) {
  os << "# File: " << fileOp.getId() << "\n";
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
  func::FuncOp functionOp = mlir::cast<func::FuncOp>(op);

  return interleaveCommaWithError(arguments, os, [&](BlockArgument arg) {
    std::string argName = "inputs";
    if (auto suggestNameAttr =
            functionOp.getArgAttr(arg.getArgNumber(), "emitpy.name")) {
      argName = mlir::cast<StringAttr>(suggestNameAttr).getValue();
    } else if (emitter.isInClassScope() && arg.getArgNumber() == 0) {
      argName = "self";
    }
    return emitter.emitOperand(arg, argName);
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
  emitter.reserveName(callee.str());
  StringRef methodKind = "";
  if (auto methodKindAttr =
          functionOp->getAttrOfType<StringAttr>("emitpy.method_kind")) {
    methodKind = methodKindAttr.getValue();
    if (!emitter.isInClassScope()) {
      return functionOp.emitOpError(
          "emitpy.method_kind is only valid inside a class");
    }
    if (methodKind == "classmethod") {
      os << "@classmethod\n";
    } else if (methodKind == "staticmethod") {
      os << "@staticmethod\n";
    }
  }
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

  if (!emitter.isInClassScope() && callee == "main") {
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
                                    GetAttrOp getAttrOp) {
  if (failed(emitter.emitAssignPrefix(*getAttrOp))) {
    return failure();
  }

  raw_indented_ostream &os = emitter.ostream();
  if (failed(emitter.emitOperand(getAttrOp.getObject(), "object"))) {
    return failure();
  }

  os << "." << getAttrOp.getAttrName();
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    SetAttrOp setAttrOp) {
  raw_indented_ostream &os = emitter.ostream();
  if (failed(emitter.emitOperand(setAttrOp.getObject(), "object"))) {
    return failure();
  }

  os << "." << setAttrOp.getAttrName() << " = ";
  if (failed(emitter.emitOperand(setAttrOp.getValue(), "value"))) {
    return failure();
  }
  return success();
}

static LogicalResult emitClassBase(Attribute baseAttr,
                                   raw_indented_ostream &os) {
  if (auto symbolRef = dyn_cast<SymbolRefAttr>(baseAttr)) {
    os << symbolRef.getLeafReference();
    return success();
  }
  if (auto opaque = dyn_cast<mlir::tt::emitpy::OpaqueAttr>(baseAttr)) {
    os << opaque.getValue();
    return success();
  }
  return failure();
}

static LogicalResult printOperation(PythonEmitter &emitter, ClassOp classOp) {
  PythonEmitter::Scope scope(emitter);
  PythonEmitter::ClassScope classScope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  os << "class " << classOp.getSymName();
  if (auto baseClasses = classOp.getBaseClasses()) {
    if (!baseClasses->empty()) {
      os << "(";
      if (failed(interleaveCommaWithError(*baseClasses, os,
                                          [&](Attribute baseAttr) {
                                            return emitClassBase(baseAttr, os);
                                          }))) {
        return classOp.emitOpError("invalid class base attribute");
      }
      os << ")";
    }
  }
  os << ":\n";

  os.indent();
  Block &body = classOp.getBody().front();
  if (body.empty()) {
    os << "pass\n";
    os.unindent();
    return success();
  }

  for (Operation &op : body.getOperations()) {
    if (failed(emitter.emitOperation(op))) {
      return failure();
    }
  }
  os.unindent();
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    ConstantOp constantOp) {
  Attribute value = constantOp.getValue();

  if (failed(emitter.emitAssignPrefix(*constantOp))) {
    return failure();
  }

  return emitter.emitAttribute(constantOp->getLoc(), value);
}

static LogicalResult printOperation(PythonEmitter &emitter, GlobalOp globalOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << globalOp.getSymName() << " = ";
  if (failed(emitter.emitAttribute(globalOp->getLoc(),
                                   globalOp.getInitialValue()))) {
    return failure();
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    AssignGlobalOp assignGlobalOp) {
  raw_indented_ostream &os = emitter.ostream();
  os << assignGlobalOp.getName() << " = ";
  if (failed(emitter.emitOperand(assignGlobalOp.getValue(),
                                 assignGlobalOp.getName().str()))) {
    return failure();
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    GlobalStatementOp globalStatementOp) {
  emitter.registerDeferredValue(globalStatementOp.getResult(),
                                globalStatementOp.getName());

  return emitter.emitGlobalStatement(globalStatementOp);
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    CreateDictOp createDictOp) {
  raw_indented_ostream &os = emitter.ostream();
  StringRef dictName = createDictOp.getDictName();

  emitter.registerDeferredValue(createDictOp.getResult(), dictName);

  os << dictName << " = ";

  if (createDictOp.getLiteralExpr()) {
    os << *createDictOp.getLiteralExpr();
    return success();
  }

  os << "{";
  auto items = createDictOp.getItems();
  for (size_t i = 0; i < items.size(); i += 2) {
    if (i > 0) {
      os << ", ";
    }
    if (failed(emitter.emitOperand(items[i], "dict_key"))) {
      return failure();
    }
    os << ": ";
    if (failed(emitter.emitOperand(items[i + 1], "dict_value"))) {
      return failure();
    }
  }
  os << "}";

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    SetValueForDictKeyOp op) {
  raw_indented_ostream &os = emitter.ostream();

  if (failed(emitter.emitOperand(op.getDict(), "dict_arg"))) {
    return failure();
  }

  os << "[";
  if (failed(emitter.emitOperand(op.getKey(), "dict_key"))) {
    return failure();
  }
  os << "] = ";

  if (failed(emitter.emitOperand(op.getValue(), "dict_value"))) {
    return failure();
  }
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    GetValueForDictKeyOp op) {
  if (failed(emitter.emitAssignPrefix(*op))) {
    return failure();
  }

  raw_indented_ostream &os = emitter.ostream();

  if (failed(emitter.emitOperand(op.getDict(), "dict_arg"))) {
    return failure();
  }

  os << "[";
  if (failed(emitter.emitOperand(op.getKey(), "dict_key"))) {
    return failure();
  }
  os << "]";

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, YieldOp yieldOp) {
  // YieldOp should only appear inside ExpressionOp
  // In the new design, yield should never be directly emitted
  // It's handled by the ExpressionBuilder
  return yieldOp.emitOpError("yield operation should not be directly emitted");
}

// Helper function to build an expression string
static FailureOr<std::string> buildExpressionString(ExpressionOp expressionOp,
                                                    PythonEmitter &emitter) {
  ExpressionBuilder builder(emitter);

  // Map block arguments to operands
  Block *bodyBlock = expressionOp.getBodyBlock();
  for (auto [blockArg, operand] :
       llvm::zip(bodyBlock->getArguments(), expressionOp.getOperands())) {
    // Get the name from the parent emitter and register it in the builder
    std::string argName = emitter.getOrCreateName(operand, "expr_arg").str();
    builder.mapBlockArgument(blockArg, argName);
  }

  // Find the yield op and build its expression
  auto yieldOp = cast<YieldOp>(bodyBlock->getTerminator());
  Value yieldValue = yieldOp.getResult();

  // Build the expression recursively
  return builder.buildExpression(yieldValue);
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    ExpressionOp expressionOp) {

  // Check if we should emit inline or not
  bool shouldInline = !expressionOp.getDoNotInline();

  if (shouldInline) {
    // For inline expressions, build the expression string and cache it
    auto exprStrResult = buildExpressionString(expressionOp, emitter);
    if (failed(exprStrResult)) {
      return failure();
    }

    // Cache the expression for deferred emission
    emitter.registerDeferredValue(expressionOp.getResult(), *exprStrResult);

    // Return success - the expression will be emitted when it's used
    return success();
  } else {
    // Emit non-inline: emit the body operations normally
    PythonEmitter::Scope scope(emitter);

    // Map block arguments to operands
    Block *bodyBlock = expressionOp.getBodyBlock();
    for (auto [blockArg, operand] :
         llvm::zip(bodyBlock->getArguments(), expressionOp.getOperands())) {
      emitter.registerDeferredValue(
          blockArg, emitter.getOrCreateName(operand, "expr_arg"));
    }

    // Emit all operations in the body except the yield
    for (Operation &bodyOp : bodyBlock->getOperations()) {
      if (isa<YieldOp>(bodyOp)) {
        // For the yield, just emit an assignment to the expression result
        auto yieldOp = cast<YieldOp>(bodyOp);
        if (failed(emitter.emitAssignPrefix(*expressionOp))) {
          return failure();
        }
        return emitter.emitOperand(yieldOp.getResult(), "");
      }
      if (failed(emitter.emitOperation(bodyOp))) {
        return failure();
      }
    }
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, FileOp fileOp) {
  if (!emitter.shouldEmitFile(fileOp)) {
    return success();
  }

  // Emit file label only when emitting multiple files together.
  if (emitter.isEmittingMultipleFiles()) {
    emitter.emitFileLabel(fileOp);
  }

  for (Operation &op : fileOp.getRegion().getOps()) {
    if (failed(emitter.emitOperation(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult PythonEmitter::emitOperation(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // EmitPy ops.
          .Case<CallOpaqueOp, ImportOp, AssignOp, GetAttrOp, SetAttrOp,
                ConstantOp, SubscriptOp, ClassOp, GlobalOp, AssignGlobalOp,
                GlobalStatementOp, CreateDictOp, SetValueForDictKeyOp,
                GetValueForDictKeyOp, ExpressionOp, YieldOp, FileOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<LiteralOp>([&](auto op) {
            registerDeferredValue(op.getResult(), op.getValue());
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
  // Check if this value is from an inline expression that has been cached
  if (auto *defOp = value.getDefiningOp()) {
    if (auto exprOp = dyn_cast<ExpressionOp>(defOp)) {
      if (!exprOp.getDoNotInline() && valueMapper.count(value)) {
        // Emit the cached expression string directly
        os << valueMapper.lookup(value);
        return success();
      }
    }
  }

  // Default behavior: emit the variable name
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
  // Print string attributes.
  if (auto sAttr = dyn_cast<StringAttr>(attr)) {
    os << "\"" << sAttr.getValue() << "\"";
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
      std::string name = "v_" + std::to_string(valueInScopeCount.top()++);
      os << getOrCreateName(result, name);
    });
    os << " = ";
  }
  }
  return success();
}

LogicalResult
PythonEmitter::emitGlobalStatement(GlobalStatementOp globalStatementOp) {
  os << "global " << globalStatementOp.getName();

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

std::string validateVariableName(const std::string &name) {
  std::string result;

  // Replace illegal characters with underscores
  for (char c : name) {
    if (std::isalnum(c) || c == '_') {
      result += c;
    } else {
      result += '_';
    }
  }

  if (!result.empty() && std::isdigit(result[0])) {
    result = "var_" + result;
  }

  if (!isValidPythonIdentifier(result)) {
    result = "";
  }

  return result;
}

// Assign a variable name to the result of the operation.
LogicalResult PythonEmitter::emitVariableAssignment(OpResult result,
                                                    Operation &op) {
  std::string name = "";

  if (auto suggestNameAttr = op.getAttr("emitpy.name")) {
    name = mlir::cast<StringAttr>(suggestNameAttr).getValue();
    name = validateVariableName(name);
  }
  if (name.empty()) {
    name = "var_" + std::to_string(valueInScopeCount.top()++);
  }
  os << getOrCreateName(result, name) << " = ";
  return success();
}

LogicalResult mlir::tt::emitpy::translateToPython(Operation *op,
                                                  raw_ostream &os,
                                                  std::string &fileId) {
  PythonEmitter emitter(os, fileId);
  return emitter.emitOperation(*op);
}
