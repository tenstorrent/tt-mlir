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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

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

/// Return the precedence of a operator as an integer, higher values
/// imply higher precedence.
static FailureOr<int> getOperatorPrecedence(Operation *operation) {
  return llvm::TypeSwitch<Operation *, FailureOr<int>>(operation)
      .Case<CallOpaqueOp>([&](auto op) { return 3; })
      .Case<SubscriptOp>([&](auto op) { return 2; })
      .Case<LiteralOp>([&](auto op) { return 4; })
      .Default([](auto op) { return op->emitError("unsupported operation"); });
}

namespace {
/// Emitter that uses dialect specific emitters to emit Python code.
struct PythonEmitter {
  explicit PythonEmitter(raw_ostream &os) : os(os) {
    valueInScopeCount.push(0);
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
  LogicalResult emitVariableAssignment(OpResult result);

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
  LogicalResult emitOperand(Value value);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value value);

  // Return the textual representation of a subscript operation.
  std::string getSubscriptName(SubscriptOp op);

  /// Emit an expression as a Py expression.
  LogicalResult emitExpression(ExpressionOp expressionOp);

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

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

  /// Get expression currently being emitted.
  ExpressionOp getEmittedExpression() { return emittedExpression; }

  /// Determine whether given value is part of the expression potentially being
  /// emitted.
  bool isPartOfCurrentExpression(Value value) {
    if (!emittedExpression) {
      return false;
    }
    Operation *def = value.getDefiningOp();
    if (!def) {
      return false;
    }
    auto operandExpression = dyn_cast<ExpressionOp>(def->getParentOp());
    return operandExpression == emittedExpression;
  };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Map from value to name of Python variable that contains the name.
  ValueMapper valueMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;

  /// State of the current expression being emitted.
  ExpressionOp emittedExpression;
  SmallVector<int> emittedExpressionPrecedence;

  void pushExpressionPrecedence(int precedence) {
    emittedExpressionPrecedence.push_back(precedence);
  }

  void popExpressionPrecedence() { emittedExpressionPrecedence.pop_back(); }

  static int lowestPrecedence() { return 0; }

  int getExpressionPrecedence() {
    if (emittedExpressionPrecedence.empty()) {
      return lowestPrecedence();
    }
    return emittedExpressionPrecedence.back();
  }
};
} // namespace

/// Determine whether op result should be emitted in a deferred way.
static bool hasDeferredEmission(Operation *op) {
  return isa_and_nonnull<LiteralOp>(op);
}

/// Determine whether expression expressionOp should be emitted inline, i.e.
/// as part of its user. This function recommends inlining of any expressions
/// that can be inlined unless it is used by another expression, under the
/// assumption that  any expression fusion/re-materialization was taken care of
/// by transformations run by the backend.
static bool shouldBeInlined(ExpressionOp expressionOp) {
  // Do not inline if expression is marked as such.
  if (expressionOp.getDoNotInline()) {
    return false;
  }

  // Do not inline expressions with side effects to prevent side-effect
  // reordering.
  /*  if (expressionOp.hasSideEffects()) {
     return false;
   } */

  // Do not inline expressions with multiple uses to prevent redundant
  // calculations.
  Value result = expressionOp.getResult();
  if (!result.hasOneUse()) {
    return false;
  }

  Operation *user = *result.getUsers().begin();

  // Do not inline expressions used by operations with deferred emission, since
  // their translation requires the materialization of variables.
  if (hasDeferredEmission(user)) {
    return false;
  }

  // Do not inline expressions used by ops with the PyExpressionInterface. If
  // this was intended, the user could have been merged into the expression op.
  return !isa<PyExpressionInterface>(*user);
}

StringRef PythonEmitter::getOrCreateName(Value value) {
  if (!valueMapper.count(value)) {
    valueMapper.insert(value, formatv("v{0}", ++valueInScopeCount.top()));
  }
  return *valueMapper.begin(value);
}

std::string PythonEmitter::getSubscriptName(SubscriptOp op) {
  std::string name;
  llvm::raw_string_ostream ss(name);
  auto index = op.getIndex();
  ss << "[" << getOrCreateName(index) << "]";
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
    os << keywordArgStr;
    if (keywordArgStr != "") {
      os << "=";
    }
    if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
      if (iAttr.getType().isIndex()) {
        int64_t idx = iAttr.getInt();
        Value operand = op.getOperand(idx);
        if (!emitter.hasValueInScope(operand)) {
          return op.emitOpError("operand ")
                 << idx << "'s value not defined in scope";
        }
        if (failed(emitter.emitOperand(operand))) {
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
    auto args =
        llvm::zip(*callOpaqueOp.getArgs(), *callOpaqueOp.getKeywordArgs());
    LogicalResult emittedArgs =
        callOpaqueOp.getArgs() ? interleaveCommaWithError(args, os, emitArgs)
                               : emitter.emitOperands(op);
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
  return interleaveCommaWithError(arguments, os, [&](BlockArgument arg) {
    return emitter.emitOperand(arg);
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
    if (failed(emitter.emitOperand(returnOp.getOperand(0)))) {
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
  if (failed(emitter.emitOperand(subscriptOp.getOperand(0)))) {
    return failure();
  }

  if (auto exprOp = dyn_cast<ExpressionOp>(subscriptOp->getParentOp())) {
    if (failed(emitter.emitOperand(subscriptOp.getOperand(1)))) {
      return failure();
    }
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
                                    ExpressionOp expressionOp) {
  if (shouldBeInlined(expressionOp)) {
    return success();
  }

  Operation &op = *expressionOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op))) {
    return failure();
  }

  return emitter.emitExpression(expressionOp);
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
          .Case<CallOpaqueOp, ImportOp, AssignOp, ConstantOp, ExpressionOp,
                SubscriptOp>([&](auto op) { return printOperation(*this, op); })
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

  if (getEmittedExpression() ||
      (isa<ExpressionOp>(op) && shouldBeInlined(cast<ExpressionOp>(op)))) {
    return success();
  }

  os << "\n";

  return success();
}

LogicalResult PythonEmitter::emitOperand(Value value) {
  if (isPartOfCurrentExpression(value)) {
    Operation *def = value.getDefiningOp();
    assert(def && "Expected operand to be defined by an operation");
    FailureOr<int> precedence = getOperatorPrecedence(def);
    if (failed(precedence)) {
      return failure();
    }

    // Sub-expressions with equal or lower precedence need to be
    // parenthesized, as they might be evaluated in the wrong order
    // depending on the shape of the expression tree.
    bool encloseInParenthesis = precedence.value() <= getExpressionPrecedence();
    if (encloseInParenthesis) {
      os << "(";
    }
    pushExpressionPrecedence(precedence.value());

    if (failed(emitOperation(*def))) {
      return failure();
    }

    if (encloseInParenthesis) {
      os << ")";
    }

    popExpressionPrecedence();
    return success();
  }

  auto expressionOp = dyn_cast_if_present<ExpressionOp>(value.getDefiningOp());
  if (expressionOp && shouldBeInlined(expressionOp)) {
    return emitExpression(expressionOp);
  }
  os << getOrCreateName(value);
  return success();
}

LogicalResult PythonEmitter::emitOperands(Operation &op) {
  return interleaveCommaWithError(op.getOperands(), os, [&](Value value) {
    // If an expression is being emitted, push lowest
    // precedence as these
    // operands are either wrapped by parenthesis.
    if (getEmittedExpression()) {
      pushExpressionPrecedence(lowestPrecedence());
    }
    if (failed(emitOperand(value))) {
      return failure();
    }
    if (getEmittedExpression()) {
      popExpressionPrecedence();
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

LogicalResult PythonEmitter::emitExpression(ExpressionOp expressionOp) {
  assert(emittedExpressionPrecedence.empty() &&
         "Expected precedence stack to be empty");
  Operation *rootOp = expressionOp.getRootOp();

  emittedExpression = expressionOp;
  FailureOr<int> precedence = getOperatorPrecedence(rootOp);
  if (failed(precedence)) {
    return failure();
  }
  pushExpressionPrecedence(precedence.value());

  if (failed(emitOperation(*rootOp))) {
    return failure();
  }

  popExpressionPrecedence();
  assert(emittedExpressionPrecedence.empty() &&
         "Expected precedence stack to be empty");
  emittedExpression = nullptr;

  return success();
}

LogicalResult PythonEmitter::emitAssignPrefix(Operation &op) {
  // If op is being emitted as part of an expression, bail out.
  if (getEmittedExpression()) {
    return success();
  }

  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (failed(emitVariableAssignment(result))) {
      return failure();
    }
    break;
  }
  default: {
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << " = ";
  }
  }
  return success();
}

LogicalResult PythonEmitter::emitVariableAssignment(OpResult result) {
  os << getOrCreateName(result) << " = ";
  return success();
}

LogicalResult mlir::tt::emitpy::translateToPython(Operation *op,
                                                  raw_ostream &os) {
  PythonEmitter emitter(os);
  return emitter.emitOperation(*op);
}
