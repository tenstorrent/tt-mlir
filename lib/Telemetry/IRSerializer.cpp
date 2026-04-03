// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Telemetry/IRSerializer.h"
#include "ttmlir/Telemetry/GraphTypes.h"
#include "ttmlir/Telemetry/TelemetryVisitor.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>

using namespace mlir;
using namespace mlir::tt::telemetry;

namespace {

std::string getLocationString(Operation *op) {
  std::string locStr;
  llvm::raw_string_ostream os(locStr);
  op->getLoc().print(os);
  return locStr;
}

std::string attrToString(Attribute attr) {
  std::string str;
  llvm::raw_string_ostream os(str);
  attr.print(os);
  return str;
}

/// Bulk constant data (weights, resources) would blow up storage if printed in
/// full. Summarize element attributes instead of stringifying their contents.
/// Small structural attrs (DenseArrayAttr for shapes/permutations, etc.) are
/// not ElementsAttr and pass through untouched.
bool isBulkElements(Attribute attr) {
  return isa<DenseIntOrFPElementsAttr, DenseStringElementsAttr,
             DenseResourceElementsAttr, SparseElementsAttr>(attr);
}

std::string summarizeElements(Attribute attr) {
  auto elements = cast<ElementsAttr>(attr);
  std::string summary;
  llvm::raw_string_ostream os(summary);
  os << "<elided ";
  if (auto shaped = dyn_cast<ShapedType>(elements.getType())) {
    os << shaped.getNumElements() << "x";
    shaped.getElementType().print(os);
  }
  os << ">";
  return summary;
}

/// Serialize attributes of an op through the visitor chain, eliding bulk data.
void serializeAttrs(Operation *op, TelemetryVisitorRegistry &registry,
                    llvm::StringMap<std::string> &attrs) {
  for (auto namedAttr : op->getAttrs()) {
    StringRef name = namedAttr.getName();
    Attribute attr = namedAttr.getValue();
    if (isBulkElements(attr)) {
      attrs[name] = summarizeElements(attr);
      continue;
    }
    // Let visitors handle the attr. If all return true, stringify it.
    if (registry.visitAttr(name, attr, attrs)) {
      attrs[name] = attrToString(attr);
    }
  }
}

/// Mutable state threaded through the recursive walk.
struct SerializeContext {
  TelemetryVisitorRegistry &registry;
  SnapshotData &snapshot;
  llvm::DenseMap<Operation *, std::string> &opToId;
  llvm::DenseMap<Value, std::string> &valueToTensorId;
  // Resolved during the walk, materialized into edges afterwards (a callee or
  // referenced symbol may be defined after its use).
  llvm::SmallVector<std::pair<Operation *, Operation *>> &pendingCalls;
  llvm::SmallVector<std::tuple<Operation *, Operation *, std::string>>
      &pendingRefs;
  // Per-op printed-MLIR line/col, captured while printing the snapshot (null if
  // the MLIR was not printed for this snapshot).
  const mlir::AsmState::LocationMap *lineMap;
};

/// Populate the generic, interface-derived fields of an op.
void populateOpData(Operation *op, OpData &data) {
  data.opName = op->getName().getStringRef().str();
  data.dialect = op->getName().getDialectNamespace().str();
  data.location = getLocationString(op);
  data.isTerminator = op->hasTrait<OpTrait::IsTerminator>();

  if (auto symName =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
    data.symName = symName.getValue().str();
  }
  if (auto fn = dyn_cast<FunctionOpInterface>(op)) {
    data.isFunc = true;
    data.numArgs = static_cast<int>(fn.getNumArguments());
    data.numResults = static_cast<int>(fn.getNumResults());
  }
}

/// Record call edges and symbol references for later edge materialization.
/// Purely interface/attribute driven -- no op-name checks.
///
/// A call is resolved either via CallOpInterface, or via the conventional
/// `callee` symbol attribute used by call-like ops that do not implement the
/// interface (e.g. ttcore.load_cached, which invokes a cached const-eval
/// function). All other symbol-ref attributes become generic REFERENCES edges.
void collectSymbolUses(Operation *op, SerializeContext &ctx) {
  SymbolRefAttr calleeSym;
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    calleeSym = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
  }
  if (!calleeSym) {
    calleeSym = op->getAttrOfType<FlatSymbolRefAttr>("callee");
  }
  if (calleeSym) {
    if (Operation *callee =
            SymbolTable::lookupNearestSymbolFrom(op, calleeSym)) {
      ctx.pendingCalls.push_back({op, callee});
    }
  }

  for (NamedAttribute na : op->getAttrs()) {
    // Skip the callee symbol already modelled as a call edge.
    if (calleeSym && na.getName().getValue() == "callee") {
      continue;
    }
    if (auto sym = dyn_cast<SymbolRefAttr>(na.getValue())) {
      if (Operation *target = SymbolTable::lookupNearestSymbolFrom(op, sym)) {
        ctx.pendingRefs.push_back({op, target, na.getName().getValue().str()});
      }
    }
  }
}

void emitResultTensors(Operation *op, const std::string &opId,
                       SerializeContext &ctx) {
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    Value result = op->getResult(i);
    TensorData tensorData;
    tensorData.id = generateUUID();
    tensorData.producerId = opId;
    tensorData.producerType = "op";
    tensorData.resultIdx = static_cast<int>(i);
    ctx.registry.visitValue(result, tensorData);
    ctx.valueToTensorId[result] = tensorData.id;
    ctx.snapshot.tensors.push_back(std::move(tensorData));
  }
}

void emitOperandEdges(Operation *op, const std::string &opId,
                      SerializeContext &ctx) {
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    auto it = ctx.valueToTensorId.find(op->getOperand(i));
    if (it != ctx.valueToTensorId.end()) {
      EdgeData edge;
      edge.tensorId = it->second;
      edge.consumerId = opId;
      edge.operandIdx = static_cast<int>(i);
      ctx.snapshot.edges.push_back(std::move(edge));
    }
  }
}

/// Recursively serialize an op and everything nested under it. Returns the
/// assigned UUID, or empty if the op (and its subtree) was pruned by a visitor.
std::string serializeOp(Operation *op, const std::string &parentId,
                        int regionIdx, int blockIdx, int order,
                        SerializeContext &ctx) {
  OpData opData;
  opData.id = generateUUID();
  opData.parentId = parentId;
  opData.regionIdx = regionIdx;
  opData.blockIdx = blockIdx;
  opData.order = order;
  populateOpData(op, opData);
  if (ctx.lineMap) {
    auto it = ctx.lineMap->find(op);
    if (it != ctx.lineMap->end()) {
      opData.mlirLine = static_cast<int>(it->second.first);
    }
  }

  // A visitor may prune this op and its entire subtree (e.g. const-eval funcs).
  if (!ctx.registry.visitOp(op, opData)) {
    return "";
  }

  serializeAttrs(op, ctx.registry, opData.attrs);
  ctx.opToId[op] = opData.id;
  std::string opId = opData.id;
  ctx.snapshot.ops.push_back(std::move(opData));

  collectSymbolUses(op, ctx);
  emitResultTensors(op, opId, ctx);
  emitOperandEdges(op, opId, ctx);

  for (unsigned ri = 0; ri < op->getNumRegions(); ++ri) {
    Region &region = op->getRegion(ri);
    int bi = 0;
    for (Block &block : region) {
      // Block arguments become first-class values before their consumers.
      for (BlockArgument arg : block.getArguments()) {
        BlockArgData baData;
        baData.id = generateUUID();
        baData.parentId = opId;
        baData.regionIdx = static_cast<int>(ri);
        baData.blockIdx = bi;
        baData.argIdx = static_cast<int>(arg.getArgNumber());

        TensorData tensorData;
        tensorData.id = generateUUID();
        tensorData.producerId = baData.id;
        tensorData.producerType = "block_arg";
        tensorData.resultIdx = 0;
        ctx.registry.visitValue(arg, tensorData);
        ctx.valueToTensorId[arg] = tensorData.id;

        ctx.snapshot.blockArgs.push_back(std::move(baData));
        ctx.snapshot.tensors.push_back(std::move(tensorData));
      }

      int childOrder = 0;
      for (Operation &child : block) {
        serializeOp(&child, opId, static_cast<int>(ri), bi, childOrder++, ctx);
      }
      ++bi;
    }
  }

  return opId;
}

} // namespace

SnapshotData mlir::tt::telemetry::serialize(
    Operation *rootOp, TelemetryVisitorRegistry &registry,
    const std::string &tag, const std::string &passName, int snapshotIndex,
    const mlir::AsmState::LocationMap *lineMap) {
  SnapshotData snapshot;
  snapshot.snapshotId = generateUUID();
  snapshot.tag = tag;
  snapshot.passName = passName;
  snapshot.snapshotIndex = snapshotIndex;
  auto now = std::chrono::system_clock::now();
  snapshot.timestampUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             now.time_since_epoch())
                             .count();

  llvm::DenseMap<Operation *, std::string> opToId;
  llvm::DenseMap<Value, std::string> valueToTensorId;
  llvm::SmallVector<std::pair<Operation *, Operation *>> pendingCalls;
  llvm::SmallVector<std::tuple<Operation *, Operation *, std::string>>
      pendingRefs;
  SerializeContext ctx{registry,     snapshot,    opToId, valueToTensorId,
                       pendingCalls, pendingRefs, lineMap};

  serializeOp(rootOp, /*parentId=*/"", /*regionIdx=*/0, /*blockIdx=*/0,
              /*order=*/0, ctx);

  // Materialize symbol edges now that every op has an id. Endpoints pruned by a
  // visitor simply drop out.
  for (auto &[caller, callee] : pendingCalls) {
    auto c = opToId.find(caller), e = opToId.find(callee);
    if (c != opToId.end() && e != opToId.end()) {
      snapshot.calls.push_back({c->second, e->second});
    }
  }
  for (auto &[user, target, attrName] : pendingRefs) {
    auto u = opToId.find(user), t = opToId.find(target);
    if (u != opToId.end() && t != opToId.end()) {
      snapshot.refs.push_back({u->second, t->second, attrName});
    }
  }

  return snapshot;
}
