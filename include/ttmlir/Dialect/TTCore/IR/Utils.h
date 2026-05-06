// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_IR_UTILS_H
#define TTMLIR_DIALECT_TTCORE_IR_UTILS_H

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::tt::ttcore {

constexpr inline llvm::StringLiteral g_kvCacheAttrName = "ttcore.kv_cache";

class DeviceOp;
class DeviceAttr;
class SystemDescAttr;

inline constexpr llvm::StringRef getDefaultDeviceName() {
  return "default_device";
}

SystemDescAttr getCurrentScopeSystemDesc(Operation *op);

DeviceOp lookupDeviceOp(Operation *op,
                        llvm::StringRef deviceName = getDefaultDeviceName());

DeviceAttr lookupDevice(Operation *op, SymbolRefAttr deviceName);

DeviceAttr lookupDevice(Operation *op,
                        llvm::StringRef deviceName = getDefaultDeviceName());

ChipDescAttr getOpChipDescAttr(Operation *op);

// Create a global memref in the top-level module's symbol table.
mlir::memref::GlobalOp createGlobal(ModuleOp moduleOp, StringRef name,
                                    MemRefType type, ElementsAttr value,
                                    bool constant = true,
                                    bool privateVisibility = true,
                                    size_t alignment = 0);

// Overload auto-generating the name for the above.
mlir::memref::GlobalOp createGlobal(ModuleOp moduleOp, MemRefType type,
                                    ElementsAttr value, bool constant = true,
                                    bool privateVisibility = true,
                                    size_t alignment = 0);

// Helper function to check if a block argument is consteval-able (Parameter or
// Constant)
inline bool isConstOrParamArg(mlir::BlockArgument blockArg,
                              mlir::func::FuncOp funcOp) {
  if (auto typeAttr = funcOp.getArgAttrOfType<ArgumentTypeAttr>(
          blockArg.getArgNumber(), ArgumentTypeAttr::name)) {
    auto argTypeValue = typeAttr.getValue();
    return argTypeValue == ArgumentType::Parameter ||
           argTypeValue == ArgumentType::Constant;
  }
  return false;
}

// Filters out the constant parameters from the function signature.
inline llvm::SmallPtrSet<mlir::BlockArgument, 4>
getConstsAndParams(mlir::func::FuncOp funcOp) {
  if (funcOp.isDeclaration()) {
    return {};
  }

  llvm::SmallPtrSet<mlir::BlockArgument, 4> constsAndParams;

  for (auto arg : funcOp.getArguments()) {
    if (isConstOrParamArg(arg, funcOp)) {
      constsAndParams.insert(arg);
    }
  }

  return constsAndParams;
}

// Forward-propagation analysis for `valueTracesToConstantArgs`. Caches the
// answer for every value in a func::FuncOp using an O(num_ops) walk; lookups
// are O(1). The slow path (recursive use-def walk) is O(num_ops) per call and
// gets called many times during pattern matching in passes like
// TTIREraseInverseOps -> O(num_ops^2) overall. For deep models (e.g. 10+
// layer DeepSeek) the slow path dominates compile time (~55s in
// TTIREraseInverseOps at 10 layer).
//
// The analysis acts as a `RewriterBase::Listener` so it can be installed on
// `GreedyRewriteConfig::setListener`. Any IR mutation (insert / modify /
// replace / erase) flips a dirty flag; the next `lookup` rebuilds the cache
// from scratch. This is correct in the face of MLIR's Value pointer
// recycling: stale pointers are wiped on every rebuild and never observed.
//
// Semantics match `valueTracesToConstantArgs` bug-for-bug: a value traces to
// constant args iff its use-def chain reaches at least one CONST/PARAM block
// argument and no non-CONST/PARAM block argument.
class ConstevalForwardAnalysis : public mlir::RewriterBase::Listener {
public:
  // Captures `root` (typically a func::FuncOp). `lookup` lazily builds the
  // cache on first use and rebuilds whenever a Listener notification has
  // marked it dirty.
  explicit ConstevalForwardAnalysis(mlir::Operation *root);

  // Returns std::nullopt if the value is unknown to this analysis (e.g.
  // created after the most recent rebuild or in an unvisited region).
  // Callers should fall back to the slow path on nullopt.
  std::optional<bool> lookup(mlir::Value v);

  size_t size() const { return cache_.size(); }

  // Number of times the cache has been (re)built. Useful for tests that
  // verify the listener-based invalidation actually keeps the cache live
  // across pattern rewrites.
  size_t rebuildCount() const { return rebuildCount_; }

  // RewriterBase::Listener overrides — any IR mutation invalidates the
  // cache. Insertion alone is also flagged: the new op's results are not in
  // the cache, but Value pointer recycling means we cannot trust other
  // entries either.
  void notifyOperationInserted(mlir::Operation *,
                               mlir::OpBuilder::InsertPoint) override {
    dirty_ = true;
  }
  void notifyOperationModified(mlir::Operation *) override { dirty_ = true; }
  void notifyOperationReplaced(mlir::Operation *, mlir::Operation *) override {
    dirty_ = true;
  }
  void notifyOperationReplaced(mlir::Operation *, mlir::ValueRange) override {
    dirty_ = true;
  }
  void notifyOperationErased(mlir::Operation *) override { dirty_ = true; }

private:
  void rebuild();

  mlir::Operation *root_;
  bool dirty_ = true;
  size_t rebuildCount_ = 0;
  // Per-value: (hasConstOrParamArg, hasNonConstOrParamArg). We need both bits
  // because the result is `hasConst && !hasNonConst`.
  llvm::DenseMap<mlir::Value, std::pair<bool, bool>> cache_;
};

// Thread-local scope guard. While alive, the contained analysis is the
// "active" one and `valueTracesToConstantArgs` will consult it on lookup.
// Caller is responsible for installing the analysis as a
// `RewriterBase::Listener` on any greedy rewrite driver invoked inside the
// scope (otherwise mutations will not invalidate the cache and answers can
// go stale).
class ConstevalAnalysisScope {
public:
  explicit ConstevalAnalysisScope(ConstevalForwardAnalysis &analysis);
  ~ConstevalAnalysisScope();
  ConstevalAnalysisScope(const ConstevalAnalysisScope &) = delete;
  ConstevalAnalysisScope &operator=(const ConstevalAnalysisScope &) = delete;

private:
  ConstevalForwardAnalysis *previous_;
};

// Internal: returns the active scope's analysis, or nullptr if none.
ConstevalForwardAnalysis *getActiveConstevalAnalysis();

// This function will return true if a given Value is the result of operations
// performed only between  block arguments in which have been marked as
// consteval-able (Parameter or Constant ArgumentType).
inline bool valueTracesToConstantArgs(const mlir::Value &value) {
  if (auto *analysis = getActiveConstevalAnalysis()) {
    if (auto cached = analysis->lookup(value)) {
      return *cached;
    }
    // Cache miss (e.g. value created after the most recent rebuild) — fall
    // through to the slow recursive path below.
  }

  auto useDefChain = ttmlir::utils::getUseDefChain(value);
  auto subgraphBlockArgs =
      ttmlir::utils::filterBlockArguments(useDefChain.getArrayRef());
  mlir::func::FuncOp funcOp = nullptr;

  if (!subgraphBlockArgs.empty()) {
    mlir::Block *argOwner = subgraphBlockArgs.front().getOwner();
    funcOp =
        mlir::dyn_cast_or_null<mlir::func::FuncOp>(argOwner->getParentOp());
  }
  if (!funcOp) {
    return false;
  }

  for (auto blockArg : subgraphBlockArgs) {
    // Require ownership by the same func::FuncOp before consulting its
    // argument attributes; otherwise an inner-region block argument could
    // accidentally be classified as CONST/PARAM via funcOp.getArgAttr at
    // a matching argNumber. Non-function block arguments never carry
    // ttcore::ArgumentType so the value cannot be const-eval-traceable.
    if (blockArg.getOwner()->getParentOp() != funcOp.getOperation()) {
      return false;
    }
    if (!isConstOrParamArg(blockArg, funcOp)) {
      return false;
    }
  }

  return true;
}

bool isTiled(RankedTensorType tensorType);

ArrayRef<int64_t> getTensorTileShape(RankedTensorType tensorType);

ArrayRef<int64_t> getTensorTileShapeOrEmpty(RankedTensorType tensorType);

llvm::SmallVector<int64_t, 2> collapseGridTo2D(ArrayRef<int64_t> gridShape);

// Retrieve the layout from the shaped type (ie. getEncoding for tensors and
// getLayout for memrefs).
inline DeviceLayoutInterface getDeviceLayout(ShapedType shapedType) {
  if (auto tensor = mlir::dyn_cast_if_present<RankedTensorType>(shapedType)) {
    return mlir::dyn_cast_if_present<DeviceLayoutInterface>(
        tensor.getEncoding());
  }

  if (auto memref = mlir::dyn_cast_if_present<MemRefType>(shapedType)) {
    return mlir::dyn_cast_if_present<DeviceLayoutInterface>(memref.getLayout());
  }

  return nullptr;
}

// Convenience overload that extracts the shaped type from a value.
inline DeviceLayoutInterface getDeviceLayout(Value value) {
  return getDeviceLayout(mlir::cast<ShapedType>(value.getType()));
}

inline bool hasDeviceLayout(ShapedType shapedType) {
  return getDeviceLayout(shapedType) != nullptr;
}

inline bool hasDeviceLayout(Value value) {
  return hasDeviceLayout(mlir::cast<ShapedType>(value.getType()));
}

// Helper function to derive grid shape from tensor OR memref using underlying
// layout attr.
inline ArrayRef<int64_t> getGridShape(Value tensorOrMemref) {
  TT_assertv((mlir::isa<RankedTensorType>(tensorOrMemref.getType()) ||
              mlir::isa<MemRefType>(tensorOrMemref.getType())),
             "Expected a tensor or memref type");
  return ttcore::getDeviceLayout(tensorOrMemref)
      .getGridShape(mlir::cast<ShapedType>(tensorOrMemref.getType()));
}

// Helper function to derive shard shape from tensor OR memref using underlying
// layout attr.
inline ArrayRef<int64_t> getShardShape(Value tensorOrMemref) {
  TT_assertv((mlir::isa<RankedTensorType>(tensorOrMemref.getType()) ||
              mlir::isa<MemRefType>(tensorOrMemref.getType())),
             "Expected a tensor or memref type");
  return ttcore::getDeviceLayout(tensorOrMemref)
      .getShardShape(mlir::cast<ShapedType>(tensorOrMemref.getType()));
}

Type getOperandInnerElementType(const mlir::Value operand);

// Convert a TensorType with MetalLayoutAttr encoding into a MemRefType with
// appropriate layout attributes (Shard/View/Host/Interleaved).
bufferization::BufferLikeType
getBufferType(Type type, bool isView,
              std::optional<MetalLayoutAttr> hostInfo = std::nullopt);

// ArgumentType helpers

// Retrieves the ArgumentType for a given function argument, defaulting to
// Input if not specified.
inline ArgumentType getFunctionArgumentType(func::FuncOp op, size_t argIndex) {
  auto argAttrDict = op.getArgAttrDict(argIndex);
  if (argAttrDict && argAttrDict.contains(ArgumentTypeAttr::name)) {
    Attribute attr = argAttrDict.get(ArgumentTypeAttr::name);
    auto argTypeAttr = mlir::cast<ttcore::ArgumentTypeAttr>(attr);
    return argTypeAttr.getValue();
  }

  // Default to Input if not specified
  return ArgumentType::Input;
}

// Checks if the function argument is of type Input.
inline bool isInputArgumentType(func::FuncOp op, size_t argIndex) {
  return getFunctionArgumentType(op, argIndex) == ArgumentType::Input;
}

// Checks if the function argument is of type Constant or Parameter.
inline bool isConstantOrParameterArgumentType(func::FuncOp op,
                                              size_t argIndex) {
  ArgumentType argType = getFunctionArgumentType(op, argIndex);
  return argType == ArgumentType::Constant ||
         argType == ArgumentType::Parameter;
}

// Checks if the function argument has the ttcore.kv_cache attribute attached.
inline bool isKVCacheArgument(func::FuncOp op, size_t argIndex) {
  return op.getArgAttr(argIndex, ttcore::g_kvCacheAttrName) != nullptr;
}

} // namespace mlir::tt::ttcore

#endif // TTMLIR_DIALECT_TTCORE_IR_UTILS_H
