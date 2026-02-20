// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNND2MFUSING
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// 1. Find exit ops: eltwise ops where all consumers are non-eltwise
// 2. From each exit, DFS backward through eltwise producers
// 3. Include producer P only if ALL of P's consumers are already in the fusion
//    group (this ensures no escaping from middle of a fusion group)
// 4. Stop at non-eltwise producers (these are entry points)
// 5. Multiple entry points allowed
class ElementwiseFusionAnalysis {
public:
  using FusionGroup = llvm::SmallVector<Operation *>;

  explicit ElementwiseFusionAnalysis(Operation *rootOp) { analyze(rootOp); }

  llvm::ArrayRef<FusionGroup> getFusionGroups() const { return fusionGroups; }

  static bool isElementwiseOp(Operation *op) {
    if (!op) {
      return false;
    }
    // Eltwise unary ops.
    if (mlir::isa<AbsOp, CbrtOp, CeilOp, SignOp, CosOp, ExpOp, ErfOp, ErfcOp,
                  FloorOp, GeluOp, IsFiniteOp, LogicalNotOp, BitwiseNotOp,
                  NegOp, TanOp, TanhOp, ReciprocalOp, ReluOp, SinOp, SqrtOp,
                  RsqrtOp, SigmoidOp, HardsigmoidOp, SiluOp, MishOp, LogOp,
                  Log1pOp, Expm1Op>(op)) {
      return true;
    }
    // Eltwise binary ops.
    if (mlir::isa<AddOp, DivideOp, MultiplyOp, SubtractOp, EqualOp, NotEqualOp,
                  GreaterEqualOp, GreaterThanOp, LessEqualOp, LessThanOp,
                  LogicalAndOp, LogicalOrOp, LogicalXorOp, LogicalRightShiftOp,
                  BitwiseAndOp, BitwiseOrOp, BitwiseXorOp, MaximumOp, MinimumOp,
                  RemainderOp, LogicalLeftShiftOp, Atan2Op, PowTensorOp>(op)) {
      return true;
    }

    return false;
  }

private:
  static bool isFusionGroupExit(Operation *op) {
    if (!isElementwiseOp(op)) {
      return false;
    }
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (isElementwiseOp(user)) {
          return false;
        }
      }
    }
    return true;
  }

  void analyze(Operation *rootOp) {
    // Find all fusion group exit ops by walking backwards.
    // Build out each fusion group backwards from the found exit ops.
    llvm::SmallVector<Operation *> exitOps;
    rootOp->walk([&](Block *block) {
      for (Operation &op : llvm::reverse(*block)) {
        if (isFusionGroupExit(&op) && !visitedOps.contains(&op)) {
          exitOps.push_back(&op);
        }
      }
    });

    for (Operation *exitOp : exitOps) {
      FusionGroup group = buildFusionGroup(exitOp);

      if (group.size() >= 2) {
        for (Operation *op : group) {
          visitedOps.insert(op);
        }
        fusionGroups.push_back(std::move(group));
      }
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::D2MFusion,
                 "ElementwiseFusionAnalysis found {} valid fusion groups",
                 fusionGroups.size());
    for (const auto &group : fusionGroups) {
      TTMLIR_DEBUG(ttmlir::LogComponent::D2MFusion,
                   "  Fusion group with {} ops", group.size());
      for ([[maybe_unused]] Operation *op : group) {
        TTMLIR_DEBUG(ttmlir::LogComponent::D2MFusion, "    {} at {}",
                     op->getName(), op->getLoc());
      }
    }
  }

  // DFS backward from exit op and include producer only if all its consumers
  // are in the fusion set.
  FusionGroup buildFusionGroup(Operation *exitOp) {
    llvm::DenseSet<Operation *> fusionSet;
    llvm::SmallVector<Operation *> stack;

    fusionSet.insert(exitOp);
    stack.push_back(exitOp);

    while (!stack.empty()) {
      Operation *cur = stack.pop_back_val();

      for (Value operand : cur->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (!producer || !isElementwiseOp(producer) ||
            visitedOps.contains(producer) || fusionSet.contains(producer)) {
          continue;
        }

        // All consumers of producers must be in the fusion group set
        bool allConsumersInSet =
            llvm::all_of(producer->getUsers(), [&](Operation *user) {
              return fusionSet.contains(user);
            });

        if (allConsumersInSet) {
          fusionSet.insert(producer);
          stack.push_back(producer);
        }
      }
    }

    // Return ordered fusion group.
    FusionGroup fusionGroup;
    Block *block = exitOp->getBlock();
    for (Operation &op : *block) {
      if (fusionSet.contains(&op)) {
        fusionGroup.push_back(&op);
      }
    }
    return fusionGroup;
  }

  llvm::DenseSet<Operation *> visitedOps;
  llvm::SmallVector<FusionGroup> fusionGroups;
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

class TTNND2MFusingPass : public impl::TTNND2MFusingBase<TTNND2MFusingPass> {
public:
  using impl::TTNND2MFusingBase<TTNND2MFusingPass>::TTNND2MFusingBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    ElementwiseFusionAnalysis analysis(moduleOp);
    auto fusionGroups = analysis.getFusionGroups();
    if (fusionGroups.empty()) {
      return;
    }

    IRRewriter rewriter(&getContext());
    for (unsigned groupIdx = 0; groupIdx < fusionGroups.size(); ++groupIdx) {
      if (failed(wrapFusionGroup(fusionGroups[groupIdx], rewriter, groupIdx))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult wrapFusionGroup(llvm::ArrayRef<Operation *> fusionGroup,
                                IRRewriter &rewriter, unsigned groupIdx) {
    if (fusionGroup.empty()) {
      return success();
    }

    llvm::DenseSet<Operation *> fusionGroupSet(fusionGroup.begin(),
                                               fusionGroup.end());

    // Identify inputs: values produced outside fusion group that are consumed
    // by fusion group ops.
    llvm::SetVector<Value> inputs;
    for (Operation *op : fusionGroup) {
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!fusionGroupSet.contains(defOp)) {
          inputs.insert(operand);
        }
      }
    }

    Operation *exitOp = fusionGroup.back();
    llvm::SmallVector<Value> outputs(exitOp->getResults());
    if (outputs.empty()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::D2MFusion,
                   "Fusion group {} has no external outputs", groupIdx);
    }

    Operation *firstOp = fusionGroup.front();
    // Place the subgraph after the last op that defines any input, so all
    // inputs dominate the subgraph.
    Operation *lastInputDefiner = nullptr;
    for (Value v : inputs) {
      Operation *defOp = v.getDefiningOp();
      if (defOp &&
          (!lastInputDefiner || lastInputDefiner->isBeforeInBlock(defOp))) {
        lastInputDefiner = defOp;
      }
    }
    if (lastInputDefiner) {
      rewriter.setInsertionPointAfter(lastInputDefiner);
    } else {
      rewriter.setInsertionPoint(firstOp);
    }
    Location loc = firstOp->getLoc();

    // Get device for creating empty output buffers for DPS.
    auto device = utils::getOrInsertDevice(rewriter, firstOp);
    ttcore::GridAttr deviceGrid = ttcore::lookupDevice(firstOp).getWorkerGrid();
    llvm::SmallVector<Type> inputTypes;
    llvm::transform(inputs, std::back_inserter(inputTypes),
                    [](Value v) { return v.getType(); });

    llvm::SmallVector<Type> outputTypes;
    llvm::SmallVector<Value> outputBuffers;
    Operation *lastEmptyOp = nullptr;
    for (Value v : outputs) {
      auto tensorType = mlir::cast<RankedTensorType>(v.getType());
      outputTypes.push_back(tensorType);

      auto layoutAttr = mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());
      auto shapeAttr =
          ShapeAttr::get(rewriter.getContext(), tensorType.getShape());
      auto dtypeAttr = ttcore::DataTypeAttr::get(rewriter.getContext(),
                                                 layoutAttr.getDataType());
      auto tensorLayoutAttr =
          LayoutAttr::get(rewriter.getContext(), layoutAttr.getLayout());
      auto memoryConfigAttr = MemoryConfigAttr::get(layoutAttr, deviceGrid);

      auto emptyOp = rewriter.create<EmptyOp>(
          loc, tensorType, device, shapeAttr, dtypeAttr, tensorLayoutAttr,
          memoryConfigAttr);
      outputBuffers.push_back(emptyOp.getResult());
      lastEmptyOp = emptyOp.getOperation();
    }

    auto funcType = rewriter.getFunctionType(inputTypes, outputTypes);

    std::string funcName = "d2m_subgraph_" + std::to_string(groupIdx);

    // Get the parent module to create the function at module scope.
    ModuleOp parentModule = firstOp->getParentOfType<ModuleOp>();
    if (!parentModule) {
      return firstOp->emitError("Could not find parent module");
    }

    SymbolTable symbolTable(parentModule);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(parentModule.getBody());

    // Create private function at module scope.
    auto funcOp = func::FuncOp::create(loc, funcName, funcType);
    funcOp.setPrivate();
    symbolTable.insert(funcOp);
    Block *funcBlock = funcOp.addEntryBlock();
    rewriter.setInsertionPointToStart(funcBlock);

    // Map original input values to function arguments, clone ops, create
    // return.
    IRMapping mapping;
    mapping.map(inputs.getArrayRef(), funcOp.getArguments());

    for (Operation *op : fusionGroup) {
      rewriter.clone(*op, mapping);
    }
    llvm::SmallVector<Value> returnValues;
    llvm::transform(outputs, std::back_inserter(returnValues),
                    [&](Value v) { return mapping.lookup(v); });
    rewriter.create<func::ReturnOp>(loc, returnValues);

    // Place subgraph after all its operands: after last input definer and after
    // the empty output buffers we just created (so output buffers dominate).
    if (lastEmptyOp) {
      rewriter.setInsertionPointAfter(lastEmptyOp);
    } else if (lastInputDefiner) {
      rewriter.setInsertionPointAfter(lastInputDefiner);
    } else {
      rewriter.setInsertionPoint(firstOp);
    }
    auto dispatchOp = rewriter.create<D2MSubgraphOp>(
        loc, outputTypes, inputs.getArrayRef(), outputBuffers,
        SymbolRefAttr::get(rewriter.getContext(), funcName));

    for (auto [origOutput, dispatchResult] :
         llvm::zip(outputs, dispatchOp.getResults())) {
      rewriter.replaceAllUsesExcept(origOutput, dispatchResult, dispatchOp);
    }

    for (auto it = fusionGroup.rbegin(); it != fusionGroup.rend(); ++it) {
      rewriter.eraseOp(*it);
    }

    return success();
  }
};

} // namespace

} // namespace mlir::tt::ttnn
