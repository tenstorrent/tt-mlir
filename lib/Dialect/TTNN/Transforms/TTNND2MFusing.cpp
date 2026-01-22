// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <deque>

#define DEBUG_TYPE "ttnn-d2m-fusing"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNND2MFUSING
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// 1. Find exit ops: eltwise ops where all consumers are non-eltwise
// 2. From each exit, BFS backward through eltwise producers
// 3. Include producer P only if ALL of P's consumers are already in F
//    (this ensures no escaping from middle of a fusion group)
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

    if (mlir::isa<AbsOp, CbrtOp, CeilOp, SignOp, CosOp, ExpOp, ErfOp, ErfcOp,
                  FloorOp, GeluOp, IsFiniteOp, LogicalNotOp, BitwiseNotOp,
                  NegOp, TanOp, AtanOp, TanhOp, ReciprocalOp, ReluOp, Relu6Op,
                  SinOp, SqrtOp, RsqrtOp, SigmoidOp, HardsigmoidOp, SiluOp,
                  MishOp, LogOp, Log1pOp, Expm1Op, LeakyReluOp>(op)) {
      return true;
    }

    if (mlir::isa<AddOp, DivideOp, MultiplyOp, SubtractOp, EqualOp, NotEqualOp,
                  GreaterEqualOp, GreaterThanOp, LessEqualOp, LessThanOp,
                  LogicalAndOp, LogicalOrOp, LogicalXorOp, LogicalRightShiftOp,
                  BitwiseAndOp, BitwiseOrOp, BitwiseXorOp, MaximumOp, MinimumOp,
                  RemainderOp, LogicalLeftShiftOp, Atan2Op, PowTensorOp,
                  PowScalarOp>(op)) {
      return true;
    }

    return false;
  }

private:
  // Check if op is a fusion group exit (eltwise with no eltwise consumers).
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
    llvm::SmallVector<Operation *> exits;
    rootOp->walk([&](Block *block) {
      for (Operation &op : llvm::reverse(*block)) {
        if (isFusionGroupExit(&op) && !visitedOps.contains(&op)) {
          exits.push_back(&op);
        }
      }
    });

    // Build a fusion group backward from each exit.
    for (Operation *exit : exits) {
      if (visitedOps.contains(exit)) {
        continue;
      }

      FusionGroup group = buildFusionGroup(exit);

      if (group.size() >= 2) {
        for (Operation *op : group) {
          visitedOps.insert(op);
        }
        fusionGroups.push_back(std::move(group));
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "ElementwiseFusionAnalysis found " << fusionGroups.size()
                   << " valid fusion groups:\n";
      for (const auto &group : fusionGroups) {
        llvm::dbgs() << "  Fusion group with " << group.size() << " ops:\n";
        for (Operation *op : group) {
          llvm::dbgs() << "    " << op->getName() << " at " << op->getLoc()
                       << "\n";
        }
      }
    });
  }

  // BFS backward from exit op and include producer only if all its consumers
  // are in the fusion set. BFS order guarantees correctness since consumers are
  // processed before their producers.
  FusionGroup buildFusionGroup(Operation *exit) {
    llvm::DenseSet<Operation *> fusionSet;
    std::deque<Operation *> stack;

    fusionSet.insert(exit);
    stack.push_back(exit);

    while (!stack.empty()) {
      Operation *cur = stack.back();
      stack.pop_back();

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
    Block *block = exit->getBlock();
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

    // Identify inputs: values produced outside group, consumed by group ops.
    llvm::SmallVector<Value> inputs;
    llvm::DenseSet<Value> inputSet;
    for (Operation *op : fusionGroup) {
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!fusionGroupSet.contains(defOp) && !inputSet.contains(operand)) {
          inputs.push_back(operand);
          inputSet.insert(operand);
        }
      }
    }

    // The last op in topological order is the exit - its results are the
    // outputs.
    Operation *exitOp = fusionGroup.back();
    llvm::SmallVector<Value> outputs(exitOp->getResults());

    if (outputs.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Fusion group " << groupIdx
                              << " has no external outputs\n");
    }

    // Insert before the first group op.
    Operation *firstOp = fusionGroup.front();
    rewriter.setInsertionPoint(firstOp);
    Location loc = firstOp->getLoc();

    // Get device for creating empty output buffers.
    auto device = utils::getOrInsertDevice(rewriter, firstOp);

    // Build function type and create empty outputs for DPS.
    llvm::SmallVector<Type> inputTypes;
    for (Value v : inputs) {
      inputTypes.push_back(v.getType());
    }

    ttcore::GridAttr deviceGrid = ttcore::lookupDevice(firstOp).getWorkerGrid();

    llvm::SmallVector<Type> outputTypes;
    llvm::SmallVector<Value> outputBuffers;
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
    }

    auto funcType = rewriter.getFunctionType(inputTypes, outputTypes);

    std::string funcName = "d2m_subgraph_" + std::to_string(groupIdx);

    // Create DispatchD2MOp with ttnn.empty ops as pre-allocated outputs.
    auto dispatchOp = rewriter.create<DispatchD2MOp>(
        loc, outputTypes, inputs, outputBuffers,
        SymbolRefAttr::get(rewriter.getContext(), funcName));

    // Create body with nested module and function.
    Region &body = dispatchOp.getBody();
    Block *bodyBlock = &body.emplaceBlock();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(bodyBlock);

      auto nestedModule = rewriter.create<ModuleOp>(loc);

      rewriter.setInsertionPointToStart(nestedModule.getBody());
      auto funcOp = func::FuncOp::create(loc, funcName, funcType);
      nestedModule.push_back(funcOp);

      Block *funcBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(funcBlock);

      // Map original input values to function arguments, clone ops, create
      // return.
      IRMapping mapping;
      mapping.map(inputs, funcOp.getArguments());

      for (Operation *op : fusionGroup) {
        rewriter.clone(*op, mapping);
      }
      auto returnValues = llvm::map_to_vector(
          outputs, [&](Value v) { return mapping.lookup(v); });
      rewriter.create<func::ReturnOp>(loc, returnValues);
    }

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
