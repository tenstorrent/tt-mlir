// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "D2MOpScheduler"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MOPSCHEDULER
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
struct D2MOpSchedulerRewriter final : public OpRewritePattern<GenericOp> {
public:
  D2MOpSchedulerRewriter(mlir::MLIRContext *ctx)
      : OpRewritePattern<GenericOp>(ctx) {};

  // OpTreeNode for building trees for register allocation using
  // Sethi-Ullman algorithm. Supports unary, binary, and ternary operations.
  class OpTreeNode {
  public:
    Operation *op = nullptr;
    BlockArgument blockArg = nullptr;
    bool isBlockArg = false;
    unsigned int weight = 0;
    OpTreeNode *left = nullptr;
    OpTreeNode *middle = nullptr;
    OpTreeNode *right = nullptr;
    OpTreeNode *parent = nullptr;

    OpTreeNode() = default;
    OpTreeNode(Operation *op) : op(op) {}
    OpTreeNode(BlockArgument blockArg) : blockArg(blockArg), isBlockArg(true) {}
    ~OpTreeNode() {
      delete left;
      delete middle;
      delete right;
    }
  };

  // Build an OpTreeNode binary tree from affine loops by tracing back from
  // the final affine.store in the innermost loop.
  static std::pair<std::unique_ptr<OpTreeNode>, SmallVector<OpTreeNode *>>
  buildOpTree(Operation *outermostLoop) {
    SmallVector<OpTreeNode *> leafNodes;

    if (!outermostLoop) {
      return {nullptr, leafNodes};
    }

    // Find the innermost affine.for loop using post-order traversal.
    // Default walk is post-order, first loop visited is innermost.
    affine::AffineForOp innermostLoop = nullptr;
    outermostLoop->walk([&](affine::AffineForOp loop) {
      innermostLoop = loop;
      return WalkResult::interrupt();
    });

    if (!innermostLoop) {
      return {nullptr, leafNodes};
    }

    // Find the last affine.store in the innermost loop.
    affine::AffineStoreOp finalStore = nullptr;
    innermostLoop.getBody()->walk([&](affine::AffineStoreOp storeOp) {
      finalStore = storeOp;
      return WalkResult::interrupt();
    });

    if (!finalStore) {
      return {nullptr, leafNodes};
    }

    // Get the operation that generates the stored SSA value.
    Value storedValue = finalStore.getValue();
    Operation *rootOp = storedValue.getDefiningOp();
    if (!rootOp) {
      return {nullptr, leafNodes};
    }

    // Get the block containing the affine loop operations.
    Block *affineBlock = innermostLoop.getBody();

    // Build the tree recursively starting from the root operation.
    auto rootNode = buildOpTreeRecursive(rootOp, affineBlock, leafNodes);
    return {std::move(rootNode), leafNodes};
  }

  // Recursive helper to build the OpTreeNode from an operation.
  // Works with affine loop blocks, treating affine.load operations as leaf
  // nodes.
  static std::unique_ptr<OpTreeNode>
  buildOpTreeRecursive(Operation *op, Block *affineBlock,
                       SmallVector<OpTreeNode *> &leafNodes) {

    // Create the node for this operation.
    auto node = std::make_unique<OpTreeNode>(op);

    // Process operands to build child nodes.
    SmallVector<std::unique_ptr<OpTreeNode>> children;

    for (Value operand : op->getOperands()) {
      if (Operation *definingOp = operand.getDefiningOp()) {
        // If operand is an affine.load, treat it as a leaf node.
        if (mlir::isa<affine::AffineLoadOp>(definingOp)) {
          auto leafNode = std::make_unique<OpTreeNode>(definingOp);
          leafNodes.push_back(leafNode.get());
          children.push_back(std::move(leafNode));
        }
        // If operand is defined by a compute operation in the affine block,
        // recurse.
        else if (definingOp->getBlock() == affineBlock) {
          children.push_back(
              buildOpTreeRecursive(definingOp, affineBlock, leafNodes));
        }
      }
    }

    // Assign children based on operand count.
    if (children.size() == 1) {
      // Unary operation.
      node->left = children[0].release();
      node->left->parent = node.get();
    } else if (children.size() == 2) {
      // Binary operation.
      node->left = children[0].release();
      node->right = children[1].release();
      node->left->parent = node.get();
      node->right->parent = node.get();
    } else if (children.size() == 3) {
      // Ternary operation.
      node->left = children[0].release();
      node->middle = children[1].release();
      node->right = children[2].release();
      node->left->parent = node.get();
      node->middle->parent = node.get();
      node->right->parent = node.get();
    }

    return node;
  }

  // Mark node weights using the Sethi-Ullman algorithm for register
  // allocation. Uses post-order traversal (children before parent).
  static void markNodeWeights(OpTreeNode *node) {
    if (!node) {
      return;
    }

    // Post-order: process children first.
    markNodeWeights(node->left);
    markNodeWeights(node->middle);
    markNodeWeights(node->right);

    // Calculate weight for this node.
    if (!node->left && !node->middle && !node->right) {
      // Leaf node (block argument).
      node->weight = 1;
    } else if (node->middle == nullptr && node->right == nullptr) {
      // Unary operation.
      assert(node->left && "Unary operation must have left child");
      node->weight = node->left->weight;
    } else if (node->middle == nullptr) {
      // Binary operation.
      assert(node->left && "Binary operation must have left child");
      assert(node->right && "Binary operation must have right child");
      unsigned leftWeight = node->left->weight;
      unsigned rightWeight = node->right->weight;

      if (leftWeight == rightWeight) {
        node->weight = leftWeight + 1;
      } else {
        node->weight = std::max(leftWeight, rightWeight);
      }
    } else {
      // Ternary operation.
      assert(node->left && "Ternary operation must have left child");
      assert(node->middle && "Ternary operation must have middle child");
      assert(node->right && "Ternary operation must have right child");
      unsigned leftWeight = node->left->weight;
      unsigned middleWeight = node->middle->weight;
      unsigned rightWeight = node->right->weight;

      // Sort weights in descending order for optimal evaluation order.
      SmallVector<unsigned, 3> weights = {leftWeight, middleWeight,
                                          rightWeight};
      llvm::sort(weights, std::greater<unsigned>());

      // Sethi-Ullman formula: max(w[0], 1 + max(w[1], 1 + w[2]))
      // Models: evaluate heaviest first, store (1 reg), evaluate next (while
      // holding 1), store (2 regs), evaluate lightest (while holding 2).
      unsigned innerMax = std::max(weights[1], 1 + weights[2]);
      node->weight = std::max(weights[0], 1 + innerMax);
    }
  }

  // Print the operation tree for debugging/visualization.
  static void printOpTree(OpTreeNode *node, int depth = 0) {
    if (!node) {
      return;
    }

    // Create indentation based on depth.
    std::string indent(depth * 2, ' ');

    // Print current node information.
    if (node->isBlockArg) {
      // This is a block argument leaf node.
      LDBG() << indent << "├─ BlockArg #" << node->blockArg.getArgNumber()
             << " | Weight: " << node->weight << " [LEAF]";
    } else if (node->op && mlir::isa<affine::AffineLoadOp>(node->op)) {
      // This is an affine.load leaf node.
      LDBG() << indent << "├─ Op: " << node->op->getName()
             << " | Weight: " << node->weight << " [LEAF]";
    } else {
      // This is an operation node.
      LDBG() << indent << "├─ Op: " << node->op->getName()
             << " | Weight: " << node->weight;
    }

    // Print children with appropriate labels.
    if (node->left || node->middle || node->right) {
      if (node->left) {
        LDBG() << indent << "  Left:";
        printOpTree(node->left, depth + 1);
      }
      if (node->middle) {
        LDBG() << indent << "  Middle:";
        printOpTree(node->middle, depth + 1);
      }
      if (node->right) {
        LDBG() << indent << "  Right:";
        printOpTree(node->right, depth + 1);
      }
    }
  }

  // Apply Sethi-Ullman ordering by recursively traversing the tree and
  // moving operations to minimize register usage. Traverses depth-first,
  // prioritizing the branch with higher weight.
  static void applySeethiUllmanOrdering(PatternRewriter &rewriter,
                                        OpTreeNode *node,
                                        DenseSet<Operation *> &movedOps) {
    if (!node || !node->op) {
      return;
    }

    // Collect all children and sort by weight (highest first).
    SmallVector<OpTreeNode *> children;
    if (node->left) {
      children.push_back(node->left);
    }
    if (node->middle) {
      children.push_back(node->middle);
    }
    if (node->right) {
      children.push_back(node->right);
    }

    // Sort children by weight in descending order (higher weight first).
    llvm::sort(children, [](OpTreeNode *a, OpTreeNode *b) {
      return a->weight > b->weight;
    });

    // Recursively process children depth-first in weight order.
    for (OpTreeNode *child : children) {
      applySeethiUllmanOrdering(rewriter, child, movedOps);
    }

    // Check if this is a leaf node (affine.load).
    bool isLeaf = mlir::isa<affine::AffineLoadOp>(node->op);

    // Check if all children have been moved (for non-leaf nodes).
    bool allChildrenMoved = true;
    if (node->left && node->left->op) {
      allChildrenMoved &= movedOps.contains(node->left->op);
    }
    if (node->middle && node->middle->op) {
      allChildrenMoved &= movedOps.contains(node->middle->op);
    }
    if (node->right && node->right->op) {
      allChildrenMoved &= movedOps.contains(node->right->op);
    }

    // Move the operation if it's a leaf or all children have been handled.
    if (isLeaf || allChildrenMoved) {
      // Move the operation to the current insertion point.
      node->op->moveBefore(rewriter.getInsertionBlock(),
                           rewriter.getInsertionPoint());

      // Mark this operation as moved.
      movedOps.insert(node->op);

      // Update insertion point to after the moved operation.
      rewriter.setInsertionPointAfter(node->op);
    }
  }

  // Entry point for Sethi-Ullman register allocation reordering.
  // Finds the innermost loop, sets insertion point, and applies reordering.
  static void reorderOperationsSethiUllman(PatternRewriter &rewriter,
                                           Operation *outermostLoop,
                                           OpTreeNode *opTreeRoot) {
    if (!opTreeRoot || !outermostLoop) {
      return;
    }

    // Find the innermost affine.for loop.
    affine::AffineForOp innermostLoop = nullptr;
    outermostLoop->walk([&](affine::AffineForOp loop) {
      innermostLoop = loop;
      return WalkResult::interrupt();
    });

    if (!innermostLoop) {
      return;
    }

    // Set insertion point to the beginning of the innermost loop's body.
    Block *innermostBody = innermostLoop.getBody();
    rewriter.setInsertionPointToStart(innermostBody);

    // Track which operations have been moved.
    DenseSet<Operation *> movedOps;

    // Apply Sethi-Ullman ordering starting from the root.
    applySeethiUllmanOrdering(rewriter, opTreeRoot, movedOps);

    LDBG() << "=== Sethi-Ullman Reordering Applied ===";
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    if (!op.isNontriviallyEltwiseFused()) {
      return failure();
    }

    // Collect all root affine.for loops inside d2m.generic
    SmallVector<Operation *, 8> rootLoops;

    Operation *oop = op.getOperation();
    oop->walk<WalkOrder::PreOrder>([&](affine::AffineForOp loop) {
      if (loop->hasAttr("d2m.linalg_root")) {

        // skip already scheduled loops
        if (!loop->hasAttr("d2m.scheduled")) {
          rootLoops.push_back(loop);
          loop->setAttr("d2m.scheduled", rewriter.getUnitAttr());
        }

        // skip other loops nested within root loop
        WalkResult::skip();
      }
      WalkResult::advance();
    });

    if (rootLoops.empty()) {
      return failure();
    }

    bool modified = false;

    for (auto &loop : rootLoops) {
      auto [opTreeRoot, leafNodes] = buildOpTree(loop);

      if (opTreeRoot) {
        // Mark node weights using Sethi-Ullman algorithm.
        markNodeWeights(opTreeRoot.get());

        LDBG() << "========== Operation Tree ===========";
        printOpTree(opTreeRoot.get());
        LDBG() << "=====================================";

        // Reorder operations.
        reorderOperationsSethiUllman(rewriter, loop, opTreeRoot.get());
        modified = true;
      }
    }

    return success(modified);
  }
};
} // namespace

namespace {
class D2MOpScheduler : public impl::D2MOpSchedulerBase<D2MOpScheduler> {
public:
  using impl::D2MOpSchedulerBase<D2MOpScheduler>::D2MOpSchedulerBase;

  void runOnOperation() final {
    // Early exit if op scheduler is disabled
    if (!enableOpScheduler) {
      return;
    }

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<D2MOpSchedulerRewriter>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
