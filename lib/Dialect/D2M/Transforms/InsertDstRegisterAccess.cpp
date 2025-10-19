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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
struct D2MInsertDstRegisterAccessRewriter final
    : public OpRewritePattern<GenericOp> {
public:
  D2MInsertDstRegisterAccessRewriter(mlir::MLIRContext *ctx, bool useTileMatmul,
                                     unsigned maxDstPhysicalSizeTiles)
      : OpRewritePattern<GenericOp>(ctx), useTileMatmul(useTileMatmul),
        maxDstPhysicalSizeTiles(maxDstPhysicalSizeTiles) {};

  // OpTreeNode for building binary trees for register allocation using
  // Sethi-Ullman algorithm.
  class OpTreeNode {
  public:
    Operation *op = nullptr;
    BlockArgument blockArg = nullptr;
    bool isBlockArg = false;
    unsigned int weight = 0;
    OpTreeNode *left = nullptr;
    OpTreeNode *right = nullptr;
    OpTreeNode *parent = nullptr;

    OpTreeNode() = default;
    OpTreeNode(Operation *op) : op(op) {}
    OpTreeNode(BlockArgument blockArg) : blockArg(blockArg), isBlockArg(true) {}
    ~OpTreeNode() {
      delete left;
      delete right;
    }
  };

  // Build an OpTreeNode binary tree from affine loops by tracing back from
  // the final affine.store in the innermost loop.
  static std::pair<std::unique_ptr<OpTreeNode>, SmallVector<OpTreeNode *>>
  buildOpTree(Operation *outermostLoop) {
    int dbgCnt = 0;

    SmallVector<OpTreeNode *> leafNodes;

    llvm::errs() << "BOT " << dbgCnt++ << "\n";

    if (!outermostLoop) {
      return {nullptr, leafNodes};
    }

    llvm::errs() << "BOT " << dbgCnt++ << "\n";

    // Find the innermost affine.for loop using post-order traversal.
    // Post-order visits children before parents, so the last loop is innermost.
    affine::AffineForOp innermostLoop = nullptr;
    outermostLoop->walk([&](affine::AffineForOp loop) {
      llvm::errs() << "Found loop: " << loop << "\n";
      innermostLoop = loop;
      return WalkResult::interrupt();
    });

    llvm::errs() << "BOT " << dbgCnt++ << "\n";

    if (!innermostLoop) {
      return {nullptr, leafNodes};
    }

    llvm::errs() << "BOT " << dbgCnt++ << "\n";
    llvm::errs() << *innermostLoop << "\n";

    // Find the last affine.store in the innermost loop.
    affine::AffineStoreOp finalStore = nullptr;
    innermostLoop.getBody()->walk([&](affine::AffineStoreOp storeOp) {
      finalStore = storeOp;
      return WalkResult::interrupt();
    });

    llvm::errs() << "BOT " << dbgCnt++ << "\n";

    if (!finalStore) {
      return {nullptr, leafNodes};
    }

    llvm::errs() << "BOT " << dbgCnt++ << "\n";

    // Get the operation that generates the stored SSA value.
    Value storedValue = finalStore.getValue();
    Operation *rootOp = storedValue.getDefiningOp();
    if (!rootOp) {
      return {nullptr, leafNodes};
    }

    llvm::errs() << "BOT " << dbgCnt++ << "\n";

    // Get the block containing the affine loop operations.
    Block *affineBlock = innermostLoop.getBody();

    llvm::errs() << "BOT " << dbgCnt++ << "\n";

    // Build the tree recursively starting from the root operation.
    auto rootNode = buildOpTreeRecursive(rootOp, affineBlock, leafNodes);
    return {std::move(rootNode), leafNodes};
  }

  // Recursive helper to build the OpTreeNode from an operation.
  // Works with affine loop blocks, treating affine.load operations as leaf nodes.
  static std::unique_ptr<OpTreeNode>
  buildOpTreeRecursive(Operation *op, Block *affineBlock,
                       SmallVector<OpTreeNode *> &leafNodes) {

    int dbgCnt = 0;
    // Create the node for this operation.
    auto node = std::make_unique<OpTreeNode>(op);

    llvm::errs() << "BOT RRR " << dbgCnt++ << "\n";

    // Process operands to build child nodes.
    SmallVector<std::unique_ptr<OpTreeNode>> children;

    for (Value operand : op->getOperands()) {
      llvm::errs() << "BOT RRR " << dbgCnt++ << "\n";
      if (Operation *definingOp = operand.getDefiningOp()) {
        llvm::errs() << "BOT RRR " << dbgCnt++ << "\n";
        // If operand is an affine.load, treat it as a leaf node.
        if (mlir::isa<affine::AffineLoadOp>(definingOp)) {
          llvm::errs() << "BOT RRR " << dbgCnt++ << "\n";
          auto leafNode = std::make_unique<OpTreeNode>(definingOp);
          leafNodes.push_back(leafNode.get());
          children.push_back(std::move(leafNode));
        }
        // If operand is defined by a compute operation in the affine block, recurse.
        else if (definingOp->getBlock() == affineBlock) {
          llvm::errs() << "BOT RRR " << dbgCnt++ << "\n";
          children.push_back(
              buildOpTreeRecursive(definingOp, affineBlock, leafNodes));
        }
      }
    }
    llvm::errs() << "BOT RRR " << dbgCnt++ << "\n";

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
    markNodeWeights(node->right);

    // Calculate weight for this node.
    if (!node->left && !node->right) {
      // Leaf node (block argument).
      node->weight = 1;
    } else if (node->right == nullptr) {
      // Unary operation.
      assert(node->left && "Unary operation must have left child");
      node->weight = node->left->weight;
    } else {
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
      llvm::errs() << indent << "├─ BlockArg #" << node->blockArg.getArgNumber();
      llvm::errs() << " | Weight: " << node->weight << " [LEAF]";
    } else if (node->op && mlir::isa<affine::AffineLoadOp>(node->op)) {
      // This is an affine.load leaf node.
      llvm::errs() << indent << "├─ Op: " << node->op->getName();
      llvm::errs() << " | Weight: " << node->weight << " [LEAF]";
    } else {
      // This is an operation node.
      llvm::errs() << indent << "├─ Op: " << node->op->getName();
      llvm::errs() << " | Weight: " << node->weight;
    }
    llvm::errs() << "\n";

    // Print children with appropriate labels.
    if (node->left || node->right) {
      if (node->left) {
        llvm::errs() << indent << "  Left:\n";
        printOpTree(node->left, depth + 1);
      }
      if (node->right) {
        llvm::errs() << indent << "  Right:\n";
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

    // Determine traversal order based on child weights.
    // Higher weight branch is visited first. If weights are equal, left is default.
    OpTreeNode *firstChild = nullptr;
    OpTreeNode *secondChild = nullptr;
    
    if (node->left && node->right) {
      // Binary operation: choose based on weight.
      if (node->right->weight > node->left->weight) {
        firstChild = node->right;
        secondChild = node->left;
      } else {
        // Left child first if weights are equal or left is heavier.
        firstChild = node->left;
        secondChild = node->right;
      }
    } else if (node->left) {
      // Unary operation: only left child.
      firstChild = node->left;
    } // right child is not used unless op is binary

    // Recursively process children depth-first.
    if (firstChild) {
      applySeethiUllmanOrdering(rewriter, firstChild, movedOps);
    }
    if (secondChild) {
      applySeethiUllmanOrdering(rewriter, secondChild, movedOps);
    }

    // Check if this is a leaf node (affine.load).
    bool isLeaf = mlir::isa<affine::AffineLoadOp>(node->op);
    
    // Check if all children have been moved (for non-leaf nodes).
    bool allChildrenMoved = true;
    if (node->left && node->left->op) {
      allChildrenMoved &= movedOps.contains(node->left->op);
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
  static void reorderOperationsSethi(PatternRewriter &rewriter,
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

    llvm::errs() << "\n=== Sethi-Ullman Reordering Applied ===\n";
  }

  // Insert DST register loads/stores (simplified version - all use index 0).
  // Called after Sethi-Ullman reordering has arranged operations optimally.
  static void insertDstAccessAndAllocation(PatternRewriter &rewriter,
                                            Operation *outermostLoop,
                                            unsigned dstCapacity,
                                            Location loc) {
    if (!outermostLoop) {
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

    Block *innermostBody = innermostLoop.getBody();
    
    // Collect loop induction variables for DST indexing.
    SmallVector<Value> loopInductionVars;
    Operation *currentLoop = innermostLoop;
    while (auto forOp = dyn_cast<affine::AffineForOp>(currentLoop)) {
      loopInductionVars.insert(loopInductionVars.begin(), 
                               forOp.getInductionVar());
      currentLoop = forOp->getParentOp();
    }

    // First pass: determine type and create DST memref.
    MemRefType dstType = nullptr;
    for (Operation &op : *innermostBody) {
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(&op)) {
        auto memrefType = loadOp.getMemRefType();
        if (!dstType) {
          // Build DST type: [dstCapacity, tile_dims...]
          SmallVector<int64_t> dstShape;
          dstShape.push_back(dstCapacity);
          dstShape.append(memrefType.getShape().begin(), 
                          memrefType.getShape().end());
          dstType = MemRefType::get(
              dstShape, memrefType.getElementType(),
              AffineMap::getMultiDimIdentityMap(dstShape.size(), 
                                                rewriter.getContext()),
              rewriter.getAttr<ttcore::MemorySpaceAttr>(
                  ttcore::MemorySpace::RegisterDst));
        }
        break;
      }
    }

    if (!dstType) {
      llvm::errs() << "Warning: Could not infer DST type\n";
      return;
    }

    // Create DST memref (acquire_dst) BEFORE the outermostLoop.
    rewriter.setInsertionPoint(outermostLoop);
    llvm::errs() << "\n=== Creating AcquireDstOp before outermostLoop ===\n";
    auto acquireDst = rewriter.create<AcquireDstOp>(loc, dstType);
    Value dst = acquireDst.getResult();
    llvm::errs() << "AcquireDstOp created successfully\n";

    llvm::errs() << "\n=== Starting DST Access Insertion (Index 0 Only) ===\n";
    llvm::errs() << "DST Capacity: " << dstCapacity << " slices\n";

    // Build AffineMap for DST access with constant 0 embedded.
    // Creates map: affine_map<(d0, d1, ...) -> (0, d0, d1, ...)>
    // This is the idiomatic affine dialect way - constants in the map, not as SSA values.
    unsigned numLoopVars = loopInductionVars.size();
    SmallVector<AffineExpr> exprs;
    exprs.push_back(getAffineConstantExpr(0, rewriter.getContext())); // First dimension = 0
    for (unsigned i = 0; i < numLoopVars; ++i) {
      exprs.push_back(getAffineDimExpr(i, rewriter.getContext())); // d0, d1, ...
    }
    AffineMap dstAccessMap = AffineMap::get(numLoopVars, 0, exprs, rewriter.getContext());
    
    llvm::errs() << "Created DST access map: " << dstAccessMap << "\n";

    llvm::errs() << "\n=== Building new loop with DST access ===\n";
    llvm::errs() << "Innermost body has " << innermostBody->getOperations().size() << " operations\n";
    
    // Strategy: Clone the loop structure, rebuild the body with DST access, replace original.
    // This avoids modifying the block we're iterating over.
    
    // Step 1: Collect original operations (before any modifications).
    SmallVector<Operation *> originalOps;
    for (Operation &op : *innermostBody) {
      // Skip the terminator (affine.yield).
      if (!isa<affine::AffineYieldOp>(&op)) {
        originalOps.push_back(&op);
      }
    }
    llvm::errs() << "Collected " << originalOps.size() << " operations to process\n";
    
    // Step 2: Clone the loop nest to get a fresh structure with empty body.
    mlir::IRMapping irMapper;
    rewriter.setInsertionPointAfter(outermostLoop);
    Operation *clonedOuterLoop = rewriter.clone(*outermostLoop, irMapper);
    
    // Find the cloned innermost loop.
    affine::AffineForOp clonedInnermostLoop = nullptr;
    clonedOuterLoop->walk([&](affine::AffineForOp loop) {
      clonedInnermostLoop = loop;
      return WalkResult::interrupt();
    });
    
    if (!clonedInnermostLoop) {
      llvm::errs() << "ERROR: Could not find cloned innermost loop\n";
      rewriter.eraseOp(clonedOuterLoop);
      return;
    }
    
    Block *clonedBody = clonedInnermostLoop.getBody();
    
    // Erase all operations in cloned body except the terminator.
    SmallVector<Operation *> opsToErase;
    for (Operation &op : *clonedBody) {
      if (!isa<affine::AffineYieldOp>(&op)) {
        opsToErase.push_back(&op);
      }
    }
    // Drop all uses first, then erase (handles circular dependencies).
    for (Operation *op : opsToErase) {
      op->dropAllUses();
    }
    for (Operation *op : opsToErase) {
      rewriter.eraseOp(op);
    }
    
    llvm::errs() << "Cloned loop structure with empty body\n";
    
    // Get cloned loop induction variables.
    SmallVector<Value> clonedLoopVars;
    currentLoop = clonedInnermostLoop;
    while (auto forOp = dyn_cast<affine::AffineForOp>(currentLoop)) {
      clonedLoopVars.insert(clonedLoopVars.begin(), 
                            forOp.getInductionVar());
      currentLoop = forOp->getParentOp();
    }
    
    // Step 3: Rebuild operations in cloned body with DST access.
    rewriter.setInsertionPointToStart(clonedBody);
    
    // Track mapping from original values to new DST-loaded values.
    DenseMap<Value, Value> valueMapping;
    
    llvm::errs() << "Rebuilding operations with DST access...\n";
    
    for (Operation *origOp : originalOps) {
      // Handle affine.load from L1 memory.
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(origOp)) {
        auto memspace = ttcore::getMemorySpace(loadOp.getMemRef());
        if (memspace != ttcore::MemorySpace::RegisterDst) {
          llvm::errs() << "  Rebuilding L1 load with DST access\n";
          
          // Map indices from original loop vars to cloned loop vars.
          SmallVector<Value> mappedIndices;
          for (Value idx : loadOp.getIndices()) {
            mappedIndices.push_back(irMapper.lookupOrDefault(idx));
          }
          
          // Clone the original load (loads from L1) with mapped indices.
          auto newLoad = rewriter.create<affine::AffineLoadOp>(
              loc, loadOp.getMemRef(), loadOp.getMap(), mappedIndices);
          
          // Store to DST.
          rewriter.create<affine::AffineStoreOp>(
              loc, newLoad.getResult(), dst, dstAccessMap, clonedLoopVars);
          
          // Load from DST.
          auto dstLoad = rewriter.create<affine::AffineLoadOp>(
              loc, dst, dstAccessMap, clonedLoopVars);
          
          // Map original result to DST-loaded value.
          valueMapping[loadOp.getResult()] = dstLoad.getResult();
        }
      }
      // Handle compute ops.
      else if (origOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
        llvm::errs() << "  Rebuilding compute op: " << origOp->getName() << "\n";
        
        // Map operands from original to new values.
        SmallVector<Value> newOperands;
        for (Value operand : origOp->getOperands()) {
          if (valueMapping.count(operand)) {
            newOperands.push_back(valueMapping[operand]);
          } else {
            newOperands.push_back(operand);
          }
        }
        
        // Clone the compute op with mapped operands.
        auto newComputeOp = rewriter.create(
            loc, origOp->getName().getIdentifier(), newOperands,
            origOp->getResultTypes(), origOp->getAttrs());
        
        // Store result to DST.
        rewriter.create<affine::AffineStoreOp>(
            loc, newComputeOp->getResult(0), dst, dstAccessMap, clonedLoopVars);
        
        // Load result from DST.
        auto dstLoad = rewriter.create<affine::AffineLoadOp>(
            loc, dst, dstAccessMap, clonedLoopVars);
        
        // Map original result to DST-loaded value.
        valueMapping[origOp->getResult(0)] = dstLoad.getResult();
      }
      // Handle affine.store to L1 memory.
      else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(origOp)) {
        auto memspace = ttcore::getMemorySpace(storeOp.getMemRef());
        if (memspace != ttcore::MemorySpace::RegisterDst) {
          llvm::errs() << "  Rebuilding L1 store with DST load\n";
          
          // Get the value to store (should be mapped to DST value).
          Value valueToStore = storeOp.getValue();
          if (valueMapping.count(valueToStore)) {
            valueToStore = valueMapping[valueToStore];
          }
          
          // Map indices from original loop vars to cloned loop vars.
          SmallVector<Value> mappedIndices;
          for (Value idx : storeOp.getIndices()) {
            mappedIndices.push_back(irMapper.lookupOrDefault(idx));
          }
          
          // Store to L1 with mapped indices.
          rewriter.create<affine::AffineStoreOp>(
              loc, valueToStore, storeOp.getMemRef(), 
              storeOp.getMap(), mappedIndices);
        }
      }
    }
    
    llvm::errs() << "Cloned loop body rebuilt successfully\n";
    
    // Step 4: Replace original loop with cloned loop.
    // Erase original loop by walking bottom-up and erasing all operations.
    SmallVector<Operation *> originalOpsToErase;
    outermostLoop->walk([&](Operation *op) {
      originalOpsToErase.push_back(op);
    });
    
    // Drop all uses, then erase in reverse order.
    for (Operation *op : originalOpsToErase) {
      op->dropAllUses();
    }
    for (Operation *op : llvm::reverse(originalOpsToErase)) {
      rewriter.eraseOp(op);
    }
    
    llvm::errs() << "\n=== DST Access Insertion Complete ===\n";
  }

  static bool hasAcquireDstOp(Region &region) {
    bool hasAcquire = !region.getOps<AcquireDstOp>().empty();
    llvm::errs() << "  hasAcquireDstOp check: " << (hasAcquire ? "YES" : "NO") << "\n";
    return hasAcquire;
  }
  
  static bool hasDstMemoryAccess(Region &region) {
    // Check if there are any loads or stores to DST memory space
    bool hasDstAccess = false;
    region.walk([&](Operation *op) {
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
        if (ttcore::getMemorySpace(loadOp.getMemRef()) == ttcore::MemorySpace::RegisterDst) {
          hasDstAccess = true;
          return WalkResult::interrupt();
        }
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        if (ttcore::getMemorySpace(storeOp.getMemRef()) == ttcore::MemorySpace::RegisterDst) {
          hasDstAccess = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    llvm::errs() << "  hasDstMemoryAccess check: " << (hasDstAccess ? "YES" : "NO") << "\n";
    return hasDstAccess;
  }

  template <typename OpT>
  using OpAndIndexOffset = std::pair<OpT, int64_t>;

  // Stores dst loads/stores, organized by common loop nests.
  struct CopyInfo {
    void push_back(affine::AffineLoadOp load, int64_t indexOffset) {
      loads.emplace_back(load, indexOffset);
    }

    void push_back(affine::AffineStoreOp store, int64_t indexOffset) {
      stores.emplace_back(store, indexOffset);
    }

    SmallVector<int64_t> guardIndices;
    SmallVector<OpAndIndexOffset<affine::AffineLoadOp>> loads;
    SmallVector<OpAndIndexOffset<affine::AffineStoreOp>> stores;
  };
  using CopyInfoMap = DenseMap<Operation *, CopyInfo>;

  class DstSliceAllocationState {
  public:
    int64_t allocate() { return nextSliceIndex++; }

    void setStoreToDst() { storedToDst = true; }
    bool didStoreToDst() { return storedToDst; }
    int64_t getCurrSliceIndex() { return nextSliceIndex - 1; }

  private:
    int64_t nextSliceIndex = 0;
    bool storedToDst = false;
  };

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    llvm::errs() << "\n=== matchAndRewrite called on GenericOp ===\n";
    llvm::errs() << "GenericOp has " << op.getNumRegions() << " regions\n";
    
    // Early check: if any region already has DST ops, skip this entire GenericOp.
    for (unsigned regionIndex = 0; regionIndex < op.getNumRegions();
         regionIndex++) {
      if (op.getRegionThreadType(regionIndex) != ThreadType::Compute) {
        llvm::errs() << "  Region " << regionIndex << ": not compute, skipping\n";
        continue;
      }
      llvm::errs() << "  Checking region " << regionIndex << " for DST ops:\n";
      Region &region = op.getRegion(regionIndex);
      
      bool hasAcquire = hasAcquireDstOp(region);
      bool hasDstAccess = hasDstMemoryAccess(region);
      
      if (hasAcquire || hasDstAccess) {
        llvm::errs() << "!!! Region " << regionIndex << " already has DST ops (acquire=" 
                     << hasAcquire << ", access=" << hasDstAccess 
                     << "), returning failure to prevent reprocessing !!!\n";
        return failure();  // Don't process this op again
      }
    }
    
    llvm::errs() << "No DST ops found in any region, proceeding with processing\n";
    
    bool modified = false;
    for (unsigned regionIndex = 0; regionIndex < op.getNumRegions();
         regionIndex++) {
      if (op.getRegionThreadType(regionIndex) != ThreadType::Compute) {
        continue;
      }

      Region &region = op.getRegion(regionIndex);
      Block &block = region.getBlocks().front();

      Type largestDstType = utils::getRegionLargestDstElemType(region);
      unsigned dstCapacity =
          ttcore::getOpChipDescAttr(op).getDstLogicalSizeTiles(
              largestDstType, false, maxDstPhysicalSizeTiles);

      // Temporary START
      // TODO(mbagherbeik) remove when fixing fusions/dstAllocs
      
      // NumOperands includes the output/init operand ( > vs >= )
      // halve the capacity if we have a binary fused op
      if (op->getNumOperands() > 2) {
        dstCapacity /= 2;
      }
      // halve capacity again if fused op is ternary or more
      if (op->getNumOperands() > 3) {
        dstCapacity /= 2;
      }
      // halve capacity again if data type is more than 16 bits
      if (largestDstType.getIntOrFloatBitWidth() > 16) {
        dstCapacity /= 2;
      }
      // Temporary END

      bool linalgToAffineFailed = false;
      block.walk([&](linalg::GenericOp linalgGenericOp) {
        if (!useTileMatmul && hasTileMatmul(linalgGenericOp)) {
          linalgToAffineFailed |= rewriteTileMatmulAsTileMatmulBlock(
              rewriter, op, region, linalgGenericOp, dstCapacity, modified);
          return;
        }

        rewriter.setInsertionPoint(linalgGenericOp);
        // Apply linalg to affine loops pass.
        auto linalgLoops =
            linalg::linalgOpToAffineLoops(rewriter, linalgGenericOp);
        if (failed(linalgLoops)) {
          linalgToAffineFailed = true;
          return;
        }
        rewriter.eraseOp(linalgGenericOp);


        // if all parallel --> no reductions --> eltwise only
        bool isEltwiseOnly = op.isAllParallel();
        // doesn't contain skippable eltwise ops (i.e. went through eltwise fusion)
        bool noSkip = !op.hasSkipOpEltwiseFusionTrait();

        int dbgCnt = 0;

        llvm::errs() << dbgCnt++ << "\n";

        if (isEltwiseOnly && noSkip) {
          // NEW: Try tree-based approach with Sethi-Ullman register allocation.
          // Now working with affine loops instead of linalg.generic.
          Operation *outermostLoop = !linalgLoops.value().empty() 
                                      ? linalgLoops.value().front() 
                                      : nullptr;

          
          llvm::errs() << dbgCnt++ << "\n";

          
          if (outermostLoop) {
            llvm::errs() << dbgCnt++ << "\n";
            auto [opTreeRoot, leafNodes] = buildOpTree(outermostLoop);
            llvm::errs() << dbgCnt++ << "\n";
            if (opTreeRoot) {
              llvm::errs() << dbgCnt++ << "\n";
              // Mark node weights using Sethi-Ullman algorithm.
              markNodeWeights(opTreeRoot.get());

              llvm::errs() << "\n=== Operation Tree (Sethi-Ullman) ===\n";
              printOpTree(opTreeRoot.get());
              llvm::errs() << "=====================================\n\n";

              // Apply Sethi-Ullman ordering to reorder operations.
              reorderOperationsSethi(rewriter, outermostLoop, opTreeRoot.get());

              // Insert DST register loads/stores and perform allocation.
              insertDstAccessAndAllocation(rewriter, outermostLoop, dstCapacity,
                                           op.getLoc());
              
              modified = true;
            }
          }
        }
        else {
          // original allocator
          modified |= insertDstRegisterAccess(rewriter, op, region, dstCapacity,
                                              !linalgLoops.value().empty()
                                                  ? linalgLoops.value().front()
                                                  : nullptr);
        }
      });
      if (linalgToAffineFailed) {
        llvm::errs() << "=== Linalg to affine failed, returning failure ===\n";
        return failure();
      }
    }
    llvm::errs() << "=== matchAndRewrite complete, modified=" << modified << " ===\n";
    return success(modified);
  }

  static bool
  insertDstRegisterAccess(PatternRewriter &rewriter, GenericOp op,
                          Region &region, unsigned dstCapacity,
                          Operation *outermostInnerComputeLoop = nullptr) {
    assert(region.getBlocks().size() == 1);
    if (hasAcquireDstOp(region)) {
      return false;
    }

    Location loc = op.getLoc();

    // FALLBACK: Original approach.
    // 1. Collect all loads/stores to dst organized by loop nest.
    auto [copyInfos, dstAllocation] =
        collectDstAccesses(op, region, outermostInnerComputeLoop);
    if (copyInfos.empty()) {
      return false;
    }

    // 2. Insert acquire dst.
    AcquireDstOp acquireDst =
        insertAcquireDst(rewriter, loc, region, copyInfos,
                         outermostInnerComputeLoop, dstCapacity);
    Value dst = acquireDst.getResult();

    // 3. Generate data copy loops to/from dst and output cb.
    dataCopyGenerate(rewriter, loc, dst, copyInfos);

    // 4. Rewrite stores to use dst register based on allocation.
    insertDstRegisterAllocation(rewriter, loc, dst, dstAllocation);

    return true;
  }

  static std::pair<MemRefType, int64_t>
  inferCbInfoFromAllAccesses(const CopyInfoMap &copyInfos) {
    MemRefType canonicalType = nullptr;
    int64_t maxDstSliceIdx = -1;

    for (auto [loopNest, copyInfo] : copyInfos) {
      for (auto &[loadOp, idx] : copyInfo.loads) {
        if (canonicalType == nullptr) {
          canonicalType = loadOp.getMemRefType();
        } else {
          TT_assertv(loadOp.getMemRefType().getShape() ==
                         canonicalType.getShape(),
                     "Multiple interpretations of DST not supported.");
        }
        maxDstSliceIdx = std::max(maxDstSliceIdx, idx);
      }
      for (auto &[storeOp, idx] : copyInfo.stores) {
        if (canonicalType == nullptr) {
          canonicalType = storeOp.getMemRefType();
        } else {
          TT_assertv(storeOp.getMemRefType().getShape() ==
                         canonicalType.getShape(),
                     "Multiple interpretations of DST not supported.");
        }
        maxDstSliceIdx = std::max(maxDstSliceIdx, idx);
      }
    }
    TT_assert(canonicalType != nullptr);
    TT_assert(maxDstSliceIdx >= 0);
    return {canonicalType, maxDstSliceIdx};
  }

  static AcquireDstOp insertAcquireDst(PatternRewriter &rewriter, Location loc,
                                       Region &region,
                                       const CopyInfoMap &copyInfos,
                                       Operation *outermostInnerComputeLoop,
                                       unsigned dstCapacity) {
    assert(!copyInfos.empty());
    if (outermostInnerComputeLoop) {
      rewriter.setInsertionPoint(outermostInnerComputeLoop);
    } else {
      rewriter.setInsertionPointToStart(&region.front());
    }

    auto [cbType, maxDstSliceIdx] = inferCbInfoFromAllAccesses(copyInfos);
    // Calculate dst shape as N slices of cb shape.
    const int64_t volume = ttmlir::utils::volume(cbType.getShape());
    TT_assert(volume <= dstCapacity);
    const int64_t numDstSlices = dstCapacity / volume;
    TT_assertv(maxDstSliceIdx < numDstSlices,
               "Insufficient DST capacity for all operands.");
    SmallVector<int64_t> dstShape({numDstSlices});
    dstShape.append(cbType.getShape().begin(), cbType.getShape().end());
    MemRefType dstType =
        MemRefType::get(dstShape, cbType.getElementType(),
                        mlir::AffineMap::getMultiDimIdentityMap(
                            dstShape.size(), rewriter.getContext()),
                        rewriter.getAttr<ttcore::MemorySpaceAttr>(
                            ttcore::MemorySpace::RegisterDst));

    return rewriter.create<AcquireDstOp>(loc, dstType);
  }

  // Walk all compute ops in the region and collect all dst accesses organized
  // by loop nest. Also maintain dst register allocation state such that
  // multiple operands get unique dst indices. Currently this routine only does
  // register allocation for loads and just assumes that stores get exclusive
  // access. Returns a map of loop nest -> copy info, which contains a list of
  // loads and stores to copy into hoisted loop nests.

  // Maps each D2MGenericRegionComputeOpTrait operation result to a dest
  // register slice index and its containing loop nest.
  struct DstRegisterInfo {
    int64_t dstSliceIndex;
    Operation *outermostLoop;
  };
  using DstRegisterAllocation = DenseMap<Operation *, DstRegisterInfo>;

  // Struct to hold the results of dst access collection.
  struct DstAccessCollection {
    CopyInfoMap copyInfos;
    DstRegisterAllocation dstAllocation;
  };

  // Return both the copy nest info and dst allocation info.
  static DstAccessCollection
  collectDstAccesses(GenericOp op, Region &region,
                     Operation *outermostInnerComputeLoop) {
    CopyInfoMap copyInfos;
    DstSliceAllocationState dstSliceAllocationState;
    DstRegisterAllocation dstRegisterAllocation;
    region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
      // We're generating loads and stores for dst, so we can ignore loads and
      // stores that are already on dst.
      auto notDstMemspace = [](auto op) {
        return op && ttcore::getMemorySpace(op.getMemRef()) !=
                         ttcore::MemorySpace::RegisterDst;
      };

      // Collect loads to this op.
      for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
        if (auto potentialLoad = computeOp->getOperand(operandIdx)
                                     .getDefiningOp<affine::AffineLoadOp>();
            notDstMemspace(potentialLoad)) {
          collectDstAccess<affine::AffineLoadOp>(
              op, potentialLoad, copyInfos, dstSliceAllocationState.allocate(),
              outermostInnerComputeLoop);
        }
      }

      // Collect stores from this op.
      for (auto *user : computeOp->getUsers()) {
        if (auto potentialStore = mlir::dyn_cast<affine::AffineStoreOp>(user);
            notDstMemspace(potentialStore)) {

          assert(!dstSliceAllocationState.didStoreToDst() &&
                 "Multiple stores from last op to dst not supported");

          auto dstRegInPlace = computeOp.getDstRegInPlace();
          int64_t dstSliceIndex = -1;
          if (dstRegInPlace) {
            bool isUnaryOp = computeOp->getNumOperands() == 1;
            bool isTileMatmul = mlir::isa<d2m::TileMatmulOp>(computeOp);
            bool isReduction = mlir::isa<d2m::TileReduceMaxOp>(computeOp) ||
                               mlir::isa<d2m::TileReduceSumOp>(computeOp);
            assert((isUnaryOp || isTileMatmul || isReduction) &&
                   "Only unary ops, tile matmul, and reductions supported for "
                   "destination register in "
                   "place, multi-operand ops would reference wrong tile, but "
                   "those ops should be setting output tile.");
            dstSliceIndex = dstSliceAllocationState.getCurrSliceIndex();
          } else {
            dstSliceIndex = dstSliceAllocationState.allocate();
            dstSliceAllocationState.setStoreToDst();
          }
          collectDstAccess<affine::AffineStoreOp>(op, potentialStore, copyInfos,
                                                  dstSliceIndex,
                                                  outermostInnerComputeLoop);

        }
        // If the user isn't a store, it must be another compute consumer and we
        // need to set or allocate a dest register intermediate for it.
        else {
          assert(user->hasTrait<D2MGenericRegionComputeOpTrait>());
          assert(computeOp->hasOneUse() &&
                 "Currently we do not support multiple "
                 "users in the same compute dst region.");
          assert(computeOp->getNumResults() == 1);
          assert(!dstRegisterAllocation.contains(computeOp));
          // If op stores to dst in place, we don't need to allocate a new dst
          // register, just use the current dst index.
          int32_t allocatedIndex =
              computeOp.getDstRegInPlace()
                  ? dstSliceAllocationState.getCurrSliceIndex()
                  : dstSliceAllocationState.allocate();

          dstRegisterAllocation[computeOp] = {allocatedIndex,
                                              outermostInnerComputeLoop};
        }
      }
    });
    return {copyInfos, dstRegisterAllocation};
  }

  static BlockArgument lookThroughSubView(Value memref) {
    while (auto subView = mlir::dyn_cast_or_null<memref::SubViewOp>(
               memref.getDefiningOp())) {
      memref = subView.getSource();
    }
    if (auto *definingOp = memref.getDefiningOp();
        mlir::isa_and_nonnull<d2m::WaitOp, d2m::ReserveOp>(definingOp)) {
      memref = definingOp->getOperand(0);
    }
    return mlir::cast<BlockArgument>(memref);
  }

  // Collect a single load or store to dst organized by loop nest.
  template <typename LoadOrStoreOp>
  static void collectDstAccess(GenericOp op, LoadOrStoreOp loadOrStore,
                               CopyInfoMap &copyInfos,
                               int64_t nextDstSliceIndex,
                               Operation *outermostInnerComputeLoop) {
    if (!outermostInnerComputeLoop) {
      // If there is no outermostInnerComputeLoop, the common ancestor is the
      // operation itself.
      outermostInnerComputeLoop = loadOrStore;
    }

    auto [iter, inserted] = copyInfos.try_emplace(outermostInnerComputeLoop);
    CopyInfo &copyInfo = iter->second;
    copyInfo.push_back(loadOrStore, nextDstSliceIndex);
    SmallVector<int64_t> guardIndices = op.getNonParticipatingLoopDims(
        lookThroughSubView(loadOrStore.getMemRef()).getArgNumber());
    if (inserted) {
      // First access in this loop nest - set the guard indices.
      copyInfo.guardIndices = guardIndices;
    } else {
      // Subsequent access - verify guard indices are the same.
      assert(
          guardIndices == copyInfo.guardIndices &&
          "Expected same guard indices across all accesses in this loop nest.");
    }
  }

  static bool hasTileMatmul(linalg::GenericOp linalgGenericOp) {
    bool hasTileMatmul = false;
    linalgGenericOp->walk([&](d2m::TileMatmulOp) {
      hasTileMatmul = true;
      return WalkResult::interrupt();
    });
    return hasTileMatmul;
  }
  /*
    Expand a linalg.generic op that contains a tile_matmul into a
    tile_matmul_block.

    - Uses the linalg.generic and affine semantics to generate copy/pack loops.
    - Deletes the compute loop nest since tile_matmul_block includes the loops
    inside it.
  */
  static bool rewriteTileMatmulAsTileMatmulBlock(
      PatternRewriter &rewriter, GenericOp op, Region &region,
      linalg::GenericOp linalgGenericOp, unsigned dstCapacity, bool &modified) {
    assert(linalgGenericOp.getInputs().size() == 2 &&
           "Expected exactly 2 input for tile matmul");
    assert(linalgGenericOp.getOutputs().size() == 1 &&
           "Expected exactly 1 output for tile matmul");

    Value inputAMemref = linalgGenericOp.getInputs()[0];
    Value inputBMemref = linalgGenericOp.getInputs()[1];
    Value outputCMemref = linalgGenericOp.getOutputs()[0];

    rewriter.setInsertionPoint(linalgGenericOp);

    auto linalgLoops = linalg::linalgOpToAffineLoops(rewriter, linalgGenericOp);
    if (failed(linalgLoops)) {
      return false;
    }
    rewriter.eraseOp(linalgGenericOp);
    modified |= insertDstRegisterAccess(
        rewriter, op, region, dstCapacity,
        !linalgLoops.value().empty() ? linalgLoops.value().front() : nullptr);

    Operation *outerLoop = linalgLoops.value()[0];
    Block *parentBlk = outerLoop->getBlock();
    auto insertPos = std::next(Block::iterator(outerLoop));

    rewriter.setInsertionPoint(parentBlk, insertPos);
    for (Operation *loopOp : llvm::reverse(linalgLoops.value())) {
      rewriter.eraseOp(loopOp);
    }
    rewriter.create<d2m::TileMatmulBlockOp>(op.getLoc(), inputAMemref,
                                            inputBMemref, outputCMemref);
    return true;
  }

  static void dataCopyGenerate(PatternRewriter &rewriter, Location loc,
                               Value dst, const CopyInfoMap &copyInfos) {
    for (const auto &[loopNestOrOp, copyInfo] : copyInfos) {
      // Save this insertion point as loopNestOrOp may be replaced.
      rewriter.setInsertionPointAfter(loopNestOrOp);
      auto insertionPointAfterLoopNest = rewriter.saveInsertionPoint();

      rewriter.setInsertionPoint(loopNestOrOp);
      auto guard = insertGuardForLoopNest(rewriter, loc, copyInfo.guardIndices);
      if (guard) {
        rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
      }
      dataCopyGenerate<affine::AffineLoadOp>(
          rewriter, loopNestOrOp, copyInfo.loads,
          // Load/store dst access generation.
          [&](
            PatternRewriter &rewriter,
            Location loc,
            Value cb,
            AffineMap l1AccessMap,
            ValueRange l1AccessIndices,
            AffineMap dstAccessMap, 
            ValueRange dstAccessIndices) {
              auto l1Load = rewriter.create<affine::AffineLoadOp>(
                loc, 
                cb, 
                l1AccessMap, 
                l1AccessIndices
              );
              rewriter.create<affine::AffineStoreOp>(
                loc, 
                l1Load.getResult(), 
                dst, 
                dstAccessMap, 
                dstAccessIndices
              );
          },
          // Replacement of the original load with one from dst.
          [&](PatternRewriter &rewriter, affine::AffineLoadOp op,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
                op, dst, dstAccessMap, dstAccessIndices);
          });

      rewriter.restoreInsertionPoint(insertionPointAfterLoopNest);
      dataCopyGenerate<affine::AffineStoreOp>(
          rewriter, loopNestOrOp, copyInfo.stores,
          // Load/store dst access generation.
          [&](PatternRewriter &rewriter, Location loc, Value cb,
              AffineMap l1AccessMap, ValueRange l1AccessIndices,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            auto dstLoad = rewriter.create<affine::AffineLoadOp>(
                loc, dst, dstAccessMap, dstAccessIndices);
            Value valueToStore = dstLoad.getResult();

            // Insert dst reinterpret cast if destination CB type differs
            // from dst type
            auto cbType = mlir::cast<MemRefType>(cb.getType());
            if (valueToStore.getType() != cbType.getElementType()) {
              valueToStore = rewriter
                                 .create<d2m::DstReinterpretCastOp>(
                                     loc, cbType.getElementType(), valueToStore)
                                 .getResult();
            }

            rewriter.create<affine::AffineStoreOp>(
                loc, valueToStore, cb, l1AccessMap, l1AccessIndices);
          },
          // Replacement of the original store with one from dst.
          [&](PatternRewriter &rewriter, affine::AffineStoreOp op,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            Value valueToStore = op.getValue();
            // Insert dst reinterpret cast if value type differs from dst
            // type
            auto dstType = mlir::cast<MemRefType>(dst.getType());
            if (valueToStore.getType() != dstType.getElementType()) {
              valueToStore =
                  rewriter
                      .create<d2m::DstReinterpretCastOp>(
                          op.getLoc(), dstType.getElementType(), valueToStore)
                      .getResult();
            }
            rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
                op, valueToStore, dst, dstAccessMap, dstAccessIndices);
          });
    }
  }

  static scf::IfOp insertGuardForLoopNest(PatternRewriter &rewriter,
                                          Location loc,
                                          ArrayRef<int64_t> guardIndices) {
    if (guardIndices.empty()) {
      return nullptr;
    }
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    auto cmp = rewriter
                   .create<arith::ConstantOp>(loc, rewriter.getI1Type(),
                                              rewriter.getBoolAttr(false))
                   .getResult();
    for (int64_t index : guardIndices) {
      auto iterIndex = rewriter.create<d2m::IterIndexOp>(loc, index);
      auto eq = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                               iterIndex, zero);
      cmp = rewriter.create<arith::OrIOp>(loc, cmp, eq).getResult();
    }
    return rewriter.create<scf::IfOp>(loc, cmp);
  }

  template <typename LoadStoreOpTy>
  static void dataCopyGenerate(
      PatternRewriter &rewriter, Operation *loopNestOrOp,
      ArrayRef<OpAndIndexOffset<LoadStoreOpTy>> loadStoreOps,
      llvm::function_ref<void(PatternRewriter &, Location, Value, AffineMap,
                              ValueRange, AffineMap, ValueRange)>
          loadStoreDstAccessGenerator,
      llvm::function_ref<void(PatternRewriter &, LoadStoreOpTy, AffineMap,
                              ValueRange)>
          dstAccessReplacement) {
    if (loadStoreOps.empty()) {
      return;
    }

    mlir::IRMapping irMapper;
    // Only Clone loop nests if a loop exists.
    if (mlir::isa<affine::AffineForOp>(loopNestOrOp)) {
      rewriter.clone(*loopNestOrOp, irMapper)->walk([&](Operation *op) {
        // Erase the loop bodies except for other nested loops / yields.
        if (!mlir::isa<affine::AffineForOp, affine::AffineYieldOp,
                       affine::AffineApplyOp>(op)) {
          op->dropAllUses();
          rewriter.eraseOp(op);
        }
      });
    }

    for (auto [loadStore, dstSliceIndex] : loadStoreOps) {
      Block *fromScope = loadStore->getBlock();
      Block *toScope = irMapper.lookupOrNull(fromScope);
      if (toScope) {
        Operation *terminator = toScope->getTerminator();
        if (terminator) {
          rewriter.setInsertionPoint(terminator);
        } else {
          rewriter.setInsertionPointToEnd(toScope);
        }
      }

      // Generate the data copy loop for the load store.
      {
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStore.getLoc(), irMapper,
                         loadStore.getIndices(), dstSliceIndex,
                         loadStore.getMap());
        loadStoreDstAccessGenerator(
            rewriter, loadStore.getLoc(), loadStore.getMemRef(), l1AccessMap,
            l1AccessIndices, dstAccessMap, dstAccessIndices);
      }

      // Replace the original load store with one from dst.
      {
        // Empty IR mapper because we want to preserve original loop vars.
        mlir::IRMapping dummyIRMapper;
        rewriter.setInsertionPoint(loadStore);
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStore.getLoc(), dummyIRMapper,
                         loadStore.getIndices(), dstSliceIndex,
                         loadStore.getMap());
        dstAccessReplacement(rewriter, loadStore, dstAccessMap,
                             dstAccessIndices);
      }
    }
  }

  // Extract loop induction variables from the outermost loop operation.
  // This collects induction variables from all nested loops in the nest.
  static SmallVector<Value> extractLoopInductionVars(Operation *outermostLoop) {
    SmallVector<Value> loopInductionVars;
    if (!outermostLoop) {
      return loopInductionVars;
    }

    // Collect induction variables from all loops in the nest.
    outermostLoop->walk([&](affine::AffineForOp loop) {
      loopInductionVars.push_back(loop.getBody()->getArgument(0));
    });

    // Reverse to get innermost loops first.
    std::reverse(loopInductionVars.begin(), loopInductionVars.end());
    return loopInductionVars;
  }

  // Rewrite stores to use dst register based on allocation map.
  static void insertDstRegisterAllocation(
      PatternRewriter &rewriter, Location loc, Value dst,
      const DstRegisterAllocation &dstRegisterAllocation) {
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    if (!dstType) {
      return;
    }
    const unsigned dstRank = dstType.getRank();

    // Iterate directly through dst register allocation entries.
    for (const auto &[op, dstInfo] : dstRegisterAllocation) {
      int64_t dstSliceIndex = dstInfo.dstSliceIndex;
      SmallVector<Value> loopInductionVars =
          extractLoopInductionVars(dstInfo.outermostLoop);

      // Store the result of this operation to dst register.
      rewriter.setInsertionPoint(op);

      SmallVector<Value> storeIndices;

      // Build store indices: [dstSliceIndex, loop_vars..., 0, 0, ...] using
      // loop induction variables for the dimensions that correspond to loops.
      storeIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, dstSliceIndex));

      // Use induction variables from the allocation.
      storeIndices.append(loopInductionVars);

      // Pad with zeros for remaining dimensions.
      while (storeIndices.size() < dstRank) {
        storeIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }

      // Ensure storeIndices matches the destination memref rank.
      assert(storeIndices.size() == dstRank &&
             "storeIndices size must match destination memref rank. If it's "
             "greater, probably need to use getNonParticipatingLoopDims to "
             "skip loop dimensions: "
             "https://github.com/tenstorrent/tt-mlir/pull/"
             "5081#discussion_r2376709558");

      auto storeMap =
          AffineMap::getMultiDimIdentityMap(dstRank, rewriter.getContext());

      rewriter.setInsertionPointAfter(op);

      // Insert dst reinterpret cast if compute result type differs from
      // dst type
      Value originalResult = op->getResult(0);
      Type originalType = originalResult.getType();
      Value valueToStore = originalResult;
      Operation *castOp = nullptr;
      bool needsTypeCast = (originalType != dstType.getElementType());

      if (needsTypeCast) {
        auto cast = rewriter.create<d2m::DstReinterpretCastOp>(
            loc, dstType.getElementType(), valueToStore);
        valueToStore = cast.getResult();
        castOp = cast.getOperation();
      }

      auto storeOp = rewriter.create<affine::AffineStoreOp>(
          loc, valueToStore, dst, storeMap, storeIndices);

      auto loadedResult = rewriter.create<affine::AffineLoadOp>(
          loc, dst, storeMap, storeIndices);

      // If we cast for storage, we need to cast back to the original type
      // after loading, since downstream ops expect the original type.
      Value replacementValue = loadedResult.getResult();
      Operation *castBackOp = nullptr;
      if (needsTypeCast) {
        auto castBack = rewriter.create<d2m::DstReinterpretCastOp>(
            loc, originalType, replacementValue);
        replacementValue = castBack.getResult();
        castBackOp = castBack.getOperation();
      }

      // Replace all uses of the original result with the (possibly cast back)
      // loaded result from dst register, but exclude the store operation and
      // cast operations to avoid circular dependencies.
      rewriter.replaceUsesWithIf(
          originalResult, replacementValue, [&](mlir::OpOperand &operand) {
            Operation *owner = operand.getOwner();
            return owner != storeOp && owner != castOp && owner != castBackOp;
          });
    }
  }

  // Returns the indices and the map for the load store from L1 and Dst.
  //   tuple(l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices).
  static std::tuple<AffineMap, SmallVector<Value>, AffineMap,
                    SmallVector<Value>>
  buildIndices(PatternRewriter &rewriter, Location loc,
               const mlir::IRMapping &irMapper, ValueRange currentIndices,
               int64_t dstSliceIndex, AffineMap map) {
    AffineMap l1AccessMap = map;
    SmallVector<Value> l1AccessIndices =
        llvm::to_vector(llvm::map_range(currentIndices, [&](Value index) {
          return irMapper.lookupOrDefault(index);
        }));

    AffineMap dstAccessMap = map.insertResult(
        getAffineConstantExpr(dstSliceIndex, rewriter.getContext()), 0);
    SmallVector<Value> dstAccessIndices = l1AccessIndices;
    return {l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices};
  }

  bool useTileMatmul = false;
  unsigned maxDstPhysicalSizeTiles = 0;
};
} // namespace

namespace {
template <typename TileReduceOp>
class D2MPackerMaskResetRewriter : public OpRewritePattern<TileReduceOp> {
public:
  using OpRewritePattern<TileReduceOp>::OpRewritePattern;

  Value index(OpBuilder &rewriter, Location loc, int64_t val) const {
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                              rewriter.getIndexAttr(val));
  }

  LogicalResult matchAndRewrite(TileReduceOp op,
                                PatternRewriter &rewriter) const final {

    bool packerResetFound = false;
    op->getBlock()->walk([&](Operation *op) {
      if (auto packerReset =
              mlir::dyn_cast_or_null<d2m::PackerMaskResetOp>(op)) {
        packerResetFound = true;
      }
    });
    if (packerResetFound) {
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    ReduceDim reduceDim = op.getReduceDim();
    SmallVector<int64_t> loopBounds =
        op->template getParentOfType<GenericOp>().getLoopBounds();

    scf::IfOp ifOp;
    if (reduceDim == ReduceDim::R) {
      auto iterIndex = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(1));
      auto condOp = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndex,
          index(rewriter, op.getLoc(), loopBounds[1] - 1));
      ifOp = rewriter.create<scf::IfOp>(op.getLoc(), condOp);
    } else if (reduceDim == ReduceDim::C) {
      auto iterIndex = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(0));
      auto condOp = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndex,
          index(rewriter, op.getLoc(), loopBounds[0] - 1));
      ifOp = rewriter.create<scf::IfOp>(op.getLoc(), condOp);
    } else if (reduceDim == ReduceDim::RC) {
      auto iterIndexR = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(1));
      auto iterIndexC = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(0));
      auto condOp = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndexR,
          index(rewriter, op.getLoc(), loopBounds[1] - 1));
      auto condOp2 = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndexC,
          index(rewriter, op.getLoc(), loopBounds[0] - 1));
      auto finalCondOp =
          rewriter.create<arith::OrIOp>(op.getLoc(), condOp, condOp2);
      ifOp = rewriter.create<scf::IfOp>(op.getLoc(), finalCondOp);
    }
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    rewriter.create<d2m::PackerMaskResetOp>(op.getLoc());

    return success();
  }
};

} // namespace

namespace {
class D2MInsertDstRegisterAccess
    : public impl::D2MInsertDstRegisterAccessBase<D2MInsertDstRegisterAccess> {
public:
  using impl::D2MInsertDstRegisterAccessBase<
      D2MInsertDstRegisterAccess>::D2MInsertDstRegisterAccessBase;

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<D2MInsertDstRegisterAccessRewriter>(
        ctx, useTileMatmul, maxDstPhysicalSizeTiles.getValue());

    patterns.add<D2MPackerMaskResetRewriter<TileReduceSumOp>,
                 D2MPackerMaskResetRewriter<TileReduceMaxOp>>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
