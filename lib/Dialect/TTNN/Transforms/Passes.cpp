// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <iomanip> // For setw
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>
#include <map>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include <unordered_map>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDEALLOCATE
#define GEN_PASS_DEF_TTNNCLUSTEROPS
#define GEN_PASS_DEF_TTNNCREATEINPUTGENERATORS
#define GEN_PASS_DEF_TTNNMODIFYSIGNATURESFORDYLIB
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNDeallocate : public impl::TTNNDeallocateBase<TTNNDeallocate> {

public:
  using impl::TTNNDeallocateBase<TTNNDeallocate>::TTNNDeallocateBase;

  Operation *getLastValueUsageOp(const LivenessBlockInfo *livenessInfo,
                                 Value value) {
    Operation *startOp = livenessInfo->getStartOperation(value);
    Operation *endOp = livenessInfo->getEndOperation(value, startOp);
    auto *opOperandIter =
        llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
          return opOperand.is(value);
        });

    // In case of DPS op keep going until we find the last usage of the tensor.
    //
    while (
        opOperandIter != endOp->getOpOperands().end() &&
        isa<DestinationStyleOpInterface>(endOp) &&
        cast<DestinationStyleOpInterface>(endOp).isDpsInit(&(*opOperandIter))) {
      OpResult result =
          cast<DestinationStyleOpInterface>(endOp).getTiedOpResult(
              &(*opOperandIter));
      endOp = livenessInfo->getEndOperation(result, endOp);
      opOperandIter =
          llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
            return opOperand.is(result);
          });
    }

    return endOp;
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    module->walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return;
      }
      assert(func.getBody().hasOneBlock() &&
             "found func that didn't have one block!");
      Liveness liveness(func.getOperation());
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(&func.getBody().front());

      // Handle non DPS ops which do not store function result and are used to
      // allocate tensors. DPS ops are handled via ttnn::EmptyOp.
      //
      func->walk([&](Operation *op) {
        if (isa<DestinationStyleOpInterface>(op)) {
          return;
        }

        // Skip ops which do not have results.
        //
        if (op->getNumResults() == 0) {
          return;
        }

        // Iterate over all results of the op.
        //
        for (OpResult result : op->getResults()) {
          // Check if result is ranked tensor type.
          //
          if (!isa<RankedTensorType>(result.getType())) {
            continue;
          }

          RankedTensorType resultTy =
              mlir::cast<RankedTensorType>(result.getType());
          assert(resultTy.getEncoding());

          Operation *lastOp = getLastValueUsageOp(livenessInfo, result);

          if (isa<func::ReturnOp>(lastOp)) {
            continue;
          }

          rewriter.setInsertionPointAfter(lastOp);
          rewriter.create<DeallocateOp>(lastOp->getLoc(), result);
        }
      });
    });
  }
};

class TTNNClusterOps : public impl::TTNNClusterOpsBase<TTNNClusterOps> {
public:
  using impl::TTNNClusterOpsBase<TTNNClusterOps>::TTNNClusterOpsBase;

  static std::string locationToStr(const mlir::Location &loc) {
    std::string locStr;
    llvm::raw_string_ostream(locStr) << loc;
    return locStr;
  }

  static std::tuple<std::string, std::vector<std::string>, std::string>
  splitLocationIntoChunks(mlir::Location loc) {
    const std::string &locationStr = locationToStr(loc);
    std::string opName;
    std::vector<std::string> components;
    std::string fullPath;

    // Find the opening quote of the op name
    size_t opNameStart = locationStr.find('"');
    if (opNameStart == std::string::npos) {
      return std::make_tuple("", std::vector<std::string>{},
                             ""); // Explicit vector type
    }

    // Find the closing quote of the op name
    size_t opNameEnd = locationStr.find('"', opNameStart + 1);
    if (opNameEnd == std::string::npos) {
      return std::make_tuple("", std::vector<std::string>{},
                             ""); // Explicit vector type
    }

    // Extract the op name
    opName = locationStr.substr(opNameStart + 1, opNameEnd - opNameStart - 1);

    // Find the start of the location path
    size_t pathStart = locationStr.find("(\"", opNameEnd);
    if (pathStart == std::string::npos) {
      return std::make_tuple(opName, std::vector<std::string>{},
                             ""); // Explicit vector type
    }

    // Find the end of the location path.
    size_t pathEnd = locationStr.find("\":", pathStart);
    if (pathEnd == std::string::npos) {
      return std::make_tuple(opName, std::vector<std::string>{},
                             ""); // Explicit vector type
    }

    std::string pathString =
        locationStr.substr(pathStart + 2, pathEnd - pathStart - 2);

    // Split the path string into components
    std::stringstream ss(pathString);
    std::string component;
    while (std::getline(ss, component, '/')) {
      components.push_back(component);
      if (!fullPath.empty()) {
        fullPath += "/";
      }
      fullPath += component;
    }

    // Add op name to components
    //
    components.push_back(opName);

    return std::make_tuple(opName, components, fullPath);
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    module->walk([&](func::FuncOp funcOp) {
      auto isFixed = fixIRLocations(funcOp);
      auto x = processFunc(funcOp);
      (void)x, (void)isFixed;
    });

    (void)module;
    (void)rewriter;
  }

private:
  class OpClusterTree {
  public:
    static constexpr int MIN_THRESHOLD = 5;

    class OpClusterNode {
    public:
      OpClusterNode(std::string str) {
        myStr = str;
        myHash = OpClusterTree::hashString(str);
      }

      void addOp(mlir::Operation *op) { ops.push_back(op); }
      void addChild(OpClusterNode *op) { children.push_back(op); }
      std::string getStr() const { return myStr; };
      const std::vector<OpClusterNode *> &getChildrenNodes() const {
        return children;
      }
      const std::vector<mlir::Operation *> &getOps() const { return ops; }

    private:
      std::string myStr;
      size_t myHash;
      std::vector<OpClusterNode *> children;
      std::vector<mlir::Operation *> ops;
    };

    OpClusterTree() { rootNode = new OpClusterNode("@ROOT@"); }

    ~OpClusterTree() { delete rootNode; }

    OpClusterNode *rootNode;

  private:
    // std::unordered_map<size_t, OpClusterNode *> hashToOCNodeMap;

  public:
    void addNode(mlir::Operation *op) {
      mlir::Location loc = op->getLoc();

      assert(isLocationValid(loc));

      auto [opName, components, fullPath] =
          splitLocationIntoChunks(op->getLoc());

      if (opName == "conv2d_18.dc.conv2d.2") {
        std::cout << "found it!" << std::endl;
      }

      // Must have at least 1 component.
      //
      if (!components.size()) {
        return; // TEMOPORARY HACK!
      }
      assert(components.size());

      OpClusterNode *prev = rootNode;

      bool foundFirstNew = false;
      for (const std::string &component : components) {
        OpClusterNode *node = nullptr;
        bool exists = false;
        for (OpClusterNode *child : prev->getChildrenNodes()) {
          if (child->getStr() == component) {
            exists = true;
            node = child;
            break;
          }
        }

        if (!exists) {
          foundFirstNew = true;
        }

        if (foundFirstNew) {
          // Add node to path.
          //
          // TODO: dealloc these new-init'ed nodes
          //
          node = new OpClusterNode(component);

          // Add edge from parent to this ocNode.
          //
          prev->addChild(node);
        }

        // If last component, add Op pointer to it.
        //
        if (component == components.back()) {
          node->addOp(op); //
        }

        // Set prev for next iteration.
        //
        prev = node;
      }

      // Debug prints.
      //
      // llvm::outs() << "PRINTING OP LOC CHUNKS:" << "\n";
      // for (std::string &chunk : components) {
      //   llvm::outs() << "    " << chunk << "\n";
      // }

      // Add attr.
      //
      // if (mlir::isa<ttnn::Conv2dOp>(op)) {
      op->setAttr(
          "added_attr_string",
          mlir::StringAttr::get(op->getContext(), locationToStr(op->getLoc())));
      // }
    }

    void markNodeAsFn(const OpClusterNode *node) {
      std::string fullPath = node->getStr();
      llvm::outs() << "===========> name: " << fullPath << "\n";
      // 1. pick a name for fn
      // 2. iterate all children and their ops, marking them all with name
    }

    int analyzeNode(const OpClusterNode *node) {
      const std::vector<OpClusterNode *> &children = node->getChildrenNodes();
      const std::vector<mlir::Operation *> &ops = node->getOps();

      int score = 0;

      // Check if leaf node.
      //
      if (children.size() == 0) {

        // assert(ops.size() > 0); // must have ops if leaf node.
        if (ops.size() == 0) {
          return 0;
        }

        score = ops.size();

        if (score > MIN_THRESHOLD) {
          markNodeAsFn(node);
          return 0; // return score of 0 to parent
        }
      } else {
        // children.size() > 0

        score = ops.size();
        assert(score == 0 && "I didn't expect this, why does this happen?");

        for (const OpClusterNode *child : children) {
          score += analyzeNode(child);
        }

        if (score > MIN_THRESHOLD) {
          markNodeAsFn(node);
          // TODO: MARK NODE AS FUNCTION
          return 0; // return score of 0 to parent
        }
      }

      return score;
    }

    void runAnalysis() {
      // Start from root.
      //
      analyzeNode(rootNode);
    }

    // void printTree(const OpClusterNode *node, int indent = 0) {

    //   std::cout << std::setw(indent) << "" << node->getStr()
    //             << std::endl; // Indent and print data

    //   for (const OpClusterNode *child : rootNode->getChildrenNodes()) {
    //     printTree(child, indent + 4); // Increase indent for children
    //   }
    // }

    void printOpClusterTree(const OpClusterNode *node,
                            const std::string &prefix = "") {
      if (node == nullptr) {
        return;
      }

      std::cout << prefix << node->getStr() << std::endl;

      const std::vector<OpClusterNode *> children = node->getChildrenNodes();

      for (size_t i = 0; i < children.size(); ++i) {
        const OpClusterNode *child = children[i];
        std::string newPrefix = prefix;
        if (i < children.size() - 1) {
          newPrefix += "├── ";
        } else {
          newPrefix += "└── ";
        }

        printOpClusterTree(child, newPrefix);
      }
    }

  private:
    bool isLocationValid(const mlir::Location &loc) {
      // TODO: Actually verify location
      return true;
    }

    // std::tuple<bool, OpClusterNode *> getNodeByStrIfExists(std::string str) {
    //   auto it = hashToOCNodeMap.find(hashString(str));
    //   if (it != hashToOCNodeMap.end()) {
    //     return std::make_tuple(true, it->second);
    //   } else {
    //     return std::make_tuple(false, nullptr);
    //   }
    // }

  private:
    static size_t hashString(std::string str) {
      return std::hash<std::string>{}(str);
    }
  };

  // Should this take in region/body instead of whole func?
  mlir::LogicalResult annotateOps(func::FuncOp funcOp) {
    assert(funcOp.getBlocks().size() == 1);

    OpClusterTree ocTree = OpClusterTree();

    funcOp->walk([&ocTree](mlir::Operation *op) { ocTree.addNode(op); });

    ocTree.printOpClusterTree(ocTree.rootNode);

    ocTree.runAnalysis();

    return mlir::success();
  }

  mlir::LogicalResult fixIRLocations(func::FuncOp funcOp) {
    funcOp->walk([&](mlir::Operation *op) {
      auto [opName, components, fullPath] =
          splitLocationIntoChunks(op->getLoc());

      std::cout << "opName: " << opName << ", fullPath: " << fullPath
                << std::endl;
    });

    return mlir::success();
  }

  mlir::LogicalResult processFunc(func::FuncOp funcOp) {
    return annotateOps(funcOp);
  }
};

class TTNNCreateInputGenerators
    : public impl::TTNNCreateInputGeneratorsBase<TTNNCreateInputGenerators> {

public:
  using impl::TTNNCreateInputGeneratorsBase<
      TTNNCreateInputGenerators>::TTNNCreateInputGeneratorsBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    // Ensure that the module has a single region and a single block within that
    // region
    assert(module->getRegions().size() == 1);
    assert(module->getRegion(0).getBlocks().size() == 1);

    // Get the first block of the region at index 0
    //
    Block *firstBlock = module.getBody(0);

    // Find all the func.func ops in the module that are "forward" functions
    //
    SmallVector<func::FuncOp, 1> forwardFuncOps;
    for (mlir::Operation &op : firstBlock->getOperations()) {
      if (mlir::func::FuncOp funcOp = dyn_cast<func::FuncOp>(op)) {

        // Skip functions that are called elsewhere in the IR
        //
        // This will skip utility functions that are used by other functions,
        // only top-level "forward" functions should be considered
        //
        if (!funcOp->getUses().empty()) {
          continue;
        }

        forwardFuncOps.push_back(funcOp);
      }
    }

    // Iterate over all the func ops and add input tensor generator functions
    //
    for (mlir::func::FuncOp forwardFuncOp : forwardFuncOps) {
      // Get all the input tensors for the current forward func
      //
      llvm::SmallVector<mlir::RankedTensorType, 2> inputTensors;
      for (auto input : forwardFuncOp.getFunctionType().getInputs()) {
        inputTensors.push_back(llvm::cast<mlir::RankedTensorType>(input));
      }

      // Create a new function that will generate the input tensors
      //
      std::string inputGenFuncName =
          "createInputsFor_" + forwardFuncOp.getName().str();

      // Create function type
      //
      mlir::TypeRange returnTypeRange =
          mlir::TypeRange(forwardFuncOp.getFunctionType().getInputs());
      FunctionType functionType =
          mlir::FunctionType::get(&getContext(), {}, returnTypeRange);

      // Set insertion point to end of first block
      //
      rewriter.setInsertionPointToEnd(firstBlock);

      // Create the function
      //
      func::FuncOp inputGenFuncOp = rewriter.create<mlir::func::FuncOp>(
          module->getLoc(), inputGenFuncName, functionType);

      // Add a Block to func op and set insertion point to the beginning of the
      // Block
      //
      ::mlir::Block *currFnBlock = inputGenFuncOp.addEntryBlock();
      rewriter.setInsertionPointToStart(currFnBlock);

      // Create the input tensors
      //
      SmallVector<Value, 2> generatedTensors;
      for (Type tensorType : returnTypeRange) {
        assert(llvm::isa<mlir::RankedTensorType>(tensorType));

        RankedTensorType tensor =
            llvm::cast<mlir::RankedTensorType>(tensorType);

        // Get the layout attribute
        //
        ttnn::TTNNLayoutAttr layoutAttr =
            mlir::cast<ttnn::TTNNLayoutAttr>(tensor.getEncoding());

        // Get the shape of the tensor, tensor layout, and data type
        //
        ShapeAttr shapeAttr =
            ttnn::ShapeAttr::get(&getContext(), tensor.getShape());
        ttnn::LayoutAttr tensorLayoutAttr =
            ttnn::LayoutAttr::get(&getContext(), layoutAttr.getLayout());
        DataTypeAttr dTypeAttr =
            DataTypeAttr::get(&getContext(), layoutAttr.getDataType());

        // Create a new tensor
        //
        ttnn::OnesOp onesOp = rewriter.create<ttnn::OnesOp>(
            forwardFuncOp->getLoc(), tensorType, shapeAttr, dTypeAttr,
            tensorLayoutAttr, nullptr, nullptr);

        // If tensor is meant to be on device, add ToDevice op
        //
        if (layoutAttr.isDeviceBufferType()) {
          ttnn::GetDeviceOp device =
              ttnn::utils::getOrInsertDevice(rewriter, onesOp);

          mlir::Value tensorOnDevice = rewriter.create<ttnn::ToDeviceOp>(
              forwardFuncOp->getLoc(), tensorType, onesOp.getResult(),
              device.getResult(), nullptr);

          generatedTensors.push_back(tensorOnDevice);
        } else {
          generatedTensors.push_back(onesOp.getResult());
        }
      }

      // Return the generated tensors
      //
      rewriter.create<func::ReturnOp>(forwardFuncOp->getLoc(),
                                      generatedTensors);
    }

    // Create a main function to call input generators and forward funcs
    //
    {
      // Create a new function that will generate the input tensors
      //
      std::string mainFuncName = "main";

      // Create function type
      //
      mlir::Type i32Type = rewriter.getI32Type();
      mlir::TypeRange returnTypeRange = mlir::TypeRange(i32Type);
      FunctionType functionType =
          mlir::FunctionType::get(&getContext(), {}, returnTypeRange);

      // Set insertion point to end of first block
      //
      rewriter.setInsertionPointToEnd(firstBlock);

      // Create the function
      //
      func::FuncOp mainFuncOp = rewriter.create<mlir::func::FuncOp>(
          module->getLoc(), mainFuncName, functionType);

      ::mlir::Block *currFnBlock = mainFuncOp.addEntryBlock();

      // Set insertion point to the beginning of the block
      //
      rewriter.setInsertionPointToStart(currFnBlock);

      // Call the input generators
      //
      for (mlir::func::FuncOp forwardFuncOp : forwardFuncOps) {
        std::string inputGenFuncName =
            "createInputsFor_" + forwardFuncOp.getName().str();

        // Get the input generator function
        //
        mlir::func::FuncOp inputGenFuncOp =
            module.lookupSymbol<mlir::func::FuncOp>(inputGenFuncName);

        // Call the input generator function
        //
        func::CallOp createdTensors = rewriter.create<mlir::func::CallOp>(
            forwardFuncOp->getLoc(), inputGenFuncOp, ValueRange());

        rewriter.create<mlir::func::CallOp>(forwardFuncOp->getLoc(),
                                            forwardFuncOp,
                                            createdTensors->getResults());
      }

      // Return 0
      //
      // func::ReturnOp requires a Value to be returned, which means that an SSA
      // needs to be returned, hence create a constant 0 via arith::ConstantOp
      //
      Value constantZero = rewriter.create<arith::ConstantOp>(
          rewriter.getUnknownLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(0));
      rewriter.create<func::ReturnOp>(mainFuncOp->getLoc(), constantZero);
    }
  }
};

class TTNNModifySignaturesForDylib
    : public impl::TTNNModifySignaturesForDylibBase<
          TTNNModifySignaturesForDylib> {

public:
  using impl::TTNNModifySignaturesForDylibBase<
      TTNNModifySignaturesForDylib>::TTNNModifySignaturesForDylibBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // If we have a nested module structure, we want to use nested module inside
    // DeviceModule.
    tt::DeviceModuleOp deviceModule;
    for (auto &op : module.getBody()->getOperations()) {
      deviceModule = llvm::dyn_cast<tt::DeviceModuleOp>(op);
      if (deviceModule) {
        break;
      }
    }
    if (deviceModule) {
      module = dyn_cast_if_present<mlir::ModuleOp>(
          deviceModule.getBodyRegion().front().front());
      assert(module &&
             "Found tt::DeviceModuleOp but it didn't contain a single "
             "mlir::ModuleOp!");
    }
    IRRewriter rewriter(&getContext());

    // Ensure that the module has a single region and a single block within that
    // region
    assert(module->getRegions().size() == 1);
    assert(module->getRegion(0).getBlocks().size() == 1);

    // Get the first block of the region at index 0
    //
    Block *firstBlock = module.getBody(0);

    // Find all the func.func ops in the module that are "forward" functions
    //
    SmallVector<func::FuncOp, 1> forwardFuncOps;
    for (mlir::Operation &op : firstBlock->getOperations()) {
      if (mlir::func::FuncOp funcOp = dyn_cast<func::FuncOp>(op)) {

        // Skip functions that are called elsewhere in the IR
        //
        // This will skip utility functions that are used by other functions,
        // only top-level "forward" functions should be considered
        //
        if (!funcOp->getUses().empty()) {
          continue;
        }

        forwardFuncOps.push_back(funcOp);
      }
    }

    // Iterate over all the func ops and modify the signatures
    //
    for (mlir::func::FuncOp forwardFuncOp : forwardFuncOps) {
      // Replace the signature of the forward function so that all the tensor
      // arguments are packed into a single tuple, and device type is appended
      //
      mlir::FunctionType originalFuncType = forwardFuncOp.getFunctionType();
      assert(
          std::all_of(originalFuncType.getInputs().begin(),
                      originalFuncType.getInputs().end(),
                      [](Type t) { return mlir::isa<RankedTensorType>(t); }) &&
          "Expected all inputs must be of type RankedTensorType");

      // Find device op
      //
      ttnn::GetDeviceOp getDeviceOp = nullptr;
      forwardFuncOp.walk([&](ttnn::GetDeviceOp currGDOp) {
        assert(!getDeviceOp &&
               "Only one device expected, but found more than one!");
        getDeviceOp = currGDOp;
      });

      // Create Type objects for modified function signature:
      // 1. tuplifiedInputTensors: TupleType of all input tensors
      // 2. deviceType: DeviceType
      // 3. tuplifiedOutputTensors: TupleType of all output tensors
      //
      mlir::TupleType tuplifiedInputTensors =
          mlir::TupleType::get(&getContext(), originalFuncType.getInputs());
      std::optional<ttnn::DeviceType> deviceType = std::nullopt;
      if (getDeviceOp) {
        deviceType = getDeviceOp.getResult().getType();
      }
      mlir::TupleType tuplifiedOutputTensors =
          mlir::TupleType::get(&getContext(), originalFuncType.getResults());

      // Create modified function type (signature) that takes the input tuple
      // and device as operands, and returns the output tuple
      //
      SmallVector<Type> modifiedInputTypes;
      modifiedInputTypes.push_back(tuplifiedInputTensors);
      if (deviceType.has_value()) {
        modifiedInputTypes.push_back(*deviceType);
      }
      FunctionType modifiedFuncType =
          originalFuncType.clone(modifiedInputTypes, tuplifiedOutputTensors);

      rewriter.modifyOpInPlace(forwardFuncOp,
                               [&forwardFuncOp, &modifiedFuncType]() {
                                 forwardFuncOp.setType(modifiedFuncType);
                               });

      // First block of the function (often referred to as "entry block") needs
      // its arguments updated as well - the args need to match the containing
      // func's arguments; this is implemented here by first inserting the tuple
      // as the first argument of the block, inserting GetTupleElementOp ops to
      // start of the block in order to unpack tuple elements, and then
      // replacing all uses of the original block arguments with the
      // GetTupleElementOp results - after this it's finally safe to remove
      // original block arguments as they have no live uses anymore
      //
      // Additionally, the Device is added as the second argument, and the
      // GetDeviceOp that creates Device is removed
      //
      // The return statement is modified to return a tuple
      //
      Block &entryBlock = forwardFuncOp.getBlocks().front();
      size_t paramOffset = 1;
      entryBlock.insertArgument(/*index=*/0u, tuplifiedInputTensors,
                                forwardFuncOp.getLoc());
      if (deviceType.has_value()) {
        entryBlock.insertArgument(/*index=*/1u, *deviceType,
                                  forwardFuncOp.getLoc());
        paramOffset++;
      }

      rewriter.setInsertionPointToStart(&entryBlock);
      for (size_t idx = 0; idx < originalFuncType.getInputs().size(); idx++) {
        ::mlir::tt::GetTupleElementOp getTupleElementOp =
            rewriter.create<mlir::tt::GetTupleElementOp>(
                forwardFuncOp.getLoc(), forwardFuncOp.getArgument(0), idx);

        rewriter.replaceAllUsesWith(entryBlock.getArgument(paramOffset + idx),
                                    getTupleElementOp);
      }

      // Erase original arguments
      //
      entryBlock.eraseArguments(paramOffset,
                                originalFuncType.getInputs().size());

      // Remove device usage and remove the original GetDeviceOp
      //
      if (getDeviceOp) {
        rewriter.replaceAllUsesWith(getDeviceOp.getResult(),
                                    entryBlock.getArgument(1));
        rewriter.eraseOp(getDeviceOp);
      }

      // Find return statement and replace with tuple
      //
      forwardFuncOp->walk([&](mlir::func::ReturnOp returnOp) {
        rewriter.setInsertionPointAfter(returnOp);
        TupleOp tupleOp = rewriter.create<mlir::tt::TupleOp>(
            returnOp.getLoc(), returnOp.getOperands());

        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(returnOp,
                                                          tupleOp.getResult());
      });
    }
  }
};

} // namespace mlir::tt::ttnn
