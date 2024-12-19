// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
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

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDEALLOCATE
#define GEN_PASS_DEF_TTNNDECOMPOSELAYOUTS
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

class TTNNDecomposeLayouts
    : public impl::TTNNDecomposeLayoutsBase<TTNNDecomposeLayouts> {

public:
  using impl::TTNNDecomposeLayoutsBase<
      TTNNDecomposeLayouts>::TTNNDecomposeLayoutsBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());
    llvm::SmallVector<Operation *> opsToReplace;
    module->walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return;
      }
      assert(func.getBody().hasOneBlock() &&
             "found func that didn't have one block!");
      func->walk([&](Operation *op) {
        if (!isa<ttnn::ToLayoutOp>(op)) {
          return;
        }
        opsToReplace.push_back(op);
      });
    });
    for (Operation *op : opsToReplace) {
      this->createLayoutConversionOps(mlir::cast<ttnn::ToLayoutOp>(op),
                                      rewriter);
      rewriter.eraseOp(op);
    }
  }

private:
  struct LayoutInfo {
    ttnn::BufferType bufferType;
    ttnn::Layout layoutEnum;
    DataType dataType;
    ttnn::TensorMemoryLayoutAttr tensorMemoryLayout;
    llvm::SmallVector<int64_t> shardShape;

    ttnn::MemoryConfigAttr createMemoryConfigAttr(MLIRContext *context) const {
      return ttnn::MemoryConfigAttr::get(
          context, ttnn::BufferTypeAttr::get(context, bufferType),
          ttnn::ShardSpecAttr::get(context,
                                   ttnn::ShapeAttr::get(context, shardShape)),
          tensorMemoryLayout);
    }

    bool isOnHost() const {
      return bufferType == ttnn::BufferType::SystemMemory;
    }
    bool isOnDevice() const { return not isOnHost(); }
    bool isTilized() const { return layoutEnum == ttnn::Layout::Tile; }
  };

  struct OpsToCreate {
    bool createToDeviceOp = false;
    bool createFromDeviceOp = false;
    bool createToLayoutOp = false;
    bool createTypecastOp = false;
    bool createToMemoryConfigOp = false;

    bool createSomeOp() const {
      return createToLayoutOp or createTypecastOp or createToDeviceOp or
             createFromDeviceOp or createToMemoryConfigOp;
    }

    void print() const {
      llvm::errs() << "OpsToCreate{ \n"
                   << "\t"
                   << "CreateToDeviceOp: " << createToDeviceOp << "\n"
                   << "\t"
                   << "CreateFromDeviceOp: " << createFromDeviceOp << "\n"
                   << "\t"
                   << "CreateToLayoutOp: " << createToLayoutOp << "\n"
                   << "\t"
                   << "CreateTypecastOp: " << createTypecastOp << "\n"
                   << "\t"
                   << "CreateToMemoryConfigOp: " << createToMemoryConfigOp
                   << "\n"
                   << "}\n";
    }
  };

  struct OpCreationInfo {
    mlir::Value device;
    LayoutInfo input;
    LayoutInfo output;
    OpsToCreate opsToCreate;

    OpCreationInfo(mlir::Value device, const LayoutInfo &input,
                   const LayoutInfo &output, const OpsToCreate &opsToCreate)
        : device(device), input(input), output(output),
          opsToCreate(opsToCreate) {}

    bool shouldUntilize() const {
      return opsToCreate.createToLayoutOp and not output.isTilized();
    }

    bool shouldTilize() const {
      return opsToCreate.createToLayoutOp and output.isTilized();
    }
  };

  std::pair<LayoutInfo, LayoutInfo>
  getInputOutputLayouts(ttnn::ToLayoutOp op) const {
    LayoutInfo input, output;

    auto inputLayoutAttr =
        mlir::cast<TTNNLayoutAttr>(op.getInput().getType().getEncoding());

    assert(op.getMemoryConfig().has_value());
    MemoryConfigAttr outputMemoryConfig = op.getMemoryConfig().value();

    input.bufferType = inputLayoutAttr.getBufferType();
    output.bufferType = outputMemoryConfig.getBufferType().getValue();

    input.layoutEnum = inputLayoutAttr.getLayout();
    output.layoutEnum = op.getLayout();

    input.dataType = inputLayoutAttr.getDataType();
    assert(op.getDtype().has_value());
    output.dataType = op.getDtype().value();

    input.tensorMemoryLayout = inputLayoutAttr.getMemLayout();
    output.tensorMemoryLayout = outputMemoryConfig.getTensorMemoryLayout();

    input.shardShape = inputLayoutAttr.getShardShape();
    output.shardShape =
        llvm::SmallVector<int64_t>{outputMemoryConfig.getShardShapeArray()};
    return {input, output};
  }

  OpsToCreate determineRequiredOps(const LayoutInfo &input,
                                   const LayoutInfo &output) const {
    OpsToCreate opsToCreate;

    opsToCreate.createToDeviceOp =
        (input.bufferType != output.bufferType) and input.isOnHost();
    opsToCreate.createFromDeviceOp =
        (input.bufferType != output.bufferType) and output.isOnHost();

    opsToCreate.createTypecastOp = input.dataType != output.dataType;
    opsToCreate.createToLayoutOp = input.layoutEnum != output.layoutEnum;
    // TODO(bug #665):
    // Insert a ToLayoutOp manually if we're moving from device to host to
    // untilize. Since we're hardcoding tile layout, the tensor may be row
    // major in mlir, and therefore it would appear as if we don't need to
    // untilize
    opsToCreate.createToLayoutOp |=
        (opsToCreate.createFromDeviceOp and
         output.layoutEnum == ttnn::Layout::RowMajor);

    // ToDeviceOp can handle the creation of the memory config of the initial
    // device tensor
    if (not opsToCreate.createToDeviceOp and output.isOnDevice()) {
      opsToCreate.createToMemoryConfigOp =
          output.tensorMemoryLayout &&
          (input.tensorMemoryLayout != output.tensorMemoryLayout);
      opsToCreate.createToMemoryConfigOp |=
          (input.bufferType == ttnn::BufferType::DRAM and
           output.bufferType == ttnn::BufferType::L1) or
          (input.bufferType == ttnn::BufferType::L1 and
           output.bufferType == ttnn::BufferType::DRAM);
      opsToCreate.createToMemoryConfigOp |=
          (input.shardShape != output.shardShape);
    }
    return opsToCreate;
  }

  bool isCreationValid(ttnn::ToLayoutOp op, const LayoutInfo &input,
                       const LayoutInfo &output,
                       const OpsToCreate &opsToCreate) const {

    if (not opsToCreate.createSomeOp()) {
      op->emitError(
          "Redundant ttnn::ToLayoutOp - no ttnn layout ops "
          "needed, this may be due to the forcing of tile/row major layouts.");
      return false;
    }

    if (opsToCreate.createToDeviceOp and opsToCreate.createFromDeviceOp) {
      op->emitError("Cannot create both ToDeviceOp and FromDeviceOp");
      return false;
    }

    if (opsToCreate.createToMemoryConfigOp and
        output.bufferType == ttnn::BufferType::SystemMemory) {
      op->emitError(
          "ToMemoryConfigOp only supported for device output tensors");
      return false;
    }

    if (input.isOnHost() and opsToCreate.createFromDeviceOp) {
      op->emitError("Unexpected FromDeviceOp on host tensor");
      return false;
    }

    if (input.isOnDevice() and opsToCreate.createToDeviceOp) {
      op->emitError("Unexpected ToDeviceOp on device tensor");
      return false;
    }
    return true;
  }

  /* Helper functions to create ttnn layout ops */

  template <typename OpType, typename... Args>
  mlir::Value createOp(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                       mlir::Value currentInput, Args... args) const {

    rewriter.setInsertionPoint(op);
    return rewriter.create<OpType>(op.getLoc(), op.getType(), currentInput,
                                   args...);
  }

  mlir::Value createToDeviceOpIfNeeded(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    if (not info.opsToCreate.createToDeviceOp) {
      return currentInput;
    }
    ttnn::MemoryConfigAttr memoryConfigAttr =
        info.output.createMemoryConfigAttr(op.getContext());
    return this->createOp<ttnn::ToDeviceOp>(op, rewriter, currentInput,
                                            info.device, memoryConfigAttr);
  }

  // FromDeviceOp
  mlir::Value createFromDeviceOpIfNeeded(ttnn::ToLayoutOp op,
                                         IRRewriter &rewriter,
                                         mlir::Value currentInput,
                                         const OpCreationInfo &info,
                                         bool forceCreate = false) const {
    if (not info.opsToCreate.createFromDeviceOp) {
      return currentInput;
    }
    return this->createOp<ttnn::FromDeviceOp>(op, rewriter, currentInput);
  }

  mlir::Value createToLayoutOpIfNeeded(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    if (not info.opsToCreate.createToLayoutOp) {
      return currentInput;
    }
    ttnn::LayoutAttr layoutAttr =
        ttnn::LayoutAttr::get(op.getContext(), info.output.layoutEnum);
    return this->createOp<ttnn::ToLayoutOp>(
        op, rewriter, currentInput, layoutAttr, /*dtype*/ nullptr,
        /*memory_config*/ nullptr, /*device*/ nullptr);
  }

  mlir::Value createTypecastOpIfNeeded(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    if (not info.opsToCreate.createTypecastOp) {
      return currentInput;
    }
    DataTypeAttr dtypeAttr =
        DataTypeAttr::get(op.getContext(), info.output.dataType);
    return this->createOp<ttnn::TypecastOp>(op, rewriter, currentInput,
                                            dtypeAttr);
  }

  mlir::Value createToMemoryConfigOpIfNeeded(ttnn::ToLayoutOp op,
                                             IRRewriter &rewriter,
                                             mlir::Value currentInput,
                                             const OpCreationInfo &info) const {
    if (not info.opsToCreate.createToMemoryConfigOp) {
      return currentInput;
    }
    ttnn::MemoryConfigAttr memoryConfigAttr =
        info.output.createMemoryConfigAttr(op.getContext());
    return this->createOp<ttnn::ToMemoryConfigOp>(op, rewriter, currentInput,
                                                  memoryConfigAttr);
  }

  /* Functions that create ops based on the layouts of the input output tensors
   */

  void handleHostInputNoLayoutNoTypecast(ttnn::ToLayoutOp op,
                                         IRRewriter &rewriter,
                                         mlir::Value currentInput,
                                         const OpCreationInfo &info) const {
    // If the input tensor is on host and we don't need to create a ToLayoutOp
    // nor a TypecastOp we can create a ToDeviceOp and a ToMemoryConfigOp if
    // needed and return
    currentInput =
        this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
    currentInput =
        this->createToMemoryConfigOpIfNeeded(op, rewriter, currentInput, info);
    op.getResult().replaceAllUsesWith(currentInput);
  }

  void handleHostInputLayoutNoTypecast(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    assert(input.dataType == output.dataType &&
           "Data type should be the same if we're not creating typecast op");
    /* if we should untilize, untilize on host */
    if (info.shouldUntilize()) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* If we should tilize, and the data type is bfloat16, we can tilize on
     * device */
    if (info.shouldTilize() and output.dataType == DataType::BFloat16) {
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* If we should tilize, and the data type is not bfloat16, we tilize on host
     */
    if (info.shouldTilize() and output.dataType != DataType::BFloat16) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    llvm_unreachable("Unreachable code path");
  }

  void handleHostInputNoLayoutTypecast(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    assert(input.layoutEnum == output.layoutEnum &&
           "Layout should be the same if we're not creating a ToLayoutOp");

    /* If the output is already tilized, we can typecast on device */
    if (output.isTilized()) {
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* If the output is not tilized, typecast on host */
    if (not output.isTilized()) {
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    llvm_unreachable("Unreachable code path");
  }

  void handleHostInputLayoutTypecast(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                                     mlir::Value currentInput,
                                     const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;

    /* If we need to untilize and typecast, then untilize and typecast on host
     */
    if (info.shouldUntilize()) {
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* If we need to tilize and the input datatype is bfloat16
       we can tilize on device and then typecast afterwards */
    if (info.shouldTilize() and input.dataType == DataType::BFloat16) {
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* if we need to tilize and the output data type is bfloat16
       we can typecast on host and tilize on device */
    if (info.shouldTilize() and output.dataType == DataType::BFloat16) {
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* if we need to tilize and the input/ output data types are not bfloat16 do
     * everything on host */
    if (info.shouldTilize() and input.dataType != DataType::BFloat16 and
        output.dataType != DataType::BFloat16) {
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    llvm_unreachable("Unreachable code path");
  }

  void handleHostInputLayoutConversion(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    const OpsToCreate &opsToCreate = info.opsToCreate;
    if (not opsToCreate.createToLayoutOp and not opsToCreate.createTypecastOp) {
      return handleHostInputNoLayoutNoTypecast(op, rewriter, currentInput,
                                               info);
    }
    if (opsToCreate.createToLayoutOp and not opsToCreate.createTypecastOp) {
      return handleHostInputLayoutNoTypecast(op, rewriter, currentInput, info);
    }
    if (not opsToCreate.createToLayoutOp and opsToCreate.createTypecastOp) {
      return handleHostInputNoLayoutTypecast(op, rewriter, currentInput, info);
    }
    if (opsToCreate.createToLayoutOp and opsToCreate.createTypecastOp) {
      return handleHostInputLayoutTypecast(op, rewriter, currentInput, info);
    }
    llvm_unreachable("Unreachable code path");
  }

  void handleDeviceInputNoLayoutNoTypecast(ttnn::ToLayoutOp op,
                                           IRRewriter &rewriter,
                                           mlir::Value currentInput,
                                           const OpCreationInfo &info) const {
    // If the input tensor is on device and we don't need to create a ToLayoutOp
    // nor a TypecastOp we can create a FromDeviceOp and a ToMemoryConfigOp if
    // needed and return
    currentInput =
        this->createToMemoryConfigOpIfNeeded(op, rewriter, currentInput, info);
    currentInput =
        this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
    op.getResult().replaceAllUsesWith(currentInput);
  }

  void handleDeviceInputLayoutNoTypecast(ttnn::ToLayoutOp op,
                                         IRRewriter &rewriter,
                                         mlir::Value currentInput,
                                         const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    const OpsToCreate &opsToCreate = info.opsToCreate;
    assert(input.dataType == output.dataType &&
           "Data type should be the same if we're not creating typecast op");

    /* if we should untilize, untilize on host */
    /* this is the main untilize case hit when we read data from device back to
     * host at the end of the program */
    if (info.shouldUntilize() and opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* This is a rare untilize case, where we want to untilize a device tensor
       but keep it on device to handle this we need to move the tensor to host,
       untilize it, and then move it back to device */
    if (info.shouldUntilize() and not opsToCreate.createFromDeviceOp) {
      // Force-create a FromDeviceOp
      currentInput =
          this->createOp<ttnn::FromDeviceOp>(op, rewriter, currentInput);
      // untilize on host
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert memory config if needed
      currentInput = this->createOp<ttnn::ToDeviceOp>(
          op, rewriter, currentInput, info.device,
          /* optional MemConfigAttr */ nullptr);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* If we should tilize and the input data type is bfloat16, tilize on device
     */
    if (info.shouldTilize() and input.dataType == DataType::BFloat16) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* If we should tilize and the input data type is not bfloat16, tilize on
     * host */
    if (info.shouldTilize() and input.dataType != DataType::BFloat16 and
        opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* If we want to tilize a device tensor that is not bfloat16, we need to
     * tilize on host and move it back */
    if (info.shouldTilize() and input.dataType != DataType::BFloat16 and
        not opsToCreate.createFromDeviceOp) {
      // Force-create a FromDeviceOp
      currentInput =
          this->createOp<ttnn::FromDeviceOp>(op, rewriter, currentInput);
      // tilize on host
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert memory config if needed
      currentInput = this->createOp<ttnn::ToDeviceOp>(
          op, rewriter, currentInput, info.device,
          /* optional MemConfigAttr */ nullptr);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    llvm_unreachable("Unreachable code path");
  }

  void handleDeviceInputNoLayoutTypecast(ttnn::ToLayoutOp op,
                                         IRRewriter &rewriter,
                                         mlir::Value currentInput,
                                         const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const LayoutInfo &output = info.output;
    const OpsToCreate &opsToCreate = info.opsToCreate;
    assert(input.layoutEnum == output.layoutEnum &&
           "Layout should be the same if we're not creating toLayout op");

    /* If the output is tilized, typecast directly on device*/
    if (output.isTilized()) {
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* If the output is not tilized, typecast on host */
    if (not output.isTilized() and opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* Device to device untilized typecast, need to move to host first */
    if (not output.isTilized() and not opsToCreate.createFromDeviceOp) {
      // Force-create a FromDeviceOp
      currentInput =
          this->createOp<ttnn::FromDeviceOp>(op, rewriter, currentInput);
      // typecast on host
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert memory config if needed
      currentInput = this->createOp<ttnn::ToDeviceOp>(
          op, rewriter, currentInput, info.device,
          /* optional MemConfigAttr */ nullptr);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    llvm_unreachable("Unreachable code path");
  }

  void handleDeviceInputLayoutTypecast(ttnn::ToLayoutOp op,
                                       IRRewriter &rewriter,
                                       mlir::Value currentInput,
                                       const OpCreationInfo &info) const {
    const LayoutInfo &input = info.input;
    const OpsToCreate &opsToCreate = info.opsToCreate;

    /* If we need to untilize, typecast on device and untilize on host */
    if (info.shouldUntilize() and opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* Rare case of device to device untilize, typecast on device, untilize on
     * host, move back to device */
    if (info.shouldUntilize() and not opsToCreate.createFromDeviceOp) {
      // typecast on device
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      // Force-create a FromDeviceOp
      currentInput =
          this->createOp<ttnn::FromDeviceOp>(op, rewriter, currentInput);
      // untilize on host
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert memory config if needed
      currentInput = this->createOp<ttnn::ToDeviceOp>(
          op, rewriter, currentInput, info.device,
          /* optional MemConfigAttr */ nullptr);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* If we should tilize and the input data type is bfloat16, tilize and
     * typecast on device */
    if (info.shouldTilize() and input.dataType == DataType::BFloat16) {
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* If we should tilize and the input data type is not bfloat16 and we want
       to read back from device do everything on host */
    if (info.shouldTilize() and input.dataType != DataType::BFloat16 and
        opsToCreate.createFromDeviceOp) {
      currentInput =
          this->createFromDeviceOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    /* If we should tilize and the input data type is not bfloat 16 and we don't
       want to read back from device: tilize on host, move back to device, and
       typecast on device */
    if (info.shouldTilize() and input.dataType != DataType::BFloat16 and
        not opsToCreate.createFromDeviceOp) {
      // Force-create a FromDeviceOp
      currentInput =
          this->createOp<ttnn::FromDeviceOp>(op, rewriter, currentInput);
      // tilize on host
      currentInput =
          this->createToLayoutOpIfNeeded(op, rewriter, currentInput, info);
      // move back to device and convert data type/memory config if needed
      currentInput = this->createOp<ttnn::ToDeviceOp>(
          op, rewriter, currentInput, info.device,
          /* optional MemConfigAttr */ nullptr);
      currentInput =
          this->createTypecastOpIfNeeded(op, rewriter, currentInput, info);
      currentInput = this->createToMemoryConfigOpIfNeeded(op, rewriter,
                                                          currentInput, info);
      op.getResult().replaceAllUsesWith(currentInput);
      return;
    }

    llvm_unreachable("Unreachable code path");
  }

  void handleDeviceInputLayoutConversion(ttnn::ToLayoutOp op,
                                         IRRewriter &rewriter,
                                         mlir::Value currentInput,
                                         const OpCreationInfo &info) const {
    const OpsToCreate &opsToCreate = info.opsToCreate;
    if (not opsToCreate.createToLayoutOp and not opsToCreate.createTypecastOp) {
      handleDeviceInputNoLayoutNoTypecast(op, rewriter, currentInput, info);
      return;
    }
    if (opsToCreate.createToLayoutOp and not opsToCreate.createTypecastOp) {
      handleDeviceInputLayoutNoTypecast(op, rewriter, currentInput, info);
      return;
    }
    if (not opsToCreate.createToLayoutOp and opsToCreate.createTypecastOp) {
      handleDeviceInputNoLayoutTypecast(op, rewriter, currentInput, info);
      return;
    }
    if (opsToCreate.createToLayoutOp and opsToCreate.createTypecastOp) {
      handleDeviceInputLayoutTypecast(op, rewriter, currentInput, info);
      return;
    }
    llvm_unreachable("Unreachable code path");
  }

  /*
   * Logic for creating ops. Conditions/constraints include:
   * - When possible, we want to execute operations on device.
   * - Tilize on device requires dataformat of BFLOAT16.
   * - Typecast on device requires TILIZED tensor.
   * - Untilize on device requires even width, and page size >
   *   sizeof(uint32_t). For now, we will always untilize on host. We rarely
   * need device to device untilize, so the perf hit should be acceptable.
   */
  void createLayoutConversionOps(ttnn::ToLayoutOp op,
                                 IRRewriter &rewriter) const {
    auto [input, output] = getInputOutputLayouts(op);
    OpsToCreate opsToCreate = determineRequiredOps(input, output);
    assert(isCreationValid(op, input, output, opsToCreate) &&
           "Invalid layout conversion");
    auto device = op.getDevice();
    assert((device || output.isOnHost()) &&
           "Op device must be set for output tensors on device");
    OpCreationInfo info(device, input, output, opsToCreate);

    Value currentInput = op.getInput();

    if (input.isOnHost()) {
      handleHostInputLayoutConversion(op, rewriter, currentInput, info);
      return;
    }
    handleDeviceInputLayoutConversion(op, rewriter, currentInput, info);
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
        // TODO(svuckovic): Move from ttnn::EmptyOp to ttnn::OnesOp once #1476
        // lands
        //
        mlir::Value tensorValue = rewriter.create<ttnn::EmptyOp>(
            forwardFuncOp->getLoc(), tensorType, nullptr, shapeAttr, dTypeAttr,
            tensorLayoutAttr, nullptr);

        generatedTensors.push_back(tensorValue);
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
      mlir::TypeRange returnTypeRange = mlir::TypeRange(rewriter.getI32Type());
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
      // arguments are packed into a single tuple
      //
      mlir::FunctionType originalFuncType = forwardFuncOp.getFunctionType();
      assert(
          std::all_of(originalFuncType.getInputs().begin(),
                      originalFuncType.getInputs().end(),
                      [](Type t) { return mlir::isa<RankedTensorType>(t); }) &&
          "Expected all inputs must be of type RankedTensorType");
      mlir::TupleType inputTupleType =
          mlir::TupleType::get(&getContext(), originalFuncType.getInputs());
      FunctionType tuplifiedFuncType =
          originalFuncType.clone(inputTupleType, originalFuncType.getResults());
      rewriter.modifyOpInPlace(forwardFuncOp,
                               [&forwardFuncOp, &tuplifiedFuncType]() {
                                 forwardFuncOp.setType(tuplifiedFuncType);
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
      Block &entryBlock = forwardFuncOp.getBlocks().front();
      entryBlock.insertArgument(/*index=*/0u,
                                tuplifiedFuncType.getInputs().front(),
                                forwardFuncOp.getLoc());

      rewriter.setInsertionPointToStart(&entryBlock);
      for (size_t idx = 0; idx < originalFuncType.getInputs().size(); idx++) {
        ::mlir::tt::GetTupleElementOp getTupleElementOp =
            rewriter.create<mlir::tt::GetTupleElementOp>(
                forwardFuncOp.getLoc(), forwardFuncOp.getArgument(0), idx);

        rewriter.replaceAllUsesWith(entryBlock.getArgument(1 + idx),
                                    getTupleElementOp);
      }

      // Erase original arguments
      //
      entryBlock.eraseArguments(1, originalFuncType.getInputs().size());
    }
  }
};

} // namespace mlir::tt::ttnn
