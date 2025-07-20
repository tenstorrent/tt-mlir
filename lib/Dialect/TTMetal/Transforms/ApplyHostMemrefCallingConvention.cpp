// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Transforms/ApplyHostMemrefCallingConvention.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_APPLYHOSTMEMREFCALLINGCONVENTION
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"
} // namespace mlir::tt::ttmetal

namespace {
class MemrefGlobalRewriter
    : public mlir::OpConversionPattern<mlir::memref::GlobalOp> {
public:
  using mlir::OpConversionPattern<mlir::memref::GlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::GlobalOp op,
                  mlir::memref::GlobalOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {

    rewriter.startOpModification(op);

    mlir::MemRefType currentType = op.getType();
    auto newType = mlir::cast<mlir::MemRefType>(
        getTypeConverter()->convertType(currentType));

    op.setType(newType);
    rewriter.finalizeOpModification(op);
    return mlir::success();
  }
};

struct ApplyHostMemrefCallingConvention
    : public mlir::tt::ttmetal::impl::ApplyHostMemrefCallingConventionBase<
          ApplyHostMemrefCallingConvention> {
  void runOnOperation() final {
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::Type type) { return type; });
    typeConverter.addConversion([](mlir::MemRefType memrefType) {
      if (!mlir::dyn_cast_if_present<mlir::tt::ttcore::MemorySpaceAttr>(
              memrefType.getMemorySpace())) {

        mlir::SmallVector<int64_t> strides(memrefType.getShape().size(), 1);
        int64_t stride = 1;
        // Keep last stride as 1.
        for (int64_t i = static_cast<int64_t>(strides.size()) - 2; i >= 0;
             --i) {
          int64_t dimShape = memrefType.getShape()[i + 1];

          if ((strides.size() >= 2 &&
               static_cast<size_t>(i) == strides.size() - 2)) {
            // Align up second to last shape dimension to TILE_WIDTH.

            dimShape = ttmlir::utils::alignUp(
                dimShape, static_cast<int64_t>(mlir::tt::ttnn::TILE_WIDTH));
          }

          stride *= dimShape;
          strides[i] = stride;
        }

        mlir::MemRefType memrefTypeWithStrides = mlir::MemRefType::get(
            memrefType.getShape(), memrefType.getElementType(),
            mlir::StridedLayoutAttr::get(memrefType.getContext(), 0, strides));

        return memrefTypeWithStrides;
      }
      return memrefType;
    });

    patterns.add<mlir::tt::ttir::UniformTypeRewriter, MemrefGlobalRewriter>(
        typeConverter, context);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace mlir::tt::ttmetal {

std::unique_ptr<OperationPass<ModuleOp>>
createApplyHostMemrefCallingConventionPass() {
  return std::make_unique<ApplyHostMemrefCallingConvention>();
}

} // namespace mlir::tt::ttmetal
