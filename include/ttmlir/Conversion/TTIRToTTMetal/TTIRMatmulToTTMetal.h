#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

namespace mlir::tt::ttmetal {
class TTIRToTTMetalMatmulRewriter {
    public:
      LogicalResult matchAndRewrite(ttir::MatmulOp op,
                                    PatternRewriter &rewriter) const;
};
} // namespace mlir::tt::ttmetal