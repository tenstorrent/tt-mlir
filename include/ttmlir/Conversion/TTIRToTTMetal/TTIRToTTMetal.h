
#ifndef TTMLIR_CONVERSION_TTIRTOTTMETAL_TTIRTOTTMETAL_H
#define TTMLIR_CONVERSION_TTIRTOTTMETAL_TTIRTOTTMETAL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

void populateTTIRToTTMetalPatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTMetalPass();

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTMETAL_TTIRTOTTMETAL_H
