// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Utils/StableHLOUtils.h"
#include "mlir/IR/IRMapping.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"

namespace mlir::tt::stablehlo::utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

mlir::func::FuncOp createPrivateFunction(
    mlir::ModuleOp module, mlir::StringRef namePrefix, mlir::StringRef baseName,
    mlir::ArrayRef<mlir::Value> captures, mlir::ArrayRef<mlir::Value> escapes,
    mlir::ArrayRef<mlir::Operation *> ops) {

  mlir::OpBuilder builder(module.getContext());
  mlir::IRMapping mapping;

  // Build function type: (captures) -> (escapes types)
  llvm::SmallVector<mlir::Type> argumentTypes, resultTypes;
  argumentTypes.reserve(captures.size());
  resultTypes.reserve(escapes.size());
  for (mlir::Value v : captures) {
    argumentTypes.push_back(v.getType());
  }
  for (mlir::Value v : escapes) {
    resultTypes.push_back(v.getType());
  }

  std::string fnName;
  {
    llvm::raw_string_ostream os(fnName);
    os << namePrefix << baseName;
  }

  auto fnType = builder.getFunctionType(argumentTypes, resultTypes);
  auto func = mlir::func::FuncOp::create(module.getLoc(), fnName, fnType);
  func.setPrivate();
  module.push_back(func);

  // Create entry block and map captures to block arguments.
  mlir::Block *entry = func.addEntryBlock();
  for (auto it : llvm::enumerate(captures)) {
    mapping.map(it.value(),
                entry->getArgument(static_cast<unsigned>(it.index())));
  }

  // Clone ops in order into the new function.
  mlir::OpBuilder internalBuilder(entry, entry->end());
  for (mlir::Operation *op : ops) {
    mlir::Operation *cloned = internalBuilder.clone(*op, mapping);
    // Strip the reoutline attrs inside the callee.
    if (cloned->hasAttr(sharding_utils::kReoutlineGroupAttr)) {
      cloned->removeAttr(sharding_utils::kReoutlineGroupAttr);
    }
    if (cloned->hasAttr(sharding_utils::kReoutlineSeedAttr)) {
      cloned->removeAttr(sharding_utils::kReoutlineSeedAttr);
    }
  }

  // Emit return with remapped escape values.
  llvm::SmallVector<mlir::Value> retVals;
  retVals.reserve(escapes.size());
  for (mlir::Value esc : escapes) {
    mlir::Value escVal = mapping.lookupOrNull(esc);
    retVals.push_back(escVal);
  }
  internalBuilder.create<mlir::func::ReturnOp>(func.getLoc(), retVals);

  return func;
}
#endif // TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::stablehlo::utils
