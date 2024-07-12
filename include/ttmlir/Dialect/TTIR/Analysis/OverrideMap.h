#include "llvm/Support/JSON.h"

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_OVERRIDEMAP_H
#define TTMLIR_DIAlECT_TTIR_ANALYSIS_OVERRIDEMAP_H

namespace mlir::tt {

class OverrideMap {
private:
  static OverrideMap *instance;
  OverrideMap() {}
  llvm::json::Object *overrideMap;

public:
  static OverrideMap *getInstance();
  void fromJSON(llvm::StringRef jsonString);
  bool containsOverride(llvm::StringRef attrName);
  llvm::json::Object *getOverride(llvm::StringRef attrName);
};

OverrideMap *OverrideMap::instance = nullptr;

} // namespace mlir::tt

#endif
