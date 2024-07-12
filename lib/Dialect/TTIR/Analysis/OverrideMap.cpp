// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/OverrideMap.h"

namespace mlir::tt {

OverrideMap *OverrideMap::getInstance() {
  if (instance == nullptr) {
    instance = new OverrideMap();
  }
  return instance;
}

void OverrideMap::fromJSON(llvm::StringRef jsonString) {
  llvm::Expected<llvm::json::Value> json = llvm::json::parse(jsonString);
  if (!json) {
    llvm::errs() << "Error parsing JSON: " << json.takeError() << "\n";
    return;
  }
  overrideMap = json->getAsObject();
}

bool OverrideMap::containsOverride(llvm::StringRef attrName) {
  return overrideMap->find(attrName) != overrideMap->end();
}

llvm::json::Object *OverrideMap::getOverride(llvm::StringRef attrName) {
  return overrideMap->get(attrName)->getAsObject();
}

} // namespace mlir::tt
