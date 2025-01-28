// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinTypes.h>

namespace mlir {
namespace tt {
namespace sharding_utils {

// Based on current MeshShardAttr info, finalize sharding dimensions.
static LogicalResult determineShardingDims(MeshShardAttr &meshShardAttr,
                                           bool lastTileDimReplicate) {
  // This code is based on following assumption.
  // 1. Hardware mesh is two dimenion such as 2x4, 1x2, ...
  // 2. Hardware mesh only supports either line or mesh config
  // e.g., t3k 1x8 or 2x4
  std::vector<int64_t> shardShape = meshShardAttr.shardShape;
  if (meshShardAttr.lastTileDimReplicate) {
    meshShardAttr.shardShape.pop_back();
  }
  // Determine obvious properties first.
  bool reverseOrder = (meshShardAttr.meshShape.size() == 1) ? false : true;
  int64_t totalDevices = meshShardAttr.meshShape[0];
  if (reverseOrder) {
    totalDevices *= meshShardAttr.meshShape[1];
  }
  // Detect line device config (1xN).
  bool isLineDeviceConfig = false;
  for (auto s : shardShape) {
    if (s == totalDevices) {
      isLineDeviceConfig = true;
    }
  }
  // Detect hardware mesh if non-reverse order
  if (!reverseOrder) {
    if (isLineDeviceConfig) {
      // line device config must be 1xN, not Nx1
      meshShardAttr.meshShape = {1, meshShardAttr.meshShape[0]};
    } else {
      // e.g., shardShape [1,2,4] leads to [2,4]
      meshShardAttr.meshShape.clear();
      for (auto s : shardShape) {
        if (s != 1) {
          meshShardAttr.meshShape.push_back(s);
        }
      }
      assert(meshShardAttr.meshShape.size() == 2 &&
             "Unsupported hardware mesh config other than 2-d mesh");
    }
  }
  // Determine shardDims based on the meshShape
  meshShardAttr.shardDims = {-1, -1};
  for (uint64_t i = 0; i < meshShardAttr.shardShape.size(); ++i) {
    // Check sharding dimension only
    if (meshShardAttr.shardShape[i] != 1) {
      auto shardDimIdx = 0;
      if (meshShardAttr.shardShape[i] != meshShardAttr.meshShape[0]) {
        shardDimIdx = 1;
      }
      meshShardAttr.shardDims[shardDimIdx] = i;
    }
  }

  return success();
}

// Parse GSPMD devices string and fill out MeshShardAttr info.
static LogicalResult parseGSPMDDevicesStr(StringRef &str,
                                          MeshShardAttr &meshShardAttr) {
  auto [axesStr, restStr] = str.split("<=");
  // parse devices ex. [4,2,1]
  if (!axesStr.consume_front("[") || !axesStr.consume_back("]")) {
    return failure();
  }
  SmallVector<StringRef> dimsStr;
  axesStr.split(dimsStr, ",");
  for (auto dim : dimsStr) {
    int64_t d;
    if (dim.getAsInteger<int64_t>(10, d)) {
      return failure();
    }
    meshShardAttr.shardShape.push_back(d);
  }
  // parse devices ex. [8] or [2,4]T(1,0)
  SmallVector<StringRef> reshapeStr;
  restStr.split(reshapeStr, "T");
  if (!reshapeStr[0].consume_front("[") || !reshapeStr[0].consume_back("]")) {
    return failure();
  }
  dimsStr.clear();
  reshapeStr[0].split(dimsStr, ",");
  for (auto dim : dimsStr) {
    int64_t d;
    if (dim.getAsInteger<int64_t>(10, d)) {
      return failure();
    }
    meshShardAttr.meshShape.push_back(d);
  }
  return success();
}

// OpenXLA has its own lexer, but we will use simple string-based parser here.
// This parsing is mainly based on "Sharding Attribute" section in
// https://github.com/sdasgup3/stablehlo/blob/80082431d1af0933e6202ecc8a6f8801e039235b/docs/spec.md
LogicalResult parseGSPMDShardingAttr(StringRef shardingStr,
                                     MeshShardAttr &meshShardAttr) {
  MeshShardType shardType = mlir::tt::MeshShardType::Manual;
  bool lastTileDimReplicate = false;

  // Parse sting and tokenize.
  if (!shardingStr.consume_front("{") || !shardingStr.consume_back("}")) {
    return failure();
  }
  SmallVector<StringRef> shardingStrTokens;
  shardingStr.split(shardingStrTokens, " ");

  // Parse tokens.
  for (auto str : shardingStrTokens) {
    if (str.contains("manual")) {
      assert(shardType == mlir::tt::MeshShardType::Manual &&
             "Fail to parse sharding info.");
      // manual: already sharded, so no action is needed
      meshShardAttr.shardShape.push_back(1);
    } else if (str.contains("replicated")) {
      assert(shardType == mlir::tt::MeshShardType::Manual &&
             "Fail to parse sharding info.");
      // replicated: all devices have whole data
      shardType = mlir::tt::MeshShardType::Replicate;
      meshShardAttr.shardShape.push_back(1);
    } else if (str.contains("maximal")) {
      assert(shardType == mlir::tt::MeshShardType::Manual &&
             "Fail to parse sharding info.");
      // maximal: one device has whole data
      shardType = mlir::tt::MeshShardType::Maximal;
      meshShardAttr.shardShape.push_back(1);
    } else if (str.contains("device=")) {
      // maximal should followed by "device" to put data on
      assert(shardType == mlir::tt::MeshShardType::Maximal &&
             "Fail to parse sharding info.");
      int64_t d;
      if (!str.consume_front("device=")) {
        return failure();
      }
      if (str.getAsInteger<int64_t>(10, d)) {
        return failure();
      }
      meshShardAttr.shardShape.push_back(d);
    } else if (str.contains("devices=")) {
      // other: "devices" detail sharding plan
      assert(shardType == mlir::tt::MeshShardType::Manual &&
             "Fail to parse sharding info.");
      shardType = mlir::tt::MeshShardType::Devices;
      if (!str.consume_front("devices=")) {
        return failure();
      }
      if (failed(parseGSPMDDevicesStr(str, meshShardAttr))) {
        return failure();
      }
    } else if (str.contains("last_tile_dim_replicate")) {
      assert(shardType == mlir::tt::MeshShardType::Devices &&
             "Fail to parse sharding info.");
      // other: replicate last tile dim
      lastTileDimReplicate = true;
    }
  }
  meshShardAttr.shardType = shardType;
  meshShardAttr.lastTileDimReplicate = lastTileDimReplicate;

  // Parse devices
  if (meshShardAttr.shardType == mlir::tt::MeshShardType::Devices) {
    return determineShardingDims(meshShardAttr, lastTileDimReplicate);
  }

  meshShardAttr.shardDims.push_back(-1);
  meshShardAttr.meshShape.push_back(-1);
  return success();
}

} // namespace sharding_utils
} // namespace tt
} // namespace mlir
