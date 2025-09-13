// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "llvm/Support/Error.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::gspmd_utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

// Parse meshes from the GSPMD module.
llvm::Expected<llvm::SmallVector<llvm::SmallVector<int64_t>>>
parseMeshesFromGspmdModule(mlir::ModuleOp &module) {
  llvm::SmallVector<llvm::SmallVector<int64_t>> meshes;

  // Walk through the module and find all GSPMD mesh annotations.
  mlir::WalkResult moduleResult = module.walk([&](func::FuncOp funcOp) {
    // Mesh could be determined by either the mhlo.sharding attribute on the
    // arguments or results.
    for (BlockArgument arg : funcOp.getArguments()) {
      if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
        if (!currentArgAttrDict.contains(
                mlir::tt::gspmd_utils::kXlaShardingAttr)) {
          continue;
        }

        // Once extracted, we can generate the GSPMDMeshSharding object.
        auto opShardingAttr = dyn_cast_if_present<mlir::StringAttr>(
            currentArgAttrDict.get(mlir::tt::gspmd_utils::kXlaShardingAttr));
        if (!opShardingAttr) {
          return WalkResult::interrupt();
        }

        llvm::Expected<mlir::tt::gspmd_utils::GSPMDMeshSharding>
            gspmdMeshSharding =
                mlir::tt::gspmd_utils::GSPMDMeshSharding::generate(
                    opShardingAttr.getValue(), opShardingAttr.getValue(),
                    mlir::tt::ttcore::ShardStatus::Unsharded,
                    mlir::tt::ttcore::MeshShardDirection::FullToShard);
        if (auto err = gspmdMeshSharding.takeError()) {
          return WalkResult::interrupt();
        }

        // Some stablehlo custom calls may not have a mesh shape, so we
        // skip those.
        if (!gspmdMeshSharding->getMeshShape().empty() &&
            gspmdMeshSharding->getMeshShape() !=
                llvm::SmallVector<int64_t>({-1})) {
          meshes.push_back(
              llvm::SmallVector<int64_t>(gspmdMeshSharding->getMeshShape()));
        }
      }
    }

    mlir::WalkResult funcOpResult = funcOp.walk([&](Operation *op) {
      if (!mlir::isa<mlir::stablehlo::CustomCallOp>(op)) {
        return WalkResult::advance();
      }

      mlir::stablehlo::CustomCallOp customCallOp =
          mlir::cast<mlir::stablehlo::CustomCallOp>(op);

      // Check call target name to see if it's the one we are interested in.
      auto callTargetName = customCallOp.getCallTargetNameAttr();
      if (callTargetName != gspmd_utils::kSPMDFullToShardShapeCallTargetName &&
          callTargetName != gspmd_utils::kSPMDShardToFullShapeCallTargetName) {
        return WalkResult::advance();
      }

      // Set the shard direction.
      mlir::tt::ttcore::MeshShardDirection shardDirection =
          mlir::tt::ttcore::MeshShardDirection::ShardToFull;
      if (callTargetName ==
          mlir::tt::gspmd_utils::kSPMDFullToShardShapeCallTargetName) {
        shardDirection = mlir::tt::ttcore::MeshShardDirection::FullToShard;
      }

      // We want to extract the mhlo.sharding attribute from the
      // CustomCallOp.
      auto opShardingAttr = dyn_cast_if_present<mlir::StringAttr>(
          customCallOp.getOperation()->getAttr(
              mlir::tt::gspmd_utils::kXlaShardingAttr));
      if (!opShardingAttr) {
        return WalkResult::interrupt();
      }

      // We also want to extract the mhlo.sharding attribute from this op's
      // @Sharding operand.
      auto shardingOperand = customCallOp->getOperand(0);
      auto definingOp =
          shardingOperand.getDefiningOp<mlir::stablehlo::CustomCallOp>();
      auto operandShardingAttr = dyn_cast_if_present<mlir::StringAttr>(
          definingOp.getOperation()->getAttr(
              mlir::tt::gspmd_utils::kXlaShardingAttr));

      if (!operandShardingAttr) {
        return WalkResult::interrupt();
      }

      // We also extract the shard status from the @Sharding op.
      auto shardStatusAttr =
          dyn_cast_if_present<mlir::tt::ttcore::ShardStatusAttr>(
              definingOp.getOperation()->getAttr(
                  mlir::tt::ttcore::ShardStatusAttr::name));

      // Insert default sharding status if not present.
      if (!shardStatusAttr) {
        shardStatusAttr = mlir::tt::ttcore::ShardStatusAttr::get(
            customCallOp.getContext(),
            mlir::tt::ttcore::ShardStatus::Unsharded);
      }

      // Once extracted, we can generate the GSPMDMeshSharding object.
      llvm::Expected<mlir::tt::gspmd_utils::GSPMDMeshSharding>
          gspmdMeshSharding =
              mlir::tt::gspmd_utils::GSPMDMeshSharding::generate(
                  opShardingAttr.getValue(), operandShardingAttr.getValue(),
                  shardStatusAttr.getValue(), shardDirection);
      if (auto err = gspmdMeshSharding.takeError()) {
        return WalkResult::interrupt();
      }

      // Some stablehlo custom calls may not have a mesh shape, so we
      // skip those.
      if (!gspmdMeshSharding->getMeshShape().empty() &&
          gspmdMeshSharding->getMeshShape() !=
              llvm::SmallVector<int64_t>({-1})) {
        meshes.push_back(
            llvm::SmallVector<int64_t>(gspmdMeshSharding->getMeshShape()));
      }

      return WalkResult::advance();
    });

    if (funcOpResult.wasInterrupted()) {
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (moduleResult.wasInterrupted()) {
    return llvm::createStringError(
        "Error parsing GSPMD annotations to determine the mesh shape.");
  }

  // Create 1x1 mesh if no meshes were found.
  if (meshes.empty()) {
    meshes.push_back(llvm::SmallVector<int64_t>({1, 1}));
  }

  return meshes;
}

// Check if the module has any gspmd annotations.
bool gspmdAnnotationsExist(mlir::ModuleOp &module) {
  for (auto &op : module.getBody()->getOperations()) {
    if (!mlir::isa<func::FuncOp>(op)) {
      continue;
    }

    func::FuncOp funcOp = mlir::cast<func::FuncOp>(op);
    // Check if mhlo.sharding exists for any of the arguments.
    for (BlockArgument arg : funcOp.getBody().front().getArguments()) {
      if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
        if (currentArgAttrDict.contains(
                mlir::tt::gspmd_utils::kXlaShardingAttr)) {
          return true;
        }
      }
    }

    // Check if mhlo.sharding exists for any of the results.
    mlir::FunctionType funcType = funcOp.getFunctionType();
    for (uint32_t i = 0; i < funcType.getNumResults(); i++) {
      if (auto resultAttrDict = mlir::DictionaryAttr::get(
              module.getContext(), funcOp.getResultAttrs(i))) {
        if (resultAttrDict.contains(mlir::tt::gspmd_utils::kXlaShardingAttr)) {
          return true;
        }
      }
    }

    mlir::WalkResult result = funcOp.getBody().walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::stablehlo::CustomCallOp>(op)) {
        auto customCall = mlir::cast<mlir::stablehlo::CustomCallOp>(op);
        auto callTarget = customCall.getCallTargetName();

        if (callTarget ==
                mlir::tt::gspmd_utils::kShardingCustomCallTargetName ||
            callTarget ==
                mlir::tt::gspmd_utils::kSPMDFullToShardShapeCallTargetName ||
            callTarget ==
                mlir::tt::gspmd_utils::kSPMDShardToFullShapeCallTargetName) {
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      return true;
    }
  }

  return false;
}

// Check if the module has frontend SDY attributes.
bool hasFrontendSdyAttributes(mlir::ModuleOp &module) {
  mlir::WalkResult result = module.walk([&](func::FuncOp funcOp) {
    for (BlockArgument arg : funcOp.getArguments()) {
      if (auto currentArgAttrDict = funcOp.getArgAttrDict(arg.getArgNumber())) {
        if (currentArgAttrDict.contains(kFrontendAttributesAttr)) {
          auto frontendAttrs = currentArgAttrDict.get(kFrontendAttributesAttr);
          if (auto dictAttr =
                  mlir::dyn_cast<mlir::DictionaryAttr>(frontendAttrs)) {
            if (dictAttr.contains(sharding_utils::kXlaSdyShardingAttr)) {
              return WalkResult::interrupt();
            }
          }
        }
      }
    }
    return WalkResult::advance();
  });

  return result.wasInterrupted();
}

// Update @Sharding custom call with the shard status for the argument.
void updateShardStatusForArgument(MLIRContext *context,
                                  mlir::BlockArgument &arg,
                                  mlir::NamedAttribute shardStatusNamedAttr) {
  // Check all users of the argument. If the user is a stablehlo.custom_call
  // op, we want to update the @Sharding op with the shard status.
  for (auto *user : arg.getUsers()) {
    if (!mlir::isa<mlir::stablehlo::CustomCallOp>(user)) {
      return;
    }

    // Skip non @Sharding custom calls.
    mlir::stablehlo::CustomCallOp shardingOp =
        mlir::cast<mlir::stablehlo::CustomCallOp>(user);
    if (shardingOp.getCallTargetName() !=
        mlir::tt::gspmd_utils::kShardingCustomCallTargetName) {
      return;
    }

    llvm::SmallVector<mlir::NamedAttribute> newCustomOpAttrs(
        shardingOp->getAttrDictionary().getValue());
    newCustomOpAttrs.push_back(shardStatusNamedAttr);
    shardingOp->setAttrs(mlir::DictionaryAttr::get(context, newCustomOpAttrs));
  }
}

// Update @Sharding custom call with the shard status for the result.
void updateShardStatusForResult(MLIRContext *context, func::FuncOp &funcOp,
                                uint32_t resultIdx,
                                mlir::NamedAttribute shardStatusNamedAttr) {
  // Check if terminator exists and has a defining op (in case of simple graphs
  // with no ops).
  Block &entryBlock = funcOp.getBody().front();
  Operation *terminator = entryBlock.getTerminator();
  if (!terminator) {
    return;
  }
  Value operand = terminator->getOperand(resultIdx);
  Operation *resultDefiningOp = operand.getDefiningOp();
  if (!resultDefiningOp ||
      !mlir::isa<mlir::stablehlo::CustomCallOp>(resultDefiningOp)) {
    return;
  }

  // Check all users of the result. If the user is a stablehlo.custom_call
  // op, we want to update the @Sharding op with the shard status.
  // When iterating through the results, we will run into @SPMD* calls, who's
  // operands will be the @Sharding call.
  mlir::stablehlo::CustomCallOp customCallOp =
      mlir::cast<mlir::stablehlo::CustomCallOp>(resultDefiningOp);
  if (customCallOp.getCallTargetName() !=
          mlir::tt::gspmd_utils::kSPMDShardToFullShapeCallTargetName &&
      customCallOp.getCallTargetName() !=
          mlir::tt::gspmd_utils::kSPMDFullToShardShapeCallTargetName) {
    return;
  }

  mlir::stablehlo::CustomCallOp shardingOp =
      customCallOp->getOperand(0)
          .getDefiningOp<mlir::stablehlo::CustomCallOp>();
  llvm::SmallVector<mlir::NamedAttribute> newCustomOpAttrs(
      shardingOp->getAttrDictionary().getValue());
  newCustomOpAttrs.push_back(shardStatusNamedAttr);
  shardingOp->setAttrs(mlir::DictionaryAttr::get(context, newCustomOpAttrs));
}

// Parse GSPMD devices string and fill out MeshSharding info.
llvm::Expected<bool>
parseGSPMDDevicesStr(StringRef devicesStr,
                     llvm::SmallVector<int64_t> &shardShape,
                     llvm::SmallVector<int64_t> &meshShape,
                     llvm::SmallVector<int64_t> &deviceIds) {
  // This function extract dimensions from targetDimsStr "[x,y,z]" and saves it
  // to targetDims.
  auto parseDimsFromDimensionStr = [](StringRef targetDimsStr,
                                      SmallVector<int64_t> &targetDims,
                                      bool checkSquareBracket = true) -> bool {
    if (checkSquareBracket && (!targetDimsStr.consume_front("[") ||
                               !targetDimsStr.consume_back("]"))) {
      return false;
    }
    SmallVector<StringRef> dimsStr;
    targetDimsStr.split(dimsStr, ",");
    targetDims.clear();
    for (auto dim : dimsStr) {
      int64_t d;
      if (dim.getAsInteger<int64_t>(10, d)) {
        return false;
      }
      targetDims.push_back(d);
    }
    return true;
  };

  // devciesStr can be appended by either (1) TileAssignmentDevices or
  // (2) '<=[` IotaReshapeDimensions `]` [`T` (IotaTransposeDimensions)]
  bool reshapeDevicesStrParsing = devicesStr.contains("<=");

  // devicesStr is generated by splitting whole string using space " ". Thus,
  // it is not supposed to include any trailing space. e.g.,
  // "[4,2,1]<=[2,4]T(1,0)" or "[2,4]0,1,2,3,4,5,6,7".
  auto firstClosingBracketIdx = devicesStr.find(']');
  if (firstClosingBracketIdx <= 1 ||
      firstClosingBracketIdx == StringRef::npos) {
    return llvm::createStringError(
        "Fail to parse GSPMD devices string [x,y,..]: " + devicesStr);
  }
  auto axesStr = devicesStr.take_front(firstClosingBracketIdx + 1);
  auto restStr = devicesStr.drop_front(firstClosingBracketIdx + 1);
  // Parse devices string e.g., [4,2,1] or [2,4].
  if (!parseDimsFromDimensionStr(axesStr, shardShape)) {
    return llvm::createStringError("Fail to parse GSPMD devices axes string: " +
                                   axesStr);
  }

  if (reshapeDevicesStrParsing) {
    // Parse devices string after "<=" e.g., [8] or [2,4]T(1,0).
    auto [reshapeStr, unused] = restStr.drop_front(2).split("T");
    // Parse reshape[0] string e.g., [8] or [2,4].
    if (!parseDimsFromDimensionStr(reshapeStr, meshShape)) {
      return llvm::createStringError(
          "Fail to parse GSPMD devices reshape string: " + reshapeStr);
    }
    deviceIds.clear();
    // Parse devices string after "]" e.g., 0,1,2,3,4,5,6,7.
  } else if (!parseDimsFromDimensionStr(restStr, deviceIds, false)) {
    return llvm::createStringError("Fail to parse GSPMD device id string: " +
                                   restStr);
  } else {
    // Set meshShape as the size of deviceIds such as [8] because we cannot
    // determine meshShape from the list of device ids.
    meshShape.clear();
    meshShape.push_back(deviceIds.size());
  }
  return true;
}

// Based on current MeshSharding info, finalize sharding dimensions.
llvm::Expected<bool>
determineGSPMDShardingDims(llvm::SmallVector<int64_t> &shardShape,
                           llvm::SmallVector<int64_t> &shardDims,
                           llvm::SmallVector<int64_t> &meshShape,
                           llvm::SmallVector<int64_t> &deviceIds,
                           bool lastTileDimReplicate) {
  // This code is based on following assumption.
  // 1. Hardware mesh is two dimenion such as 2x4, 1x2, ...
  // 2. Hardware mesh only supports either line or mesh config
  // e.g., t3k 1x8 or 2x4
  SmallVector<int64_t> orgShardShape = shardShape;
  if (lastTileDimReplicate) {
    shardShape.pop_back();
  }
  // Determine obvious properties first.
  bool reverseOrder = meshShape.size() != 1;
  // totalDevices is the total number of multi-chips such as 8 for t3k. Thus, no
  // overflow is expected with int64_t.
  int64_t totalDevices =
      std::accumulate(meshShape.begin(), meshShape.end(), int64_t{1},
                      std::multiplies<int64_t>());
  // Detect line device config (1xN).
  bool isLineDeviceConfig =
      llvm::any_of(orgShardShape, [&](int64_t s) { return s == totalDevices; });
  // Detect hardware mesh. For reverse order sharding, meshShape already
  // includes hardware mesh. For non reverse order case, extract hardware mesh
  // by traversing from front to back and picking none-zero values.
  if (!reverseOrder) {
    if (isLineDeviceConfig) {
      // Device with line config must be 1xN, not Nx1.
      meshShape = {1, meshShape[0]};
    } else {
      meshShape.clear();
      // e.g., orgShardShape [1,2,4] or [2,1,4] leads to [2,4]
      llvm::copy_if(orgShardShape, std::back_inserter(meshShape),
                    [](int64_t s) { return s != int64_t{1}; });
      if (!deviceIds.empty() && deviceIds[0] + 1 != deviceIds[1]) {
        // transposed shardShape if devicIds are not consecutive, so reverse the
        // meshShape. [4,2] leads to [2,4]
        std::reverse(meshShape.begin(), meshShape.end());
        reverseOrder = true;
      }
    }
  }

  if (meshShape.size() != 2) {
    // Currently, we are only supporting 2d hardware mesh config.
    return llvm::createStringError(
        "Only support 2d hardware mesh config. mesh.size()=%d",
        meshShape.size());
  }

  // Determine shardDims based on the shardShape and meshShape.
  // shard_dims indicate in which dimension we shard the tensor. For T3K,
  // detected meshShape will be [2, 4] and shard_dims will be [ a, b ] depending
  // on the sharding intention.
  // For example, if shardShape is [1,2,1,4], shard_dims is supposed to be [1,
  // 3] or if shardShape is [1,4,1,2], then shard_dims should be [3, 1].
  shardDims.assign(meshShape.size(), -1);
  // Skip the first 1 of 1xN hardware.
  uint64_t shardingCnt = isLineDeviceConfig;
  for (uint64_t i = 0; i < shardShape.size(); ++i) {
    // Check sharding dimension only.
    if (shardShape[i] != 1) {
      auto shardDimIdx =
          (reverseOrder) ? (meshShape.size() - 1 - shardingCnt) : shardingCnt;
      // Positive shardShape[i] and meshShape[shardDimIdx] is supposed to be
      // identical.
      if (shardShape[i] > 0 && shardShape[i] != meshShape[shardDimIdx]) {
        return llvm::createStringError(
            "Fail to determine shardDims. shardShape[%d] (%d) != meshShape[%d] "
            "(%d)",
            i, shardShape[i], shardDimIdx, meshShape[shardDimIdx]);
      }
      shardDims[shardDimIdx] = i;
      shardingCnt++;
    }
  }

  return true;
}

// Generate default GSPMDMeshSharding.
llvm::Expected<GSPMDMeshSharding> GSPMDMeshSharding::generateDefault() {
  return GSPMDMeshSharding{mlir::tt::ttcore::MeshShardDirection::FullToShard,
                           mlir::tt::ttcore::MeshShardType::Identity,
                           /*shardShape=*/llvm::SmallVector<int64_t>{},
                           /*shardDims=*/llvm::SmallVector<int64_t>{},
                           /*meshShape=*/llvm::SmallVector<int64_t>{},
                           /*deviceIds=*/llvm::SmallVector<int64_t>{},
                           mlir::tt::ttcore::ShardStatus::Unsharded,
                           /*opShardingStr*/ "",
                           /*operandShardingStr*/ "",
                           /*lastTileDimReplicate*/ false};
}

// OpenXLA has its own lexer, but we will use simple string-based parser here.
// This parsing is mainly based on "Sharding Attribute" section in
// https://github.com/sdasgup3/stablehlo/blob/80082431d1af0933e6202ecc8a6f8801e039235b/docs/spec.md#sharding-attribute
llvm::Expected<gspmd_utils::GSPMDMeshSharding>
gspmd_utils::GSPMDMeshSharding::generate(
    llvm::StringRef opShardingStr, llvm::StringRef operandShardingStr,
    mlir::tt::ttcore::ShardStatus shardStatus,
    mlir::tt::ttcore::MeshShardDirection shardDirection) {
  // Need to parse GSPMD sharding string and fill out MeshSharding info.
  mlir::tt::ttcore::MeshShardType shardType =
      mlir::tt::ttcore::MeshShardType::Identity;
  llvm::SmallVector<int64_t> shardShape = {-1};
  llvm::SmallVector<int64_t> shardDims = {-1};
  llvm::SmallVector<int64_t> meshShape = {-1};
  llvm::SmallVector<int64_t> deviceIds = {-1};
  bool lastTileDimReplicate = false;

  // Parse opShardingStr and tokenize.
  if (!opShardingStr.consume_front("{") || !opShardingStr.consume_back("}")) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Fail to parse opShardingStr GSPMD sharding.");
  }
  llvm::SmallVector<llvm::StringRef> opShardingStrTokens;
  opShardingStr.split(opShardingStrTokens, " ");

  // Parse operandShardingStr and tokenize.
  if (!operandShardingStr.consume_front("{") ||
      !operandShardingStr.consume_back("}")) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Fail to parse operandShardingStr GSPMD sharding.");
  }
  llvm::SmallVector<llvm::StringRef> operandShardingStrTokens;
  operandShardingStr.split(operandShardingStrTokens, " ");

  // Our goal is to map opShardingStr and operandShardingStr to a mesh shard op
  // which is represented by MeshSharding object. If any argument is
  // pre-sharded, we will create an identity mesh shard op by default. This is
  // to ensure the shapes are consistent in the graph but runtime will not
  // attempt to shard the data again. opShardingStr and operandShardingStr may
  // or may not have last_tile_dim_replicate annotated.

  // clang-format off
  // opShardingStr and operandShardingStr will have one of the following
  // combinations. All others are not legal.
  // | opShardingStr | operandShardingStr |	support         | shard_type (if not pre-sharded)
  // | replicated	   |  manual	           |  yes	          | replicate
  // | maximal	     |  manual	           |  not supported | n/a
  // | devices	     |  manual	           |  yes	          | devices
  // | manual	       |  replicated	       |  yes	          | replicate
  // | manual	       |  maximal	           |  not supported | n/a
  // | manual	       |  devices	           |  yes           |   devices
  // clang-format on

  auto containsKeyword = [](const llvm::SmallVector<llvm::StringRef> &tokens,
                            std::string keyword) -> bool {
    return llvm::any_of(
        tokens, [&](llvm::StringRef str) { return str.contains(keyword); });
  };

  // Maximal is not supported with GSPMD sharding.
  if (containsKeyword(opShardingStrTokens, "maximal") ||
      containsKeyword(operandShardingStrTokens, "maximal")) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Fail to parse GSPMD sharding with maximal.");
  }

  // Check for replicated sharding.
  if (containsKeyword(opShardingStrTokens, "replicated") ||
      containsKeyword(operandShardingStrTokens, "replicated")) {
    shardType = ttcore::MeshShardType::Replicate;
    shardShape = llvm::SmallVector<int64_t>{1};
    shardDims = llvm::SmallVector<int64_t>{-1};
    meshShape = llvm::SmallVector<int64_t>{-1};
  }

  // Check for last_tile_dim_replicate.
  if (containsKeyword(opShardingStrTokens, "last_tile_dim_replicate") ||
      containsKeyword(operandShardingStrTokens, "last_tile_dim_replicate")) {
    lastTileDimReplicate = true;
  }

  // Check for devices sharding.
  if (containsKeyword(opShardingStrTokens, "devices=") ||
      containsKeyword(operandShardingStrTokens, "devices=")) {
    shardType = ttcore::MeshShardType::Devices;

    // Extract device ids from the devices string.
    llvm::SmallVector<llvm::StringRef> shardingStrTokens =
        containsKeyword(opShardingStrTokens, "devices=")
            ? opShardingStrTokens
            : operandShardingStrTokens;

    for (auto str : shardingStrTokens) {
      if (str.consume_front("devices=")) {
        // Parse the devices string and fill out shardShape, meshShape and
        // deviceIds.
        auto error = mlir::tt::gspmd_utils::parseGSPMDDevicesStr(
            str, shardShape, meshShape, deviceIds);
        if (auto e = error.takeError()) {
          return e;
        }
      }
    }

    // Determine shard dims for devices.
    auto error = mlir::tt::gspmd_utils::determineGSPMDShardingDims(
        shardShape, shardDims, meshShape, deviceIds, lastTileDimReplicate);
    if (auto e = error.takeError()) {
      return e;
    }
  }

  // Check if the input is already pre-sharded. If it is, override shardType to
  // Identity.
  shardType = shardStatus == mlir::tt::ttcore::ShardStatus::Presharded
                  ? ttcore::MeshShardType::Identity
                  : shardType;

  return gspmd_utils::GSPMDMeshSharding{
      shardDirection,      shardType,           shardShape,
      shardDims,           meshShape,           deviceIds,
      shardStatus,         opShardingStr.str(), operandShardingStr.str(),
      lastTileDimReplicate};
}

/// Parse axis definitions from SDY mesh string format.
///
/// Extracts axis name-size pairs from a mesh axes string that follows the SDY
/// format. The input string contains axis definitions in the format:
/// "axis_name"=size Multiple axes are separated by commas and/or spaces.
///
/// Example input formats:
///   - Single axis: ""batch"=4"
///   - Multiple axes: ""x"=2, "y"=4" or ""batch"=8 "model"=2"
///   - Whitespace variations: " "data" = 4 , "model" = 2 "
std::vector<std::pair<std::string, int64_t>>
parseAxisDefinitions(const std::string &axesContent) {
  std::vector<std::pair<std::string, int64_t>> axes;
  size_t pos = 0;

  while (pos < axesContent.length()) {
    // Skip whitespace and commas
    while (pos < axesContent.length() &&
           (axesContent[pos] == ' ' || axesContent[pos] == ',')) {
      pos++;
    }
    if (pos >= axesContent.length()) {
      break;
    }

    // Find opening quote for axis name
    size_t quoteStart = axesContent.find('"', pos);
    if (quoteStart == std::string::npos) {
      break;
    }

    // Find closing quote for axis name
    size_t quoteEnd = axesContent.find('"', quoteStart + 1);
    if (quoteEnd == std::string::npos) {
      break;
    }

    // Find equals sign after axis name
    size_t equalPos = axesContent.find('=', quoteEnd + 1);
    if (equalPos == std::string::npos) {
      break;
    }

    // Extract axis name between quotes
    std::string axisName =
        axesContent.substr(quoteStart + 1, quoteEnd - quoteStart - 1);

    // Find start of size value after equals
    size_t numStart = equalPos + 1;
    // Find end of size value
    size_t numEnd = axesContent.find_first_of(",] ", numStart);
    if (numEnd == std::string::npos) {
      numEnd = axesContent.length();
    }

    // Extract size string
    std::string sizeStr = axesContent.substr(numStart, numEnd - numStart);
    // Trim trailing whitespace from size string
    while (!sizeStr.empty() &&
           (sizeStr.back() == ' ' || sizeStr.back() == '\t')) {
      sizeStr.pop_back();
    }

    if (!sizeStr.empty()) {
      axes.push_back({axisName, std::stoi(sizeStr)});
    }

    pos = numEnd;
  }

  return axes;
}

/// Parse mesh information from mhlo.frontend_attributes and create sdy.mesh.
///
/// Extracts mesh configuration from MHLO frontend attributes and creates
/// corresponding SDY mesh operations in the MLIR module. The function looks for
/// "xla.sdy.meshes" attribute in the module's "mhlo.frontend_attributes" and
/// parses the mesh definition.
///
/// Expected input format in mhlo.frontend_attributes:
///   "xla.sdy.meshes": "mesh_name=<[axis_definitions]>"
///
/// Example frontend attribute values:
///   - 1D mesh: "mesh=<[\"batch\"=4]>"
///   - 2D mesh: "mesh=<[\"x\"=2, \"y\"=4]>"
///   - Complex: "mesh=<[\"data_parallel\"=8, \"model_parallel\"=2]>"
///
/// The function supports:
///   - 1D meshes (converted to 1xN format with updated axis name)
///   - 2D meshes (used directly)
///   - Skips processing if no frontend attributes found
mlir::LogicalResult
parseMeshFromFrontendAttributes(mlir::ModuleOp &rootModule,
                                mlir::MLIRContext *context) {
  mlir::DictionaryAttr moduleAttrs = rootModule->getAttrDictionary();
  mlir::Attribute frontendAttrs =
      moduleAttrs.get(gspmd_utils::kFrontendAttributesAttr);
  mlir::DictionaryAttr dictAttr =
      mlir::dyn_cast_if_present<mlir::DictionaryAttr>(frontendAttrs);
  mlir::StringAttr meshesStr =
      dictAttr ? mlir::dyn_cast_if_present<mlir::StringAttr>(
                     dictAttr.get(sharding_utils::kXlaSdyMeshesAttr))
               : nullptr;

  if (!moduleAttrs.contains(gspmd_utils::kFrontendAttributesAttr) ||
      !dictAttr || !dictAttr.contains(sharding_utils::kXlaSdyMeshesAttr) ||
      !meshesStr) {
    return mlir::success();
  }

  std::string meshStr = meshesStr.getValue().str();
  size_t startPos = meshStr.find("<[");
  size_t endPos = meshStr.find("]>");
  if (startPos == std::string::npos || endPos == std::string::npos) {
    return mlir::success();
  }

  std::string axesContent = meshStr.substr(startPos + 2, endPos - startPos - 2);
  std::vector<std::pair<std::string, int64_t>> axes =
      gspmd_utils::parseAxisDefinitions(axesContent);

  std::string meshName = std::string(sharding_utils::kDefaultMeshName);
  if (axes.size() == 1) {
    shardy_utils::addMeshToModule(rootModule, meshName,
                                  axes[0].first + "_updated", axes[0].first, 1,
                                  axes[0].second);
  } else if (axes.size() == 2) {
    shardy_utils::addMeshToModule(rootModule, meshName, axes[0].first,
                                  axes[1].first, axes[0].second,
                                  axes[1].second);
  } else {
    rootModule.emitError(
        "Unsupported mesh configuration: only 1D and 2D meshes are supported");
    return mlir::failure();
  }

  return mlir::success();
}

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::gspmd_utils
