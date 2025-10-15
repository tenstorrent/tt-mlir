add_mlir_dialect_library(MLIRStableHLOTransforms
  AnalyzeMesh.cpp
  ApplyArgumentShardStatus.cpp
  NonSolvedGraphSdyPasses.cpp
  ConvertXlaSdyToSdy.cpp
  ShardyCCLToStableHLOCCLPatterns.cpp
  UpdateGlobalToLocalShapes.cpp
  WrapUnderManualComputation.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir
  ${TTMLIR_TOOLCHAIN_DIR}/src/shardy

  LINK_LIBS PUBLIC
  TTMLIRStableHLOUtils
  SdyDialect
  SdyRegister
  SdyTransformsPropagationOpShardingRuleBuilder
  SdyTransformsPropagationPasses
  SdyTransformsPropagationOpShardingRuleRegistry
  SdyTransformsPropagationShardingGroupMap
  SdyTransformsPropagationShardingProjection
  SdyTransformsPropagationAutoPartitionerRegistry
  SdyTransformsPropagationUtils
  SdyTransformsPropagationBasicFactorPropagation
  SdyTransformsPropagationAggressiveFactorPropagation
  SdyTransformsCommonOpProperties
  SdyTransformsCommonShardingWalker
  SdyTransformsPropagationDebuggingSourceSharding
  SdyCommonFileUtils
  SdyTransformsExportPasses
  SdyTransformsImportPasses
  SdyTransformsPasses
)
