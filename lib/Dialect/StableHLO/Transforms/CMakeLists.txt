add_mlir_dialect_library(MLIRSTABLEHLOTransforms
  AnalyzeMesh.cpp
  ApplyArgumentShardStatus.cpp
  ConvertGSPMDToSDY.cpp
  ShardyCCLToStableHLOCCLPatterns.cpp
  UpdateGlobalToLocalShapes.cpp
  WrapUnderManualComputation.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir
  ${TTMLIR_TOOLCHAIN_DIR}/src/shardy

  LINK_LIBS PUBLIC
  TTMLIRSTABLEHLOUtils
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
