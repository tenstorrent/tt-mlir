add_mlir_dialect_library(MLIRSHLOTransforms
  ShardyAutomaticParallelization.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir
  ${TTMLIR_TOOLCHAIN_DIR}/src/shardy
  ${TTMLIR_TOOLCHAIN_DIR}/src/shardy-build

  LINK_LIBS PUBLIC
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
