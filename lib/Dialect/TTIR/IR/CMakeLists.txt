add_mlir_dialect_library(MLIRTTIRDialect
        TTIRDialect.cpp
        TTIROps.cpp
        TTIROpsTypes.cpp
        TTIROpsInterfaces.cpp
        TTIRTraits.cpp
        TTIRGenericRegionOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir
        ${TTMLIR_TOOLCHAIN_DIR}/src/shardy
        ${TTMLIR_TOOLCHAIN_DIR}/src/shardy-build

        DEPENDS
        MLIRTTIROpsIncGen
        TTIRGenericRegionOpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTOpsIncGen

        LINK_LIBS PUBLIC
        MLIRTTDialect
        MLIRBufferizationTransforms
        SdyDialect
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
        )
