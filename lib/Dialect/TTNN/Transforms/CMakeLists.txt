if (TTMLIR_ENABLE_OPMODEL)
        add_definitions(-DTTMLIR_ENABLE_OPMODEL)
        set(OPMODEL_SRCS
                Fusing/RoPEFusingPattern.cpp
                Fusing/SDPAFusingPattern.cpp
                Fusing/SplitQKVFusingPatterns.cpp)
endif()

# Embedded-Python tt-lang resolver. The pass class itself
# (TTNNResolveTTLangKernels.cpp) is `-fno-rtti -fno-exceptions` like the
# rest of MLIR; the pybind11 call surface is split out into
# TTNNResolveTTLangKernelsPython.cpp so that single file can be compiled
# `-frtti -fexceptions` (required by pybind11) without leaking RTTI into
# the `mlir::Pass` hierarchy, which would trip `undefined typeinfo for
# mlir::Pass` at link time because the rest of MLIR is `-fno-rtti`.
#
# `Development` (a superset of Development.Module + Development.Embed) is
# required for the libpython link so `Py_*` symbols resolve when this
# pass is loaded into a binary that didn't bring its own libpython (e.g.
# `ttmlir-opt`).
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Locate pybind11 in the active Python environment unless the caller has
# already pointed CMake at one. Matches the discovery pattern used by the
# tt-xla PJRT plugin.
if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _ttmlir_pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _ttmlir_pybind11_rc
    )
    if(_ttmlir_pybind11_rc EQUAL 0 AND _ttmlir_pybind11_cmakedir)
        set(pybind11_DIR "${_ttmlir_pybind11_cmakedir}" CACHE PATH "pybind11 CMake config dir")
    endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

add_mlir_dialect_library(MLIRTTNNTransforms
        Passes.cpp
        TTNNResolveTTLangKernels.cpp
        TTNNResolveTTLangKernelsPython.cpp
        TTNNLowerTTLangToGeneric.cpp
        OptimizerPasses/OperationValidationAndFallback.cpp
        OptimizerPasses/Optimizer.cpp
        OptimizerPasses/DevicePassesWrapper.cpp
        OptimizerPasses/GreedyL1SpillManagement.cpp
        OptimizerPasses/GreedyMemoryLayoutPropagation.cpp
        OptimizerPasses/RowMajorLayoutPropagation.cpp
        OptimizerPasses/TTNNPrepareConv2dWeightsAndBias.cpp
        OptimizerPasses/TTNNDeduceMoEComputeLayouts.cpp
        OptimizerPasses/TTNNPrepareConv3dWeights.cpp
        TTNNAdjustDeallocs.cpp
        TTNNAllocateDistributedOpBuffers.cpp
        TTNNAllocateDistributedOpSemaphores.cpp
        TTNNCollaspeD2M.cpp
        TTNNResolveComposites.cpp
        TTNNCollectPerfMetrics.cpp
        TTNNCreateD2MSubgraphs.cpp
        TTNNFileSplit.cpp
        TTNNPrepareConstEvalCaching.cpp
        TTNNDecomposeLayouts.cpp
        Decomposition/ConcatenateHeadsDecompositionRewritePattern.cpp
        Decomposition/DistributedRMSNormDecompositionRewritePattern.cpp
        Decomposition/DistributedLayerNormDecompositionRewritePattern.cpp
        Decomposition/GroupNormDecompositionRewritePattern.cpp
        Decomposition/SDPADecodeDecompositionPattern.cpp
        Decomposition/SDPADecompositionPattern.cpp
        Decomposition/TopKDecompositionRewritePattern.cpp
        Decomposition/TTNNDecompositionPass.cpp
        OpValidator.cpp
        TTNNEmitPyWorkarounds.cpp
        TTNNConstEvalInputsToSystemMemory.cpp
        TTNNFusing.cpp
        ${OPMODEL_SRCS}
        TTNNLayout.cpp
        TTNNConfigureCCLOps.cpp
        TTNNRecoverStructure.cpp
        TTNNRecoverStructureUtils.cpp
        TTNNSetComputeKernelConfig.cpp
        TTNNMemoryManagement.cpp
        TTNNToCpp.cpp
        TTNNToPython.cpp
        TTNNTraceHoistTransform.cpp
        TTNNUniqueLocs.cpp
        TTNNWeightDtypeConversion.cpp
        TTNNKVCacheDtypeConversion.cpp
        TTNNActivationDtypeLowering.cpp
        Workarounds/Decomposition/AllGatherOpRewritePattern.cpp
        Workarounds/Decomposition/AllReduceReshapeOpRewritePattern.cpp
        Workarounds/Decomposition/AllToAllDispatchMetadataDrainCoreRewritePattern.cpp
        Workarounds/Decomposition/MoeGptLayoutRewritePattern.cpp
        Workarounds/Decomposition/MoeComputeRewritePattern.cpp
        Workarounds/Decomposition/PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern.cpp
        Workarounds/Decomposition/PagedUpdateCacheOpRewritePattern.cpp
        Workarounds/Decomposition/PointToPointOpRewritePattern.cpp
        Workarounds/Decomposition/SplitQueryKeyValueAndSplitHeadsOpRewritePattern.cpp
        Workarounds/Decomposition/EmbeddingOpSqueezeWeightRewritePattern.cpp
        Workarounds/Decomposition/GatherOpRank1RewritePattern.cpp
        Workarounds/Decomposition/SamplingOpRank2RewritePattern.cpp
        Workarounds/Decomposition/GroupNormAffineReshapeRewritePattern.cpp
        Workarounds/Decomposition/GroupNormChannelPadRewritePattern.cpp
        Workarounds/Decomposition/IntegerProdOpRewritePattern.cpp
        Workarounds/Decomposition/LinearOpRewritePattern.cpp
        Workarounds/Decomposition/NLPConcatHeadsDecodeInputRewritePattern.cpp
        Workarounds/Decomposition/ReduceScatterOpRewritePattern.cpp
        Workarounds/Decomposition/SliceStaticOpRewritePattern.cpp
        Workarounds/Decomposition/ScaledDotProductAttentionDecodeAttentionSinkRewritePattern.cpp
        Workarounds/Decomposition/ScaledDotProductAttentionDecodeBroadcastMaskRewritePattern.cpp
        Workarounds/Decomposition/ScaledDotProductAttentionPadTileDimsRewritePattern.cpp
        Workarounds/Decomposition/UpsampleOpRewritePattern.cpp
        Workarounds/Decomposition/RMSNormConfigRewritePattern.cpp
        Workarounds/Decomposition/TopKRouterGptDecompositionRewritePattern.cpp
        Workarounds/Decomposition/DistributedRMSNormWidthShardInputRewritePattern.cpp
        Workarounds/Decomposition/ReduceScatterConfigRewritePattern.cpp
        Workarounds/Decomposition/RotaryEmbeddingOpRewritePattern.cpp
        Workarounds/Decomposition/PadHighDimRewritePattern.cpp
        Workarounds/TTNNWorkaroundsPatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTNNPassesIncGen
        MLIRTTCoreOpsIncGen

        LINK_LIBS
        PUBLIC
        MLIRTTNNDialect
        MLIRTTNNAnalysis
        MLIRTTNNDiagnostics
        MLIRTTCoreDialect
        PRIVATE
        TTMLIRTTIRToTTNN
        MLIRTTNNValidation
        TTMLIRTTNNToEmitC
        TTMLIRTTIRToEmitPy
        TTMLIRTTNNToEmitPy
        )

# `add_mlir_dialect_library` wraps the source list in an
# `obj.MLIRTTNNTransforms` internal target, so a plain
# `target_compile_options` on `MLIRTTNNTransforms` doesn't reach the
# underlying compile rule. Stamp the flags directly on the source file
# instead -- this also keeps the carve-out local: the rest of
# `MLIRTTNNTransforms` (including the pass class itself, which inherits
# from the no-RTTI `mlir::Pass`) stays `-fno-rtti -fno-exceptions` and
# unaware of pybind11.
#
#   * `-frtti`        -- pybind11 uses `typeid` pervasively.
#   * `-fexceptions`  -- pybind11's `error_already_set` propagates
#                        via `throw`; LLVM/MLIR libraries default
#                        to `-fno-exceptions`.
set_source_files_properties(
    "TTNNResolveTTLangKernelsPython.cpp"
    PROPERTIES
        COMPILE_OPTIONS "-frtti;-fexceptions"
)

# `add_mlir_dialect_library` doesn't expose its created `obj.` target in
# a way that we can attach include dirs via the public CMake LINK_LIBS
# knob without also dragging the link library into every downstream
# consumer. Use a direct include-only attachment for this single TU so
# the pybind11 / Python headers are visible without making
# `pybind11::embed` a public lib interface (the libpython link is added
# explicitly to `TTMLIRCompiler` / `TTMLIRCompilerStatic` in
# `lib/CMakeLists.txt`, which is where downstream tools actually pick it
# up).
target_include_directories(obj.MLIRTTNNTransforms PRIVATE
    $<TARGET_PROPERTY:pybind11::pybind11,INTERFACE_INCLUDE_DIRECTORIES>)
