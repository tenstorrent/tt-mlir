add_mlir_dialect_library(MLIRD2MTransforms
        Allocate.cpp
        DMAOptimizations.cpp
        DecomposeArange.cpp
        DecomposeMasking.cpp
        DecomposeTopk.cpp
        GenericApplyInterchange.cpp
        GenericFusion.cpp
        GenericLinearizeMemref.cpp
        GenericRegionsToFuncs.cpp
        HoistCBAllocs.cpp
        GenericReplaceGlobals.cpp
        GenericTileComputeLoops.cpp
        GridSelection.cpp
        InsertDstRegisterAccess/Shared.cpp
        InsertDstRegisterAccess/Unscheduled.cpp
        InsertDstRegisterAccess/Scheduled.cpp
        InsertTileMatmulBlock.cpp
        LinalgToAffine.cpp
        LowerScratchAllocate.cpp
        LowerToLayout/LowerToLayout.cpp
        LowerToLayout/Plan.cpp
        MaterializeViewReturns.cpp
        OptimizeMasks.cpp
        OpScheduler.cpp
        PythonRewritesPass.cpp
        ReblockGenerics.cpp
        SFPUTileLoopFission.cpp
        InsertScratchBuffers.cpp
        InsertSpillAndScratch.cpp
        ScalarizeConstTensors.cpp
        GenerateOuterLoops.cpp
        LowerToExplicitForm.cpp
        LowerMulticastLoads.cpp
        AssignThreads.cpp
        SplitThreads.cpp
        InsertComputeCB.cpp
        SplitUnifiedThread.cpp
        PreallocateMcastSemaphores.cpp
        ScheduleDMA.cpp
        LowerDMAToFullyIndexedForm.cpp
        LowerLoadStoreOpsToDMA.cpp
        ExpandDMAReadCompositeView.cpp
        MarkSynchronizedBuffers.cpp
        NormalizeThreadArgs.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRD2MOpsIncGen
        MLIRD2MPassesIncGen
        MLIRTTCoreOpsIncGen

        LINK_LIBS PUBLIC
        MLIRArithDialect
        MLIRD2MAnalysis
        MLIRD2MUtils
        MLIRTTUtils
        )

# D2MPythonRewritesPass embeds CPython to dispatch into d2m-jit. Gated
# on TTMLIR_ENABLE_D2M_JIT so consumers that don't build d2m-jit don't
# take the Python link dep. The .cpp itself is always compiled — it
# degrades to a fail-fast stub when the define is absent (so ttmlir-opt
# advertises the flag with a clear diagnostic).
#
# We use the raw CPython C-API rather than nanobind/pybind11 because
# MLIRD2MTransforms inherits LLVM's -fno-rtti / -fno-exceptions, which
# nanobind doesn't support. Raw CPython is uglier but the surface we
# need is tiny.
if(TTMLIR_ENABLE_D2M_JIT)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)

  target_compile_definitions(obj.MLIRD2MTransforms PRIVATE TTMLIR_ENABLE_D2M_JIT)
  # The Python bindings in python/CMakeLists.txt are built with
  # MLIR_PYTHON_PACKAGE_PREFIX=ttmlir., which is baked into the capsule
  # names used by mlir-c/Bindings/Python/Interop.h. The pass needs to
  # produce capsules whose names match the consumer side, so we set the
  # same define here.
  target_compile_definitions(obj.MLIRD2MTransforms PRIVATE MLIR_PYTHON_PACKAGE_PREFIX=ttmlir.)
  target_include_directories(obj.MLIRD2MTransforms PRIVATE ${Python_INCLUDE_DIRS})
  target_link_libraries(MLIRD2MTransforms PRIVATE Python::Python)
endif()
