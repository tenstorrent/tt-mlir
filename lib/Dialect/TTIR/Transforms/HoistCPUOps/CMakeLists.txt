add_mlir_dialect_library(MLIRTTIRHoistCPUOps
        HoistCPUOps.cpp
        CPUHoistManuallyTagged.cpp
        CPUHoistConstEval.cpp
        CPUHoistNonLowerableSHLOOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTCoreOpsIncGen
)
