add_mlir_dialect_library(MLIREMITPYDialect
        EmitPyDialect.cpp
        EmitPyOps.cpp
        EmitPyTypes.cpp
        EmitPyAttrs.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIREmitPyOpsIncGen
        MLIREmitPyAttrsIncGen
        MLIRTTCoreOpsIncGen
        )
