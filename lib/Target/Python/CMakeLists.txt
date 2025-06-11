add_mlir_translation_library(EmitPyTargetPython
    TranslateToPythonRegistration.cpp
    TranslateToPython.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/Python

    LINK_LIBS PUBLIC
    MLIREMITPYDialect
    MLIRFuncDialect
)
