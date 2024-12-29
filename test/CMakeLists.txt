add_subdirectory(unittests)

llvm_canonicalize_cmake_booleans(
    MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/unittests/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/unittests/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/unittests/lit.cfg.py
)

set(TTMLIR_TEST_DEPENDS
    FileCheck count not
    ttmlir-opt
    MLIRUnitTests
    ttmlir-translate
)
if(MLIR_ENABLE_BINDINGS_PYTHON AND TTMLIR_ENABLE_BINDINGS_PYTHON)
    list(APPEND TTMLIR_TEST_DEPENDS TTMLIRPythonModules)
endif()

add_lit_testsuite(check-ttmlir "Running the ttmlir regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    --xunit-xml-output report.xml
    DEPENDS ${TTMLIR_TEST_DEPENDS}
)
set_target_properties(check-ttmlir PROPERTIES FOLDER "Tests")

add_lit_testsuites(TTMLIRStatic ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TTMLIR_TEST_DEPENDS})

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ DESTINATION test COMPONENT Test EXCLUDE_FROM_ALL)
install(DIRECTORY $ENV{TT_METAL_BUILD_HOME}/lib/ DESTINATION third_party/tt-metal/src/tt-metal-build/lib COMPONENT EmitCTestLibs EXCLUDE_FROM_ALL)
install(DIRECTORY $ENV{TT_METAL_HOME}/ DESTINATION third_party/tt-metal/src/tt-metal COMPONENT EmitCTestLibs EXCLUDE_FROM_ALL)
