add_subdirectory(unittests)

llvm_canonicalize_cmake_booleans(
    MLIR_ENABLE_BINDINGS_PYTHON
    TTMLIR_ENABLE_BINDINGS_PYTHON
)

# Configure into build dir
#
configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Configure into source dir
#
configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Configure into unit tests dir
#
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

# Add custom target to compile everything needed for tests but not execute it
add_custom_target(compile-ttmlir-tests DEPENDS ${TTMLIR_TEST_DEPENDS})

add_lit_testsuite(check-ttmlir "Running the ttmlir regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    --verbose
    --xunit-xml-output report.xml
    DEPENDS ${TTMLIR_TEST_DEPENDS}
)
set_target_properties(check-ttmlir PROPERTIES FOLDER "Tests")

# Add custom target to execute tests without compiling
add_custom_target(run-ttmlir-tests COMMAND ${CMAKE_CTEST_COMMAND} -R check-ttmlir DEPENDS check-ttmlir)

# Add custom target for optimizer models performance tests
add_custom_target(check-perf
    COMMAND ${CMAKE_COMMAND} -E env SYSTEM_DESC_PATH=${CMAKE_SOURCE_DIR}/ttrt-artifacts/system_desc.ttsys
            llvm-lit -v --param TTMLIR_ENABLE_OPTIMIZER_MODELS_PERF_TESTS=1
            ${CMAKE_CURRENT_SOURCE_DIR}/ttmlir/Silicon/TTNN/n150/optimizer/models_perf_tests
    DEPENDS ${TTMLIR_TEST_DEPENDS} ttrt
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running optimizer models performance tests (resnet, yolo_v8, segformer)"
)
set_target_properties(check-perf PROPERTIES FOLDER "Tests")

add_lit_testsuites(TTMLIRCompiler ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TTMLIR_TEST_DEPENDS})

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ DESTINATION test COMPONENT Test EXCLUDE_FROM_ALL)

# Process CMakeLists for PyKernel Demo Target
if (TTMLIR_ENABLE_PYKERNEL)
    add_subdirectory(pykernel)
endif()
