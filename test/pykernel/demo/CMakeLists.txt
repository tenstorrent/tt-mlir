# Check if required tt-mlir options are enabled
if(NOT TTMLIR_ENABLE_RUNTIME)
    message(WARNING "TTMLIR_ENABLE_RUNTIME must be ON to build the PyKernel demo")
endif()

if(NOT TTMLIR_ENABLE_PYKERNEL)
    message(WARNING "TTMLIR_ENABLE_PYKERNEL must be ON to build the PyKernel demo")
endif()

find_program(GREP_EXEC grep REQUIRED)
set(GREP_PATTERN [[^(git\+|^-r|^#)]])
set(METAL_REQUIREMENTS_PATH ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/python_env/requirements-dev.txt)

# Define a target for the demo depending on Runtime and Python Bindings
add_custom_target(pykernel-demo
    COMMENT "Setting up and running PyKernel demo"
    COMMAND ${GREP_EXEC} -vE '${GREP_PATTERN}' ${METAL_REQUIREMENTS_PATH} > ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt
    DEPENDS TTMLIRRuntime TTMLIRPythonModules)

add_custom_command(TARGET pykernel-demo
    COMMENT "Installing Requirements for Demo"
    COMMAND PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu pip install -r ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt
)

# Run the test
add_custom_command(TARGET pykernel-demo
    COMMENT "Running Pykernel Tests"
    COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR}/test.py
)

add_custom_command(TARGET pykernel-demo
    COMMENT "Running Multi-core Matrix Mult. Test with Pykernel"
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_demo.py
)

# Add info message
add_custom_command(
    TARGET pykernel-demo
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "PyKernel demo completed successfully!"
    COMMENT "Demo finished")
