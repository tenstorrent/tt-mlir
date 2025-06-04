# Check if required tt-mlir options are enabled
if(NOT TTMLIR_ENABLE_RUNTIME)
    message(FATAL_ERROR "TTMLIR_ENABLE_RUNTIME must be ON to build the PyKernel demo")
endif()

if(NOT TTMLIR_ENABLE_PYKERNEL)
    message(FATAL_ERROR "TTMLIR_ENABLE_PYKERNEL must be ON to build the PyKernel demo")
endif()

# Define a target for the demo depending on Runtime and Python Bindings
add_custom_target(pykernel-demo
    COMMENT "Setting up and running PyKernel demo"
    COMMAND ${CMAKE_COMMAND} -E env PYTHONUNBUFFERED=1 pip install --progress-bar on -v -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
    DEPENDS TTMLIRRuntime TTMLIRPythonModules)

# Run the test
add_custom_command(TARGET pykernel-demo
    COMMENT "Running Eltwise SFPU (Exp) Test with Pykernel"
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_sfpu_demo.py
)

# Add info message
add_custom_command(
    TARGET pykernel-demo
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "PyKernel demo completed successfully!"
    COMMENT "Demo finished")
