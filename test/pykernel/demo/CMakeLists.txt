# Check if required tt-mlir options are enabled
if(NOT TTMLIR_ENABLE_RUNTIME)
    message(WARNING "TTMLIR_ENABLE_RUNTIME must be ON to build the PyKernel demo")
endif()

if(NOT TTMLIR_ENABLE_PYKERNEL)
    message(WARNING "TTMLIR_ENABLE_PYKERNEL must be ON to build the PyKernel demo")
endif()

# Run the test
add_custom_target(pykernel-demo
    COMMENT "Running Pykernel Tests"
    COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR}/test.py
)

# Add info message
add_custom_command(
    TARGET pykernel-demo
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "PyKernel demo completed successfully!"
    COMMENT "Demo finished")
