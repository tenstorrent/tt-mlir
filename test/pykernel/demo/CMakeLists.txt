# Check if required tt-mlir options are enabled
if(NOT TTMLIR_ENABLE_RUNTIME)
    message(WARNING "TTMLIR_ENABLE_RUNTIME must be ON to build the PyKernel demo")
endif()

if(NOT TTMLIR_ENABLE_PYKERNEL)
    message(WARNING "TTMLIR_ENABLE_PYKERNEL must be ON to build the PyKernel demo")
endif()

# Define a target for compiling pykernel dependencies only
add_custom_target(pykernel-demo-compile
    COMMENT "Compiling PyKernel demo dependencies"
    DEPENDS TTPykernelModules)

# Define a target for installing requirements and running the demo tests
add_custom_target(pykernel-demo-run
    COMMENT "Installing requirements and running PyKernel demo tests"
    COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR}/test.py
    DEPENDS pykernel)

# Add info message for demo completion
add_custom_command(
    TARGET pykernel-demo-run
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "PyKernel demo completed successfully!"
    COMMENT "Demo finished")

# Keep the original target for backward compatibility
add_custom_target(pykernel-demo
    COMMENT "Running PyKernel demo tests (backward compatibility)"
    DEPENDS pykernel-demo-run)
