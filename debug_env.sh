# Debug Environment Variables for TT-MLIR Issue #3849
# Source this file to set up debugging environment

# MLIR Debug Options
export MLIR_ENABLE_DUMP=1                    # Enable operation dumping
export MLIR_ENABLE_TIMING=1                  # Enable timing statistics
export MLIR_ENABLE_CRASH_REPRODUCER=1        # Generate crash reproducers
export MLIR_ENABLE_IR_PRINTING=1             # Print IR during transformations
export MLIR_ENABLE_LOCATION_SNAPSHOT=1       # Capture location information

# AddressSanitizer Options
export ASAN_OPTIONS="abort_on_error=1:detect_leaks=1:check_initialization_order=1:detect_stack_use_after_return=1:symbolize=1"

# UndefinedBehaviorSanitizer Options  
export UBSAN_OPTIONS="print_stacktrace=1:abort_on_error=1:symbolize=1"

# Memory Sanitizer Options (if built with MemorySanitizer)
export MSAN_OPTIONS="abort_on_error=1:print_stats=1:symbolize=1"

# Debug Logging Levels
export TTMLIR_DEBUG_LEVEL=3                  # Highest debug level
export TTMLIR_ENABLE_DEBUG_LOGS=ON           # Enable debug logs

# Crash Handling
export LLVM_ENABLE_CRASH_OVERRIDES=1         # Enable crash handling
export LLVM_DISABLE_CRASH_REPORT=0           # Enable crash reports

# Performance Profiling (optional)
export TT_RUNTIME_ENABLE_PERF_TRACE=OFF      # Keep off for debugging

# Build Configuration
export CMAKE_BUILD_TYPE=Debug
export TTMLIR_ENABLE_RUNTIME=OFF             # Disable runtime for debugging
export TTMLIR_ENABLE_PYKERNEL=ON             # Enable Python kernels for golden tests
export TTMLIR_ENABLE_STABLEHLO=OFF           # Disable StableHLO to reduce complexity
export TTMLIR_ENABLE_OPMODEL=OFF             # Disable OpModel for debugging
export CODE_COVERAGE=ON                      # Enable coverage for analysis

# Test Configuration
export PYTHON_TEST_VERBOSE=1                 # Verbose Python test output
export LIT_VERBOSE=1                         # Verbose LIT test output

echo "Debug environment configured for TT-MLIR Issue #3849"
echo "Key settings:"
echo "  - MLIR debugging: ENABLED"
echo "  - AddressSanitizer: ENABLED"
echo "  - UBSanitizer: ENABLED"
echo "  - Debug logs: ENABLED"
echo "  - Crash reproducers: ENABLED"
