# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /localdev/vwells/ttmlir-toolchain-x86/venv/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /localdev/vwells/ttmlir-toolchain-x86/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vwells/sources/tt-mlir

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vwells/sources/tt-mlir

# Utility rule file for ttrt.

# Include any custom commands dependencies for this target.
include runtime/tools/python/CMakeFiles/ttrt.dir/compiler_depend.make

# Include the progress variables for this target.
include runtime/tools/python/CMakeFiles/ttrt.dir/progress.make

runtime/tools/python/CMakeFiles/ttrt:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "python ttrt package"
	cd /home/vwells/sources/tt-mlir/runtime/tools/python && rm -f build/*.whl
	cd /home/vwells/sources/tt-mlir/runtime/tools/python && python -m pip install -r requirements.txt
	cd /home/vwells/sources/tt-mlir/runtime/tools/python && TTMLIR_ENABLE_RUNTIME=OFF TT_RUNTIME_ENABLE_TTNN=ON TT_RUNTIME_ENABLE_TTMETAL=ON TT_RUNTIME_ENABLE_PERF_TRACE=OFF TT_RUNTIME_DEBUG=OFF TT_RUNTIME_WORKAROUNDS=OFF TTMLIR_VERSION_MAJOR=0 TTMLIR_VERSION_MINOR=0 TTMLIR_VERSION_PATCH=538 SOURCE_ROOT=/home/vwells/sources/tt-mlir python -m pip wheel . --wheel-dir build --verbose
	cd /home/vwells/sources/tt-mlir/runtime/tools/python && python -m pip install build/*.whl --force-reinstall

runtime/tools/python/CMakeFiles/ttrt.dir/codegen:
.PHONY : runtime/tools/python/CMakeFiles/ttrt.dir/codegen

ttrt: runtime/tools/python/CMakeFiles/ttrt
ttrt: runtime/tools/python/CMakeFiles/ttrt.dir/build.make
.PHONY : ttrt

# Rule to build all files generated by this target.
runtime/tools/python/CMakeFiles/ttrt.dir/build: ttrt
.PHONY : runtime/tools/python/CMakeFiles/ttrt.dir/build

runtime/tools/python/CMakeFiles/ttrt.dir/clean:
	cd /home/vwells/sources/tt-mlir/runtime/tools/python && $(CMAKE_COMMAND) -P CMakeFiles/ttrt.dir/cmake_clean.cmake
.PHONY : runtime/tools/python/CMakeFiles/ttrt.dir/clean

runtime/tools/python/CMakeFiles/ttrt.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/runtime/tools/python /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/runtime/tools/python /home/vwells/sources/tt-mlir/runtime/tools/python/CMakeFiles/ttrt.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : runtime/tools/python/CMakeFiles/ttrt.dir/depend

