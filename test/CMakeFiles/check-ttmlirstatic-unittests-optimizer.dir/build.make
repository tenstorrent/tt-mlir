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

# Utility rule file for check-ttmlirstatic-unittests-optimizer.

# Include any custom commands dependencies for this target.
include test/CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/progress.make

test/CMakeFiles/check-ttmlirstatic-unittests-optimizer:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running lit suite /home/vwells/sources/tt-mlir/test/unittests/Optimizer"
	cd /home/vwells/sources/tt-mlir/test && /localdev/vwells/ttmlir-toolchain-x86/venv/bin/python3 /localdev/vwells/ttmlir-toolchain-x86/bin/llvm-lit -sv /home/vwells/sources/tt-mlir/test/unittests/Optimizer

test/CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/codegen:
.PHONY : test/CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/codegen

check-ttmlirstatic-unittests-optimizer: test/CMakeFiles/check-ttmlirstatic-unittests-optimizer
check-ttmlirstatic-unittests-optimizer: test/CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/build.make
.PHONY : check-ttmlirstatic-unittests-optimizer

# Rule to build all files generated by this target.
test/CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/build: check-ttmlirstatic-unittests-optimizer
.PHONY : test/CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/build

test/CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/clean:
	cd /home/vwells/sources/tt-mlir/test && $(CMAKE_COMMAND) -P CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/clean

test/CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/test /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/test /home/vwells/sources/tt-mlir/test/CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : test/CMakeFiles/check-ttmlirstatic-unittests-optimizer.dir/depend

