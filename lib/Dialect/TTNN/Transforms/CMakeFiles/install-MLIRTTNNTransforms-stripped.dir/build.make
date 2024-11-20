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

# Utility rule file for install-MLIRTTNNTransforms-stripped.

# Include any custom commands dependencies for this target.
include lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/progress.make

lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped:
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTNN/Transforms && /localdev/vwells/ttmlir-toolchain-x86/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -DCMAKE_INSTALL_COMPONENT="MLIRTTNNTransforms" -DCMAKE_INSTALL_DO_STRIP=1 -P /home/vwells/sources/tt-mlir/cmake_install.cmake

lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/codegen:
.PHONY : lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/codegen

install-MLIRTTNNTransforms-stripped: lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped
install-MLIRTTNNTransforms-stripped: lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/build.make
.PHONY : install-MLIRTTNNTransforms-stripped

# Rule to build all files generated by this target.
lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/build: install-MLIRTTNNTransforms-stripped
.PHONY : lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/build

lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/clean:
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTNN/Transforms && $(CMAKE_COMMAND) -P CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/cmake_clean.cmake
.PHONY : lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/clean

lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Dialect/TTNN/Transforms /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Dialect/TTNN/Transforms /home/vwells/sources/tt-mlir/lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lib/Dialect/TTNN/Transforms/CMakeFiles/install-MLIRTTNNTransforms-stripped.dir/depend

