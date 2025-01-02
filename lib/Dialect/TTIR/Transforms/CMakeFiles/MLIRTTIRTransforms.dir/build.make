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

# Include any dependencies generated for this target.
include lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/progress.make

# Include the compile flags for this target's objects.
include lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/flags.make

lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/codegen:
.PHONY : lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/codegen

# Object files for target MLIRTTIRTransforms
MLIRTTIRTransforms_OBJECTS =

# External object files for target MLIRTTIRTransforms
MLIRTTIRTransforms_EXTERNAL_OBJECTS = \
"/home/vwells/sources/tt-mlir/lib/Dialect/TTIR/Transforms/CMakeFiles/obj.MLIRTTIRTransforms.dir/Allocate.cpp.o" \
"/home/vwells/sources/tt-mlir/lib/Dialect/TTIR/Transforms/CMakeFiles/obj.MLIRTTIRTransforms.dir/Constant.cpp.o" \
"/home/vwells/sources/tt-mlir/lib/Dialect/TTIR/Transforms/CMakeFiles/obj.MLIRTTIRTransforms.dir/Generic.cpp.o" \
"/home/vwells/sources/tt-mlir/lib/Dialect/TTIR/Transforms/CMakeFiles/obj.MLIRTTIRTransforms.dir/Layout.cpp.o" \
"/home/vwells/sources/tt-mlir/lib/Dialect/TTIR/Transforms/CMakeFiles/obj.MLIRTTIRTransforms.dir/Transforms.cpp.o" \
"/home/vwells/sources/tt-mlir/lib/Dialect/TTIR/Transforms/CMakeFiles/obj.MLIRTTIRTransforms.dir/Utility.cpp.o"

lib/libMLIRTTIRTransforms.a: lib/Dialect/TTIR/Transforms/CMakeFiles/obj.MLIRTTIRTransforms.dir/Allocate.cpp.o
lib/libMLIRTTIRTransforms.a: lib/Dialect/TTIR/Transforms/CMakeFiles/obj.MLIRTTIRTransforms.dir/Constant.cpp.o
lib/libMLIRTTIRTransforms.a: lib/Dialect/TTIR/Transforms/CMakeFiles/obj.MLIRTTIRTransforms.dir/Generic.cpp.o
lib/libMLIRTTIRTransforms.a: lib/Dialect/TTIR/Transforms/CMakeFiles/obj.MLIRTTIRTransforms.dir/Layout.cpp.o
lib/libMLIRTTIRTransforms.a: lib/Dialect/TTIR/Transforms/CMakeFiles/obj.MLIRTTIRTransforms.dir/Transforms.cpp.o
lib/libMLIRTTIRTransforms.a: lib/Dialect/TTIR/Transforms/CMakeFiles/obj.MLIRTTIRTransforms.dir/Utility.cpp.o
lib/libMLIRTTIRTransforms.a: lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/build.make
lib/libMLIRTTIRTransforms.a: lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library ../../../libMLIRTTIRTransforms.a"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/Transforms && $(CMAKE_COMMAND) -P CMakeFiles/MLIRTTIRTransforms.dir/cmake_clean_target.cmake
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/Transforms && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MLIRTTIRTransforms.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/build: lib/libMLIRTTIRTransforms.a
.PHONY : lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/build

lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/clean:
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/Transforms && $(CMAKE_COMMAND) -P CMakeFiles/MLIRTTIRTransforms.dir/cmake_clean.cmake
.PHONY : lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/clean

lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/Transforms /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/Transforms /home/vwells/sources/tt-mlir/lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lib/Dialect/TTIR/Transforms/CMakeFiles/MLIRTTIRTransforms.dir/depend
