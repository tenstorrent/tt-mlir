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
include lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/progress.make

# Include the compile flags for this target's objects.
include lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/flags.make

lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/codegen:
.PHONY : lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/codegen

# Object files for target MLIRTTDialect
MLIRTTDialect_OBJECTS =

# External object files for target MLIRTTDialect
MLIRTTDialect_EXTERNAL_OBJECTS = \
"/home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.o" \
"/home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.o" \
"/home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.o"

lib/libMLIRTTDialect.a: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOpsTypes.cpp.o
lib/libMLIRTTDialect.a: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTDialect.cpp.o
lib/libMLIRTTDialect.a: lib/Dialect/TT/IR/CMakeFiles/obj.MLIRTTDialect.dir/TTOps.cpp.o
lib/libMLIRTTDialect.a: lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/build.make
lib/libMLIRTTDialect.a: lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library ../../../libMLIRTTDialect.a"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && $(CMAKE_COMMAND) -P CMakeFiles/MLIRTTDialect.dir/cmake_clean_target.cmake
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MLIRTTDialect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/build: lib/libMLIRTTDialect.a
.PHONY : lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/build

lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/clean:
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR && $(CMAKE_COMMAND) -P CMakeFiles/MLIRTTDialect.dir/cmake_clean.cmake
.PHONY : lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/clean

lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR /home/vwells/sources/tt-mlir/lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lib/Dialect/TT/IR/CMakeFiles/MLIRTTDialect.dir/depend
