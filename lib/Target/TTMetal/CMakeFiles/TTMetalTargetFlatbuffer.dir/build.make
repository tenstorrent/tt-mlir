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
include lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/progress.make

# Include the compile flags for this target's objects.
include lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/flags.make

lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/codegen:
.PHONY : lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/codegen

# Object files for target TTMetalTargetFlatbuffer
TTMetalTargetFlatbuffer_OBJECTS =

# External object files for target TTMetalTargetFlatbuffer
TTMetalTargetFlatbuffer_EXTERNAL_OBJECTS = \
"/home/vwells/sources/tt-mlir/lib/Target/TTMetal/CMakeFiles/obj.TTMetalTargetFlatbuffer.dir/TTMetalToFlatbuffer.cpp.o" \
"/home/vwells/sources/tt-mlir/lib/Target/TTMetal/CMakeFiles/obj.TTMetalTargetFlatbuffer.dir/TTMetalToFlatbufferRegistration.cpp.o"

lib/libTTMetalTargetFlatbuffer.a: lib/Target/TTMetal/CMakeFiles/obj.TTMetalTargetFlatbuffer.dir/TTMetalToFlatbuffer.cpp.o
lib/libTTMetalTargetFlatbuffer.a: lib/Target/TTMetal/CMakeFiles/obj.TTMetalTargetFlatbuffer.dir/TTMetalToFlatbufferRegistration.cpp.o
lib/libTTMetalTargetFlatbuffer.a: lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/build.make
lib/libTTMetalTargetFlatbuffer.a: lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library ../../libTTMetalTargetFlatbuffer.a"
	cd /home/vwells/sources/tt-mlir/lib/Target/TTMetal && $(CMAKE_COMMAND) -P CMakeFiles/TTMetalTargetFlatbuffer.dir/cmake_clean_target.cmake
	cd /home/vwells/sources/tt-mlir/lib/Target/TTMetal && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TTMetalTargetFlatbuffer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/build: lib/libTTMetalTargetFlatbuffer.a
.PHONY : lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/build

lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/clean:
	cd /home/vwells/sources/tt-mlir/lib/Target/TTMetal && $(CMAKE_COMMAND) -P CMakeFiles/TTMetalTargetFlatbuffer.dir/cmake_clean.cmake
.PHONY : lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/clean

lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Target/TTMetal /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Target/TTMetal /home/vwells/sources/tt-mlir/lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lib/Target/TTMetal/CMakeFiles/TTMetalTargetFlatbuffer.dir/depend

