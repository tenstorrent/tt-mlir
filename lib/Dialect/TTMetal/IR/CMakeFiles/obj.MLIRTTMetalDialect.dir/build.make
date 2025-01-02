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
include lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/progress.make

# Include the compile flags for this target's objects.
include lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/flags.make

lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/codegen:
.PHONY : lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/codegen

lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.o: lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/flags.make
lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.o: lib/Dialect/TTMetal/IR/TTMetalDialect.cpp
lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.o: lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.o"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.o -MF CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.o.d -o CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.o -c /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR/TTMetalDialect.cpp

lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.i"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR/TTMetalDialect.cpp > CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.i

lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.s"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR/TTMetalDialect.cpp -o CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.s

lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.o: lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/flags.make
lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.o: lib/Dialect/TTMetal/IR/TTMetalOps.cpp
lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.o: lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.o"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.o -MF CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.o.d -o CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.o -c /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR/TTMetalOps.cpp

lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.i"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR/TTMetalOps.cpp > CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.i

lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.s"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR/TTMetalOps.cpp -o CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.s

lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.o: lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/flags.make
lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.o: lib/Dialect/TTMetal/IR/TTMetalOpsTypes.cpp
lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.o: lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.o"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.o -MF CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.o.d -o CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.o -c /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR/TTMetalOpsTypes.cpp

lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.i"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR/TTMetalOpsTypes.cpp > CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.i

lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.s"
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR/TTMetalOpsTypes.cpp -o CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.s

obj.MLIRTTMetalDialect: lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalDialect.cpp.o
obj.MLIRTTMetalDialect: lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOps.cpp.o
obj.MLIRTTMetalDialect: lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/TTMetalOpsTypes.cpp.o
obj.MLIRTTMetalDialect: lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/build.make
.PHONY : obj.MLIRTTMetalDialect

# Rule to build all files generated by this target.
lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/build: obj.MLIRTTMetalDialect
.PHONY : lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/build

lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/clean:
	cd /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR && $(CMAKE_COMMAND) -P CMakeFiles/obj.MLIRTTMetalDialect.dir/cmake_clean.cmake
.PHONY : lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/clean

lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR /home/vwells/sources/tt-mlir/lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : lib/Dialect/TTMetal/IR/CMakeFiles/obj.MLIRTTMetalDialect.dir/depend
