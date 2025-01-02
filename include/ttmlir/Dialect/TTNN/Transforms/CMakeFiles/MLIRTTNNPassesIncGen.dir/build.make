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

# Utility rule file for MLIRTTNNPassesIncGen.

# Include any custom commands dependencies for this target.
include include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen.dir/compiler_depend.make

# Include the progress variables for this target.
include include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen.dir/progress.make

include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen: include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc
include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen: include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc

include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/bin/mlir-tblgen
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: include/ttmlir/Dialect/TTNN/Transforms/Passes.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/CodeGen/SDNodeProperties.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/CodeGen/ValueTypes.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Frontend/Directive/DirectiveBase.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Frontend/OpenACC/ACC.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Frontend/OpenMP/OMP.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/Attributes.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/Intrinsics.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsAArch64.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsAMDGPU.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsARM.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsBPF.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsDirectX.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsHexagon.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsHexagonDep.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsLoongArch.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsMips.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsNVVM.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsPowerPC.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsRISCV.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsRISCVXCV.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsRISCVXTHead.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsRISCVXsf.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsSPIRV.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsSystemZ.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsVE.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsVEVL.gen.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsWebAssembly.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsX86.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/IR/IntrinsicsXCore.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Option/OptParser.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/TableGen/Automaton.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/TableGen/SearchableTable.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/GenericOpcodes.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/GlobalISel/Combine.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/GlobalISel/RegisterBank.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/GlobalISel/SelectionDAGCompat.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/GlobalISel/Target.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/Target.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetCallingConv.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetInstrPredicate.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetItinerary.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetMacroFusion.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetPfmCounters.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetSchedule.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: /localdev/vwells/ttmlir-toolchain-x86/include/llvm/Target/TargetSelectionDAG.td
include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc: include/ttmlir/Dialect/TTNN/Transforms/Passes.td
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/vwells/sources/tt-mlir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building Passes.h.inc..."
	cd /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTNN/Transforms && /localdev/vwells/ttmlir-toolchain-x86/bin/mlir-tblgen --gen-pass-decls -I /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTNN/Transforms -I/localdev/vwells/ttmlir-toolchain-x86/include -I/localdev/vwells/ttmlir-toolchain-x86/include -I/home/vwells/sources/tt-mlir/include -I/home/vwells/sources/tt-mlir/include /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTNN/Transforms/Passes.td --write-if-changed -o /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc

include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen.dir/codegen:
.PHONY : include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen.dir/codegen

MLIRTTNNPassesIncGen: include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen
MLIRTTNNPassesIncGen: include/ttmlir/Dialect/TTNN/Transforms/Passes.h.inc
MLIRTTNNPassesIncGen: include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen.dir/build.make
.PHONY : MLIRTTNNPassesIncGen

# Rule to build all files generated by this target.
include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen.dir/build: MLIRTTNNPassesIncGen
.PHONY : include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen.dir/build

include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen.dir/clean:
	cd /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTNN/Transforms && $(CMAKE_COMMAND) -P CMakeFiles/MLIRTTNNPassesIncGen.dir/cmake_clean.cmake
.PHONY : include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen.dir/clean

include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen.dir/depend:
	cd /home/vwells/sources/tt-mlir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTNN/Transforms /home/vwells/sources/tt-mlir /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTNN/Transforms /home/vwells/sources/tt-mlir/include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : include/ttmlir/Dialect/TTNN/Transforms/CMakeFiles/MLIRTTNNPassesIncGen.dir/depend
