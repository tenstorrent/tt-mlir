# TT-MLIR Environment Variable-Controlled IR Dumping

This document describes the new environment variable-controlled IR dumping feature that has been integrated into TT-MLIR's MLIRModuleLogger infrastructure.

## Environment Variables

The following environment variables control IR dumping behavior:

### `TTMLIR_DUMP_IR`
- **Values**: `per_pass`, `per_dialect` (plus alternative names below)
- **Default**: Disabled (empty/unset)
- **Description**: Controls IR dumping behavior
  - `per_pass` (or `all`, `every_pass`, `pass_by_pass`, `detailed`): Dump IR after every pass
  - `per_dialect` (or `final`, `end_only`, `pipeline_end`, `summary`): Dump only one IR per pipeline

### `TTMLIR_DUMP_IR_DIR`
- **Values**: Path to directory (e.g., `/path/to/dump/directory`)
- **Default**: `./ir_dumps`
- **Description**: Directory where IR dumps will be saved

### `TTMLIR_DUMP_IR_ACTION`
- **Values**: `overwrite`, `append`
- **Default**: `overwrite`
- **Description**: Controls how to handle existing IR dumps
  - `overwrite` (default): Clear target directory and start pass numbering from 0
  - `append`: Continue pass numbering from the highest existing index + 1


## Usage Examples

### Per-Pass Dumping (Detailed)
```bash
# Dump IR after every pass (detailed mode)
export TTMLIR_DUMP_IR=per_pass
ttmlir-opt test/ttmlir/Conversion/ArithToStableHLO/constant_op.mlir -pass1 -pass2
# Creates: ir_dumps/constant_op/unknown/0_PRE-PIPELINE.mlir, 1_pass1.mlir, 2_pass2.mlir, etc.
```

### Per-Dialect Dumping (Summary)
```bash
# Only dump one IR per pipeline (summary mode) - final result
export TTMLIR_DUMP_IR=per_dialect
ttmlir-opt input.mlir -ttir-to-ttnn-backend-pipeline
# Creates: ir_dumps/model/1_PIPELINE_FINAL.mlir (contains final IR after all passes)
```

### Custom Directory
```bash
# Dump IR to a specific directory
export TTMLIR_DUMP_IR=per_pass
export TTMLIR_DUMP_IR_DIR=/tmp/my_ir_dumps
ttmlir-opt input.mlir -pass1 -pass2
```

### Append to Existing Dumps
```bash
# Continue pass numbering from existing dumps (useful for multi-stage compilation)
export TTMLIR_DUMP_IR=per_pass
export TTMLIR_DUMP_IR_ACTION=append
ttmlir-opt input.mlir -ttir-to-ttnn-backend-pipeline
# If previous run had files up to 5_pass.mlir, this run starts at 6_pass.mlir
```

### Overwrite Existing Dumps (Default)
```bash
# Clear existing dumps and start fresh (default behavior)
export TTMLIR_DUMP_IR=per_pass
export TTMLIR_DUMP_IR_ACTION=overwrite  # This is the default
ttmlir-opt input.mlir -ttir-to-ttnn-backend-pipeline
# Always starts from 0_PRE-PIPELINE.mlir regardless of existing files
```

### Complete Setup
```bash
# Full configuration with per-pass dumping
export TTMLIR_DUMP_IR=per_pass
export TTMLIR_DUMP_IR_DIR=/home/user/ir_analysis
ttmlir-opt input.mlir -ttir-to-ttnn-backend-pipeline
```

## Output Files

The IR dumping system creates different files depending on the dump mode:

### Per-Pass Mode (`per_pass`)
Creates detailed IR dumps after every pass in a structured subdirectory layout:

#### Directory Structure
- **Format**: `<model_name>/<total_pass_count>_<pass_name>.mlir`
- **Example**: `constant_op/3_stablehlo-to-ttir.mlir`

#### Pass IR Dumps
- **Filename format**: `<total_pass_count>_<pass_name>.mlir`
- **Content**: Complete MLIR module after the specified pass
- **Numbering**: Continuous across multiple pipeline runs when using `TTMLIR_DUMP_IR_ACTION=append`
- **Special files**:
  - `0_PRE-PIPELINE.mlir`: IR before any passes run

### Per-Dialect Mode (`per_dialect`)
Creates one IR dump per pipeline (summary mode):

#### Pipeline Final IR Dump
- **Filename format**: `1_PIPELINE_FINAL.mlir`
- **Content**: Complete MLIR module after the final pass in the pipeline
- **Behavior**: Stores IR in memory after each pass, writes to disk only once at pipeline completion
- **Performance**: Minimal I/O overhead - only serializes and writes the final IR state
- **Purpose**: Provides a single snapshot of the pipeline's final state rather than detailed per-pass evolution

## Integration Points

The IR dumping feature is automatically enabled across all TT-MLIR tooling:

1. **ttmlir-opt**: All pass execution is monitored
2. **Python bindings**: All pipeline functions have IR dumping support
3. **Frontend integration**: Ready for tt-xla, tt-torch, tt-forge-fe
4. **ttrt**: Will automatically support IR dumping when using TT-MLIR pipelines

## Technical Details

### Implementation
- Extends existing `MLIRModuleLogger` infrastructure
- Uses MLIR's `registerActionHandler` for pass monitoring
- Preserves location information and debug data
- Thread-safe file writing
- Automatic directory creation

### Performance Impact
- Minimal overhead when disabled (single environment variable check)
- When enabled, adds file I/O overhead proportional to IR size and number of passes
- No impact on compilation correctness or optimization

### File Safety
- Automatically creates dump directories
- Sanitizes pass names for safe filenames
- Handles path separators and special characters
- Overwrites existing files with same names

## Troubleshooting

### Common Issues

1. **No files created**: Ensure `TTMLIR_DUMP_IR=per_pass` or `TTMLIR_DUMP_IR=per_dialect` is set
2. **Permission errors**: Check write permissions for dump directory
3. **Large file sizes**: Use `per_dialect` mode for summary output instead of detailed per-pass dumps
4. **Path issues**: Use absolute paths for `TTMLIR_DUMP_IR_DIR`

### Debugging
```bash
# Verify environment variables are set
env | grep TTMLIR_DUMP_IR

# Test with simple input
echo "func.func @test() { return }" | ttmlir-opt
```

## Future Enhancements

Potential future improvements:
- Timestamp-based file naming
- Compression support for large IR dumps
- Integration with external analysis tools
- Real-time IR diff generation
- Web-based IR viewer integration
