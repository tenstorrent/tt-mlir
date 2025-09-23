# TT-MLIR Environment Variable-Controlled IR Dumping

This document describes the new environment variable-controlled IR dumping feature that has been integrated into TT-MLIR's MLIRModuleLogger infrastructure.

## Environment Variables

The following environment variables control IR dumping behavior:

### `TTMLIR_DUMP_IR`
- **Values**: `1` or `true` (enable), `0` or `false` (disable)
- **Default**: Disabled
- **Description**: Master switch to enable IR dumping

### `TTMLIR_DUMP_IR_DIR`
- **Values**: Path to directory (e.g., `/path/to/dump/directory`)
- **Default**: `./ir_dumps`
- **Description**: Directory where IR dumps will be saved

### `TTMLIR_DUMP_IR_PASSES`
- **Values**: Comma-separated list of pass names (e.g., `pass1,pass2,pass3`)
- **Default**: All passes (if empty)
- **Description**: Only dump IR for specified passes

### `TTMLIR_DUMP_IR_DIALECTS`
- **Values**: `1` or `true` (enable), `0` or `false` (disable)
- **Default**: Disabled
- **Description**: Enable dumping of dialect creation information

### `TTMLIR_DUMP_IR_DEBUG_INFO`
- **Values**: `1` or `true` (enable), `0` or `false` (disable)
- **Default**: Enabled
- **Description**: Preserve debug info in IR dumps

## Usage Examples

### Basic Usage
```bash
# Enable IR dumping with default settings
export TTMLIR_DUMP_IR=1
ttmlir-opt input.mlir -pass1 -pass2
```

### Custom Directory
```bash
# Dump IR to a specific directory
export TTMLIR_DUMP_IR=1
export TTMLIR_DUMP_IR_DIR=/tmp/my_ir_dumps
ttmlir-opt input.mlir -pass1 -pass2
```

### Specific Passes Only
```bash
# Only dump IR for specific passes
export TTMLIR_DUMP_IR=1
export TTMLIR_DUMP_IR_PASSES="ttir-to-ttnn,ttnn-layout"
ttmlir-opt input.mlir -ttir-to-ttnn-backend-pipeline
```

### With Dialect Creation Info
```bash
# Enable dialect creation dumping
export TTMLIR_DUMP_IR=1
export TTMLIR_DUMP_IR_DIALECTS=1
ttmlir-opt input.mlir -pass1 -pass2
```

### Complete Setup
```bash
# Full configuration
export TTMLIR_DUMP_IR=1
export TTMLIR_DUMP_IR_DIR=/home/user/ir_analysis
export TTMLIR_DUMP_IR_PASSES="ttir-layout,ttnn-optimizer"
export TTMLIR_DUMP_IR_DIALECTS=1
export TTMLIR_DUMP_IR_DEBUG_INFO=1
ttmlir-opt input.mlir -ttir-to-ttnn-backend-pipeline
```

## Output Files

The IR dumping system creates the following files in the specified directory:

### Pass IR Dumps
- **Filename format**: `{pass_name}.mlir`
- **Content**: Complete MLIR module after the specified pass
- **Special files**:
  - `PRE-PIPELINE.mlir`: IR before any passes run

### Dialect Creation Logs
- **Filename format**: `dialect_{dialect_name}_created.log`
- **Content**: Information about when dialects are created and loaded dialects list

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

1. **No files created**: Ensure `TTMLIR_DUMP_IR=1` is set
2. **Permission errors**: Check write permissions for dump directory
3. **Large file sizes**: Use `TTMLIR_DUMP_IR_PASSES` to limit output
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
