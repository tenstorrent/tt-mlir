#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Ultra-fast SEQUENTIAL script for massive parameter sweeps on single device
# Uses fast_profiler_sum.py for efficient trace replay kernel time extraction
# Supports multiple trace replay iterations with averaging

source env/activate
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_DIR=/localdev/brapanan/tt-mlir/third_party/tt-metal/src/tt-metal/generated/profiler/
export TT_METAL_HOME=/localdev/brapanan/tt-mlir/third_party/tt-metal/src/tt-metal
export PYTHONPATH="${PYTHONPATH}:${TT_METAL_HOME}/ttnn:${TT_METAL_HOME}:${TT_METAL_HOME}/tools"

# Number of trace replay iterations (default: 10)
NUM_ITERATIONS=${NUM_ITERATIONS:-10}

# Run mode: "both" (default), "jit", or "ttnn"
RUN_MODE=${RUN_MODE:-both}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${TT_METAL_PROFILER_DIR}/sweep_${TIMESTAMP}.csv"

echo "Configuration:"
echo "  Trace Replay Iterations: ${NUM_ITERATIONS}"
echo "  Run Mode: ${RUN_MODE}"
echo "  Output File: ${SUMMARY_FILE}"
echo ""

echo "DEPTH,BATCH_SIZE,LAYER_SIZE,VERSION,TOTAL_TIME_S,AVG_LATENCY_US,THROUGHPUT_SAMPLES_PER_SEC,DEVICE_KERNEL_NS,PERCENT_OF_TTNN_PERF,STATUS" > "$SUMMARY_FILE"

echo "Generating configuration list..."

# Configure your parameter ranges here
DEPTHS=($(seq 1 16))
# Powers of 2: 32, 64, 128, 256, 512, 1024, 2048 for batch sizes
BATCH_SIZES=(1 32 64 128 256 512 1024 2048)
# Powers of 2: 512, 1024, 2048, 4096 for layer sizes (starting from 512)
LAYER_SIZES=(512 1024 2048 4096)

# Set versions based on run mode
case "$RUN_MODE" in
    "jit")
        VERSIONS=("jit")
        ;;
    "ttnn")
        VERSIONS=("ttnn")
        ;;
    "both")
        VERSIONS=("ttnn" "jit")
        ;;
    *)
        echo "Error: Invalid RUN_MODE '$RUN_MODE'. Must be 'jit', 'ttnn', or 'both'."
        exit 1
        ;;
esac

# Uncomment below for testing with smaller ranges
#DEPTHS=(1)
#BATCH_SIZES=(32)
#LAYER_SIZES=(512)

# Count total
TOTAL_CONFIGS=0
for DEPTH in "${DEPTHS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for LAYER_SIZE in "${LAYER_SIZES[@]}"; do
            for VERSION in "${VERSIONS[@]}"; do
                ((TOTAL_CONFIGS++))
            done
        done
    done
done

echo "Total configurations: ${TOTAL_CONFIGS}"
echo ""

CURRENT=0

# Store results for comparison
declare -A ttnn_kernel_times
declare -A jit_kernel_times

# Buffer for batch writes
WRITE_BUFFER=""
BUFFER_SIZE=10
BUFFER_COUNT=0

for DEPTH in "${DEPTHS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for LAYER_SIZE in "${LAYER_SIZES[@]}"; do
            for VERSION in "${VERSIONS[@]}"; do
                ((CURRENT++))

                echo "[${CURRENT}/${TOTAL_CONFIGS}] D=${DEPTH} B=${BATCH_SIZE} L=${LAYER_SIZE} V=${VERSION}"

                export DEPTH BATCH_SIZE LAYER_SIZE
                OUTPUT_NAME="run_${DEPTH}_${BATCH_SIZE}_${LAYER_SIZE}_${VERSION}"
                CONFIG_KEY="${DEPTH}_${BATCH_SIZE}_${LAYER_SIZE}"

                # Select test file based on version
                if [ "$VERSION" = "jit" ]; then
                    TEST_FILE="test/ttnn-jit/test_jit_resnet_block_sharded.py"
                else
                    TEST_FILE="test/ttnn-jit/test_ttnn_resnet_block_sharded.py"
                fi

                # Run test, capture only metrics we need
                if timeout 45s python -m tracy -r -m pytest "$TEST_FILE" -x -svv 2>&1 | \
                   grep -E "(Total Time|Avg Latency|Throughput):" > "${TT_METAL_PROFILER_DIR}/${OUTPUT_NAME}_m.txt"; then

                    TOTAL_TIME=$(grep "Total Time:" "${TT_METAL_PROFILER_DIR}/${OUTPUT_NAME}_m.txt" | awk '{print $3}')
                    AVG_LATENCY=$(grep "Avg Latency:" "${TT_METAL_PROFILER_DIR}/${OUTPUT_NAME}_m.txt" | awk '{print $3}')
                    THROUGHPUT=$(grep "Throughput:" "${TT_METAL_PROFILER_DIR}/${OUTPUT_NAME}_m.txt" | awk '{print $2}')                    # Find the latest profiler CSV faster - use ls with modification time
                    LATEST_CSV=$(ls -t ${TT_METAL_PROFILER_DIR}/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)

                    if [ -n "$LATEST_CSV" ] && [ -f "$LATEST_CSV" ]; then
                        # Use fast profiler sum to get average trace replay kernel time across iterations
                        DEVICE_KERNEL_NS=$(python3 fast_profiler_sum.py "$LATEST_CSV" --iterations "$NUM_ITERATIONS")

                        # Store kernel time for comparison
                        if [ "$VERSION" = "ttnn" ]; then
                            ttnn_kernel_times[$CONFIG_KEY]=$DEVICE_KERNEL_NS
                        else
                            jit_kernel_times[$CONFIG_KEY]=$DEVICE_KERNEL_NS
                        fi
                    else
                        DEVICE_KERNEL_NS="NO_CSV"
                    fi

                    # Calculate speedup if both versions have run
                    JIT_SPEEDUP="N/A"
                    if [ "$VERSION" = "jit" ] && [ "$RUN_MODE" = "both" ] && [ -n "${ttnn_kernel_times[$CONFIG_KEY]}" ] && [ "$DEVICE_KERNEL_NS" != "NO_CSV" ]; then
                        TTNN_TIME=${ttnn_kernel_times[$CONFIG_KEY]}
                        # Calculate percentage: (ttnn_time / jit_time) * 100
                        # If jit is faster, this will be > 100%
                        JIT_SPEEDUP=$(awk "BEGIN {printf \"%.2f\", ($TTNN_TIME / $DEVICE_KERNEL_NS) * 100}")
                    fi

                    # Buffer results for batch write
                    RESULT_LINE="${DEPTH},${BATCH_SIZE},${LAYER_SIZE},${VERSION},${TOTAL_TIME},${AVG_LATENCY},${THROUGHPUT},${DEVICE_KERNEL_NS},${JIT_SPEEDUP},SUCCESS"
                    WRITE_BUFFER="${WRITE_BUFFER}${RESULT_LINE}\n"
                    ((BUFFER_COUNT++))

                    # Flush buffer every 10 writes
                    if [ $BUFFER_COUNT -ge $BUFFER_SIZE ]; then
                        echo -e "$WRITE_BUFFER" >> "$SUMMARY_FILE"
                        WRITE_BUFFER=""
                        BUFFER_COUNT=0
                    fi

                    # Cleanup - delete the profiler CSV after extracting data
                    rm -f "${TT_METAL_PROFILER_DIR}/${OUTPUT_NAME}_m.txt"
                    [ -n "$LATEST_CSV" ] && rm -f "$LATEST_CSV"

                else
                    # Buffer failed results too
                    RESULT_LINE="${DEPTH},${BATCH_SIZE},${LAYER_SIZE},${VERSION},FAILED,FAILED,FAILED,FAILED,N/A,FAILED"
                    WRITE_BUFFER="${WRITE_BUFFER}${RESULT_LINE}\n"
                    ((BUFFER_COUNT++))

                    if [ $BUFFER_COUNT -ge $BUFFER_SIZE ]; then
                        echo -e "$WRITE_BUFFER" >> "$SUMMARY_FILE"
                        WRITE_BUFFER=""
                        BUFFER_COUNT=0
                    fi

                    rm -f "${TT_METAL_PROFILER_DIR}/${OUTPUT_NAME}_m.txt"
                fi

                # Progress summary every 100 configs
                if [ $((CURRENT % 100)) -eq 0 ]; then
                    SUCCESS=$(tail -n +2 "$SUMMARY_FILE" | grep -c "SUCCESS" || echo "0")
                    FAIL=$(tail -n +2 "$SUMMARY_FILE" | grep -c "FAILED" || echo "0")
                    echo "  Progress: ${SUCCESS} success, ${FAIL} failed"
                fi
            done
        done
    done
done

# Flush any remaining buffered writes
if [ $BUFFER_COUNT -gt 0 ]; then
    echo -e "$WRITE_BUFFER" >> "$SUMMARY_FILE"
fi

echo ""
echo "================================================================"
echo "Sweep Complete!"
echo "================================================================"
echo "Results: ${SUMMARY_FILE}"
echo ""

SUCCESS=$(tail -n +2 "$SUMMARY_FILE" | grep -c "SUCCESS" || echo "0")
FAIL=$(tail -n +2 "$SUMMARY_FILE" | grep -c "FAILED" || echo "0")
echo "Summary: ${SUCCESS} success / ${FAIL} failed / ${TOTAL_CONFIGS} total"
echo ""
echo "Sample results:"
head -n 6 "$SUMMARY_FILE"
