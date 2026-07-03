#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

echo "Run Optimizer Models Perf Tests"

# Enable perf feature for lit if running in perf mode
LIT_PARAMS=""
if [ "$1" = "perf" ]; then
    LIT_PARAMS="-D TTMLIR_ENABLE_OPTIMIZER_MODELS_PERF_TESTS=1"
fi

### Host name is random, watcher & asserts not helpful.
# for i in {1..3}; do
#     llvm-lit -v $LIT_PARAMS --xunit-xml-output $TEST_REPORT_PATH $BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/optimizer
#     ls -lh $BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/optimizer/generated/watcher
#     # Or '--order=lexical'
#     rm -f $BUILD_DIR/test/.lit_test_times.txt
# done

lscpu
nproc
free -m
free -g
free -h
vmstat -s

MEM_SAMPLE_INTERVAL_SECONDS="${MEM_SAMPLE_INTERVAL_SECONDS:-0.1}"
CLK_TCK="$(getconf CLK_TCK)"

current_start_ticks() {
    awk -v hz="$CLK_TCK" '{ printf "%d\n", $1 * hz }' /proc/uptime
}

process_start_ticks() {
    local pid="$1"
    local proc_stat
    local rest

    { IFS= read -r proc_stat <"/proc/$pid/stat"; } 2>/dev/null || return 1
    rest="${proc_stat##*) }"
    awk '{ print $20 }' <<<"$rest"
}

monitor_lit_process_memory() {
    local lit_pid="$1"
    local samples_file="$2"
    local min_start_ticks="$3"
    local proc_dir
    local pid
    local name
    local exe
    local uid
    local start_ticks
    local cmdline
    local vmpeak_kb
    local vmhwm_kb

    while kill -0 "$lit_pid" 2>/dev/null; do
        for proc_dir in /proc/[0-9]*; do
            pid="${proc_dir##*/}"
            [[ -r "/proc/$pid/status" ]] || continue

            read -r name uid vmpeak_kb vmhwm_kb < <(
                awk '
                    /^Name:/ { name = $2 }
                    /^Uid:/ { uid = $2 }
                    /^VmPeak:/ { vmpeak = $2 }
                    /^VmHWM:/ { vmhwm = $2 }
                    END { print name, uid, vmpeak + 0, vmhwm + 0 }
                ' "/proc/$pid/status" 2>/dev/null || echo " 0 0 0"
            )

            [[ "$uid" == "$EUID" ]] || continue
            start_ticks="$(process_start_ticks "$pid" 2>/dev/null || true)"
            [[ -n "$start_ticks" ]] || continue
            ((start_ticks >= min_start_ticks)) || continue

            exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
            { cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline")"; } 2>/dev/null || cmdline=""
            [[ "$name" == *ttmlir-opt* || "$name" == *python* || \
                "$exe" == *ttmlir-opt* || "$exe" == *python* || \
                "$cmdline" == *ttmlir-opt* || "$cmdline" == *python* ]] || continue

            cmdline="${cmdline//$'\t'/ }"
            cmdline="${cmdline//$'\n'/ }"
            cmdline="${cmdline//$'\r'/ }"
            exe="${exe//$'\t'/ }"
            exe="${exe//$'\n'/ }"
            exe="${exe//$'\r'/ }"
            [[ -n "$cmdline" ]] || cmdline="[$name]"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$pid" "$name" "$exe" \
                "$vmpeak_kb" "$vmhwm_kb" "$start_ticks" "$cmdline" >>"$samples_file"
        done

        sleep "$MEM_SAMPLE_INTERVAL_SECONDS"
    done
}

print_lit_process_memory_high_water_marks() {
    local iteration="$1"
    local samples_file="$2"

    echo
    echo "Iteration $iteration ttmlir-opt/python process memory high water marks:"
    if [[ ! -s "$samples_file" ]]; then
        echo "  No matching processes observed."
        return
    fi

    awk -F '\t' '
        function format_kb(kb) {
            return kb ? kb " kB" : "unknown"
        }
        {
            key = $1 "\t" $6
            pid[key] = $1
            if ($2 != "") {
                name[key] = $2
            }
            if ($3 != "") {
                exe[key] = $3
            }
            if (cmdline[key] == "" || length($7) > length(cmdline[key])) {
                cmdline[key] = $7
            }
            if ($4 + 0 > vmpeak[key]) {
                vmpeak[key] = $4 + 0
            }
            if ($5 + 0 > vmhwm[key]) {
                vmhwm[key] = $5 + 0
            }
        }
        END {
            printf "  %-8s %-16s %-18s %-18s %-32s %s\n", "PID", "Name", "VmPeak", "VmHWM", "Executable", "Command"
            for (key in pid) {
                printf "  %-8s %-16s %-18s %-18s %-32s %s\n", pid[key], name[key], format_kb(vmpeak[key]), format_kb(vmhwm[key]), exe[key], cmdline[key]
            }
        }
    ' "$samples_file"
}

run_lit_once() {
    local iteration="$1"
    local samples_file
    local lit_pid
    local monitor_pid
    local lit_rc
    local min_start_ticks

    samples_file="$(mktemp)"
    min_start_ticks="$(current_start_ticks)"
    llvm-lit -v $LIT_PARAMS --xunit-xml-output $TEST_REPORT_PATH $BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/optimizer &
    lit_pid="$!"

    monitor_lit_process_memory "$lit_pid" "$samples_file" "$min_start_ticks" &
    monitor_pid="$!"

    if wait "$lit_pid"; then
        lit_rc=0
    else
        lit_rc="$?"
    fi

    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true

    print_lit_process_memory_high_water_marks "$iteration" "$samples_file"
    rm -f "$samples_file"

    return "$lit_rc"
}

iteration=0
for i in {1..3}; do
    ((iteration += 1))
    run_lit_once "$iteration"
    rm -f $BUILD_DIR/test/.lit_test_times.txt
done
