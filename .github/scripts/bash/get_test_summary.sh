#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

runid=$1
rm -rf test_reports
mkdir test_reports
gh run download $runid --repo tenstorrent/tt-mlir --pattern "test-reports-*" --dir test_reports || echo "No reports found"

step_number=""
summaries=$(find test_reports -name "summary_*.md" -type f)
for summary_file in $summaries; do
    filename=$(basename "$summary_file")
    if [[ $filename =~ summary_([^_]+)_([^_]+)_([^_]+)_([^_]+)\.md ]]; then
        machine="${BASH_REMATCH[1]}"
        image="${BASH_REMATCH[2]}"
        jobno="${BASH_REMATCH[3]}"
        jobid="${BASH_REMATCH[4]}"
        echo "Processing: machine=$machine, image=$image, jobno=$jobno, jobid=$jobid"

        if [ -z "$step_number" ]; then
            step_number=$(gh run view --job $jobid -v | sed -n "/ (ID $jobid)/,/^ANNOTATIONS/p" | grep -n "Run Tests" | cut -d: -f1)
            step_number=$((step_number - 1))
            echo "Found step number: $step_number"
        fi

        gh run view --log --job $jobid | sed -n -E '/[0-9]{7}Z ##\[group\]Run # Run Tests/,/[0-9]{7}Z ##\[group\]/p' >log.txt
        test_lines=($(grep -E -n "Running test [0-9]+\-" log.txt | cut -d: -f1))
        rm log.txt

        echo "## Tests for $machine, $image" >>$GITHUB_STEP_SUMMARY
        while IFS= read -r line; do
            if [[ "$line" == *"SUCCESS"* ]]; then
                test_prefix="+"
            else
                test_prefix="-"
            fi
            if [[ $line =~ ^(Test\ [0-9]+):\ (.*)$ ]]; then
                test_name="${BASH_REMATCH[1]}"
                test_result="${BASH_REMATCH[2]}"
            else
                echo "$test_prefix $line"
                continue
            fi
            if [ ${#test_lines[@]} -gt 0 ]; then
            test_line=${test_lines[0]}
            test_lines=("${test_lines[@]:1}")
            echo "$test_prefix [$test_name](https://github.com/tenstorrent/tt-mlir/actions/runs/$runid/job/$jobid?pr=5249#step:$step_number:$test_line) $test_result" >>$GITHUB_STEP_SUMMARY
            fi
        done < "$summary_file"
    fi
    echo "" >>$GITHUB_STEP_SUMMARY
done
