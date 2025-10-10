#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

runid=$1
extract_lines=$2
rm -rf test_reports
mkdir test_reports
echo "Downloading test reports for run ID: $runid"
gh run download $runid --repo tenstorrent/tt-mlir --pattern "test-reports-*" --dir test_reports || echo "No reports found"

echo "Parsing test summaries..."
step_number=""
rm -f _summary.md
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

        test_lines=()
        if [ -n "$extract_lines" ]; then
            echo "Extracting test lines from logs for job ID: $jobid"
            curl -L -H "Authorization: token $GH_TOKEN" -H "Accept: */*" "https://api.github.com/repos/tenstorrent/tt-mlir/actions/jobs/$jobid/logs" | \
                sed -n -E '/[0-9]{7}Z ##\[group\]Run # Run Tests/,/[0-9]{7}Z ##\[group\]/p' >log.txt
            test_lines=($(grep -E -n "Running test [0-9]+\-" log.txt | cut -d: -f1))
            rm log.txt
        fi

        echo "### Tests for $machine, $image" >>_summary.md
        while IFS= read -r line; do
            if [[ "$line" == *"SUCCESS"* ]]; then
                test_prefix="- ![#c5f015](https://placehold.co/15x15/c5f015/c5f015.png)"
            else
                test_prefix="- ![#f03c15](https://placehold.co/15x15/f03c15/f03c15.png)"
            fi
            if [ ${#test_lines[@]} -gt 0 ]; then
                test_line=${test_lines[0]}
                test_lines=("${test_lines[@]:1}")
            else
                test_line=1
            fi
                echo "$test_prefix [$line](https://github.com/tenstorrent/tt-mlir/actions/runs/$runid/job/$jobid#step:$step_number:$test_line)" >>_summary.md
        done < "$summary_file"
    fi
    echo "" >>_summary.md
done

if [ ! -s _summary.md ]; then
    echo "No test summaries found."
    echo "No test summaries available." >_summary.md
else
    echo "Summary of all tests:"
    cat _summary.md
fi
