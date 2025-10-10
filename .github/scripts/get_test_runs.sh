#!/bin/bash

runid=$1
extract_lines=$2
rm -rf test_reports
mkdir test_reports
gh run download $runid --repo tenstorrent/tt-mlir --pattern "test-reports-*" --dir test_reports || echo "No reports found"

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
            gh run view --log --job $jobid | sed -n -E '/[0-9]{7}Z ##\[group\]Run # Run Tests/,/[0-9]{7}Z ##\[group\]/p' >log.txt
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
            if [[ $line =~ ^(Test\ [0-9]+):\ (.*)$ ]]; then
                test_name="${BASH_REMATCH[1]}"
                test_result="${BASH_REMATCH[2]}"
            else
                echo "$test_prefix $line </span>" >>_summary.md
                continue
            fi
            if [ ${#test_lines[@]} -gt 0 ]; then
                test_line=${test_lines[0]}
                test_lines=("${test_lines[@]:1}")
                echo "$test_prefix [$test_name $test_result](https://github.com/tenstorrent/tt-mlir/actions/runs/$runid/job/$jobid#step:$step_number:$test_line)" >>_summary.md
            else
                echo "$test_prefix [$test_name $test_result](https://github.com/tenstorrent/tt-mlir/actions/runs/$runid/job/$jobid#step:$step_number:1)" >>_summary.md
            fi
        done < "$summary_file"
    fi
    echo "" >>_summary.md
done
