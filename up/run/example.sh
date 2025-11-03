#!/bin/bash

# to make sure exit status is propagated even when using tee to capture output logs
set -o pipefail

# make sure logs directory exists
mkdir -p logs/build

# log commit being bisected
echo -e "\n\nbisecting commit: $(git log -1 --oneline)\n" >> logs/bisect_log.log
# before checking every commit, show the current window of commits left to bisect
# echo -e "Currently bisecting commit:\n$(git log -1 --oneline)\n\nWindow remaining:\n$(git bisect visualize --oneline)" > logs/bisect_status.log

# install tt-smi if not installed. uncomment if tt-smi used
which tt-smi || pip install git+https://github.com/tenstorrent/tt-smi |& tee logs/tt_smi_install.log
tt-smi -r

# test using command exit status, if it fails, mark commit as bad
COMMAND="echo 'test'"
RESULT=$?
if [ $RESULT -ne 0 ]; then
    echo -e "Test failed\n" >> logs/bisect_log.log
    exit 1
fi

# or look for error signatures
if grep -q -- "- ERROR - ERROR:" logs/ttrt_run.log; then
    exit 1
fi
if grep -q "Failed:" logs/check_ttmlir.log; then
    exit 1
fi

# if it fails earlier (e.g. build step fails) than intended test, mark commit as not testable
$COMMAND
if [ $RESULT -ne 0 ]; then
    echo -e "Build failed\n" >> logs/bisect_log.log
    exit 125
fi

# otherwise, if everything passes, mark as good commit 
echo -e "Test passed\n" >> logs/bisect_log.log
exit 0



# to test hangs, set timeout
TIMEOUT_DURATION_IN_SECONDS=900 # set 15 minute timeout
timeout -s SIGKILL $TIMEOUT_DURATION_IN_SECONDS $COMMAND
RESULT=$?
if [ $RESULT -eq 124 ] || [ $RESULT -eq 137 ]; then
    # Timeout occurred (test hang)
    tt-smi -r
    exit 1
fi

# to find how long it takes to hang, use this with descriptive logs to get timestamp per log
sudo apt-get install moreutils -y # to install ts
$COMMAND | ts -s "%H:%M:%.S"

# to test ND fails, run repeatedly
REPETITIONS=100
for i in $(seq 1 $REPETITIONS); do
  echo test $i
  $COMMAND
done

