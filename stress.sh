#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


echo "Running a stress tests on all of monarch's buck based unit tests."

paste-it() {
  local fname="/tmp/paste-it.tmp"
  echo "$" "$@" | tee $fname
  "$@" 2>&1 | tee -a $fname
  local ret=${PIPESTATUS[0]}  # Capture the exit code of the command, not tee

  echo "------------------------"
  pastry < $fname
  head $fname | grep https
  echo "------------------------"

  return "$ret"
}

# shellcheck disable=SC2046
paste-it buck test @//mode/opt $(buck uquery 'testsof(//monarch/...)') \
  -- --stress-runs 20 --return-zero-on-skips # lint-ignore
EXIT_CODE=$?
echo "Because automated stress test aren't run at diff time, please copy the past the above into your test plan."
exit "$EXIT_CODE"
