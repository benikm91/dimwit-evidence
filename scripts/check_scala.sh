#!/usr/bin/env bash
# Compiles every DimWit source in cases/.
#   Fixed.scala  -> MUST compile        (the corrected program is well typed)
#   Buggy.scala  -> MUST NOT compile    (that is the whole point of the experiment)
# A Buggy.scala that compiles anyway is not a script failure: it is a genuine
# result and is reported as MISSED, meaning DimWit's types did not rule the
# defect out. Run with -v to see the compiler output for each file.
set -uo pipefail
cd "$(dirname "$0")/.."

VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

mkdir -p report
LOG=report/scala.log
: > "$LOG"

pass=0; fail=0; missed=0
printf '%-34s %-14s %s\n' "CASE" "FILE" "OUTCOME"
printf '%s\n' "----------------------------------------------------------------------"

for f in cases/*/Fixed.scala cases/*/Buggy.scala; do
  [[ -e "$f" ]] || continue
  case_name=$(basename "$(dirname "$f")")
  kind=$(basename "$f" .scala)
  {
    echo "=================================================================="
    echo "### $f"
  } >> "$LOG"
  out=$(scala-cli compile "$f" 2>&1)
  rc=$?
  echo "$out" >> "$LOG"
  [[ $VERBOSE -eq 1 ]] && { echo "--- $f ---"; echo "$out"; }

  if [[ "$kind" == "Fixed" ]]; then
    if [[ $rc -eq 0 ]]; then
      printf '%-34s %-14s \033[32mcompiles\033[0m\n' "$case_name" "Fixed.scala"; pass=$((pass+1))
    else
      printf '%-34s %-14s \033[31mBROKEN (should compile)\033[0m\n' "$case_name" "Fixed.scala"; fail=$((fail+1))
    fi
  else
    if [[ $rc -ne 0 ]]; then
      printf '%-34s %-14s \033[32mrejected at compile time\033[0m\n' "$case_name" "Buggy.scala"; pass=$((pass+1))
    else
      printf '%-34s %-14s \033[33mMISSED (compiles)\033[0m\n' "$case_name" "Buggy.scala"; missed=$((missed+1))
    fi
  fi
done

printf '%s\n' "----------------------------------------------------------------------"
echo "as expected: $pass   dimwit missed: $missed   harness broken: $fail"
echo "full compiler output: $LOG"
[[ $fail -eq 0 ]]
