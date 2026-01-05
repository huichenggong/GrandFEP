#!/bin/bash

# Exit if no log files given:
if [ $# -lt 1 ]; then
  echo "Usage: $0 log_file1 [log_file2 ...]"
  exit 1
fi

# 1) Determine the max replica index from the *first* log file
first_log="$1"
max_index=$(
  grep -m1 "Repl ex : " "$first_log" \
    | sed -E 's/^.*Repl ex[^0-9]*//' \
    | grep -oE '[0-9]+' \
    | sort -n \
    | tail -1
)
nrep=$((max_index + 1))

echo "Log files: $@"
echo "Max replica index in $first_log is $max_index"


# Print a header showing pair indices
printf " %3d  " 0
for rep in $(seq 1 $max_index)
do
    # Print the replica index
    printf "x %3d  " $rep
done
echo ""

total_exchange=0
min_exchange=999999
max_pair=$((max_index - 1))
for l in $(seq 0 $max_pair)
do
    # printf "%2d x%2d " $l $((l+1))
    count=$(grep " $l x " $@ \
            | cat -n \
            | tail -n 1 \
            | awk '{print $1}')  # Only take the line number

    if [ -z "$count" ]; then
        count=0
    fi

    total_exchange=$((total_exchange + count))
    min_exchange=$((count < min_exchange ? count : min_exchange))
    # echo "$count"
    printf "%7d" $count
done
echo ""
echo "Minimum/Total exchange: $min_exchange/$total_exchange"
