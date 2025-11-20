#!/bin/bash

set -euo pipefail

# Function to display usage
usage() {
    echo "Usage: $0 START_JOB_ID END_JOB_ID"
    echo "Calculate mean success rate from OAR job stderr files"
    echo
    echo "Examples:"
    echo "  $0 152188 152197"
    echo "  $0 152100 152110"
}

# Validate arguments
if [[ $# -ne 2 ]]; then
    echo "Error: Please provide start and end job IDs"
    usage
    exit 1
fi

START_ID="$1"
END_ID="$2"

# Validate numeric arguments
if ! [[ "$START_ID" =~ ^[0-9]+$ ]] || ! [[ "$END_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: Job IDs must be numeric"
    exit 1
fi

if [[ "$START_ID" -gt "$END_ID" ]]; then
    echo "Error: Start ID must be less than or equal to End ID"
    exit 1
fi

# Calculate mean success rate using for loop
{
for ((id=START_ID; id<=END_ID; id++)); do
    if [[ -f "OAR.${id}.stderr" ]]; then
        cat "OAR.${id}.stderr"
    fi
done
} | grep "Successful rate" | \
awk -F': ' '{print $2}' | \
awk '{sum += $1; count++} END {if (count>0) printf "Mean success rate: %.3f\n", sum/count; else print "No success rate data found"}'