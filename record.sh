#!/bin/bash

if [ -z "$1" ]; then
    echo "usage: bash record.sh <task> [cbf]"
    echo "Available tasks: GauzeRetrieve-v1, GauzeRetrieve-v2, NeedlePick-v1, NeedlePick-v2"
    exit 0
fi

task=$1
use_cbf=$2

# Validate task
valid_tasks=("GauzeRetrieve-v1" "GauzeRetrieve-v2" "NeedlePick-v1" "NeedlePick-v2")
if [[ ! " ${valid_tasks[@]} " =~ " ${task} " ]]; then
    echo "Error: Invalid task '$task'"
    echo "Available tasks: ${valid_tasks[*]}"
    exit 1
fi

# Loop through all combinations
for episode in {0..19}; do
    for seed in {1..3}; do
        if [ "$use_cbf" = "cbf" ]; then
            echo "Running: python video.py $task --episode $episode --seed $seed --cbf"
            python video.py $task --episode $episode --seed $seed --cbf
        else
            echo "Running: python video.py $task --episode $episode --seed $seed"
            python video.py $task --episode $episode --seed $seed
        fi
    done
done
