#!/bin/bash

TASKS=(
    'NeedlePick-v1'
    'NeedlePick-v2' 
    'GauzeRetrieve-v1'
    'GauzeRetrieve-v2'
)

for task in "${TASKS[@]}"; do
    oarsub -p "host in ('byta6000i0','byt4090i0','byt4090i1')" \
           -l host=1/gpuset=1/cpu=1,walltime=168:00:00 \
           -n "${task}" \
           "bash run_eval.sh $task"
    sleep 1
done
