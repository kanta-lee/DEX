#!/bin/bash

# Configuration
TASKS=("NeedlePick-v2")
# TASKS=("NeedlePick-v1" "NeedlePick-v2" "GauzeRetrieve-v1" "GauzeRetrieve-v2")
SEEDS=(1 2 3)
METHODS=("NONE" "CBF")  # use_dcbf=False for NONE, use_dcbf=True for CBF
CHECKPOINT_BASE="./exp_local"

# OAR submission parameters
OAR_QUEUE="host in ('byt4090i0','byta6000i0','byt4090i1')"
# OAR_QUEUE="host in ('daenerys')"
OAR_RESOURCES="host=1/gpuset=1/cpu=1,walltime=168:00:00"

# Create tasks directory
mkdir -p generated_tasks

echo "Generating and submitting evaluation tasks..."

# Generate and submit tasks
for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for method in "${METHODS[@]}"; do
            
            # Determine method parameters
            if [ "$method" = "CBF" ]; then
                use_dcbf="True"
                method_suffix="_cbf"
            else
                use_dcbf="False" 
                method_suffix=""
            fi
            
            # Create task script filename
            task_script="generated_tasks/task_${task}_s${seed}${method_suffix}.sh"
            
            # Generate task script
            cat > "$task_script" << EOF
#!/bin/bash
HOME=/bd_targaryen/users/kleelakunwet/
cd ~
source miniconda3/bin/activate
conda activate dex
cd DEX
python eval.py task=$task n_eval_episodes=20 seed=$seed use_dcbf=$use_dcbf render_three_views=True ckpt_dir=$CHECKPOINT_BASE/$task/DEX/d100/s1/model
EOF
            
            # Make script executable
            chmod +x "$task_script"
            
            # Submit to OAR
            echo "Submitting: $task (seed=$seed, method=$method)"
            oarsub -p "$OAR_QUEUE" -l "$OAR_RESOURCES" -n "${task}_s${seed}${method_suffix}" "bash $task_script"
            
            # Small delay to avoid overwhelming the scheduler
            sleep 1
        done
    done
done

echo "All tasks submitted!"
echo "Total tasks: $(( ${#TASKS[@]} * ${#SEEDS[@]} * ${#METHODS[@]} ))"