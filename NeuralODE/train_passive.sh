#!/bin/bash

# DEX Evaluation Script
# Professional version with better parameter handling and flexibility

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
HOME_DIR="/bd_targaryen/users/kleelakunwet"
CONDA_ENV="dex"
WORKDIR="DEX/NeuralODE"
TRAIN_SCRIPT="train.py"

# Valid tasks for validation
VALID_TASKS=(
    'NeedlePick-v1'
    'NeedlePick-v2' 
    'GauzeRetrieve-v1'
    'GauzeRetrieve-v2'
)

# Print usage information
usage() {
    echo "Usage: $0 <task>"
    echo ""
    echo "Available tasks:"
    for task in "${VALID_TASKS[@]}"; do
        echo "  - $task"
    done
    echo ""
    echo "Example: $0 NeedlePick-v1"
}

# Initialize variables with defaults
TASK=""

# Script entry point
if [[ $# -ne 1 ]]; then
    echo "Invalid number of arguments"
    usage
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option $1"
            usage
            exit 1
            ;;
        *)
            TASK="$1"
            shift
            ;;
    esac
done

# Main execution
main() {
    echo "Starting training..."
    echo "Task: $TASK"
    echo "Conda environment: $CONDA_ENV"
    echo "Working directory: $WORKDIR"
    echo "----------------------------------------"
    
    # Set home directory and navigate
    export HOME="$HOME_DIR"
    cd "$HOME_DIR" || {
        echo "Error: Cannot navigate to home directory $HOME_DIR"
        exit 1
    }
    
    # Activate conda environment
    if [[ -f "miniconda3/bin/activate" ]]; then
        source "miniconda3/bin/activate"
    else
        echo "Error: Conda activation script not found"
        exit 1
    fi
    
    conda activate "$CONDA_ENV" || {
        echo "Error: Failed to activate conda environment '$CONDA_ENV'"
        exit 1
    }

    # Navigate to working directory
    cd "$WORKDIR" || {
        echo "Error: Cannot navigate to working directory $WORKDIR"
        exit 1
    }
    
    # Execute training
    echo "Executing: python $TRAIN_SCRIPT --task $TASK"
    echo "Training started at: $(date)"
    
    python "$TRAIN_SCRIPT" --task $TASK
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "Training completed successfully at: $(date)"
    else
        echo "Training failed with exit code: $exit_code"
        exit $exit_code
    fi
}

main "$@"