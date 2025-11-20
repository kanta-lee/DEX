#!/bin/bash

# DEX Evaluation Script
# Professional version with better parameter handling and flexibility

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
HOME_DIR="/bd_targaryen/users/kleelakunwet"
CONDA_ENV="dex"
WORKDIR="DEX"
DEFAULT_SEED=42
DEFAULT_N_EPISODES=10
DEFAULT_RENDER_THREE_VIEWS="false"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] TASK

Evaluate DEX model with specified parameters.

Required:
    TASK                    Task name to evaluate

Options:
    -n, --n-episodes NUM    Number of evaluation episodes (default: $DEFAULT_N_EPISODES)
    -s, --seed SEED         Random seed (default: $DEFAULT_SEED)
    -r, --render-views      Render three views (default: $DEFAULT_RENDER_THREE_VIEWS)
    -h, --help              Show this help message

Examples:
    $0 reach_target
    $0 -n 20 -s 123 --render-views pick_object
    $0 push_button
EOF
}

# Initialize variables with defaults
TASK=""
N_EPISODES=$DEFAULT_N_EPISODES
SEED=$DEFAULT_SEED
RENDER_THREE_VIEWS=$DEFAULT_RENDER_THREE_VIEWS

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--n-episodes)
            N_EPISODES="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -r|--render-views)
            RENDER_THREE_VIEWS="true"
            shift
            ;;
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

# Validate required parameters
if [[ -z "$TASK" ]]; then
    echo "Error: TASK argument is required"
    usage
    exit 1
fi

# Validate numeric arguments
if ! [[ "$N_EPISODES" =~ ^[0-9]+$ ]]; then
    echo "Error: n-episodes must be a positive integer"
    exit 1
fi

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "Error: seed must be a positive integer"
    exit 1
fi

# Main execution
main() {
    echo "Starting DEX evaluation..."
    echo "Task: $TASK"
    echo "Episodes: $N_EPISODES"
    echo "Seed: $SEED"
    echo "Render views: $RENDER_THREE_VIEWS"
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
    
    # Execute the evaluation
    echo "Running evaluation..."
    python eval.py \
        task="$TASK" \
        n_eval_episodes="$N_EPISODES" \
        seed="$SEED" \
        render_three_views="$RENDER_THREE_VIEWS" \
        ckpt_dir=./exp_local/$TASK/DEX/d100/s1/model
    
    echo "Evaluation completed successfully!"
}

# Run main function
main "$@"