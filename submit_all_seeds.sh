#!/bin/bash

# Batch submission script for multiple seeds
set -euo pipefail

# Configuration
SCRIPT_PATH="/bd_targaryen/users/kleelakunwet/DEX/eval_passive.sh"
N_EPISODES=20
START_SEED=1
END_SEED=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to display usage
usage() {
    cat << EOF
Usage: $0 TASK

Submit evaluation jobs for multiple seeds.

Arguments:
    TASK                    Task name to evaluate (required)

Examples:
    $0 GauzeRetrieve-v1
    $0 NeedlePass-v1
    $0 PegTransfer-v1
EOF
}

# Validate required task argument
if [[ $# -eq 0 ]]; then
    error "Task argument is required"
    usage
    exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

TASK="$1"

main() {
    info "Starting batch submission for seeds $START_SEED to $END_SEED"
    info "Task: $TASK"
    info "Episodes: $N_EPISODES"
    info "Script: $SCRIPT_PATH"
    echo "----------------------------------------"
    
    # Validate script exists
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        error "Evaluation script not found: $SCRIPT_PATH"
        exit 1
    fi
    
    # Make script executable
    chmod +x "$SCRIPT_PATH"
    
    # Submit jobs for each seed
    for seed in $(seq $START_SEED $END_SEED); do
        info "Submitting job with seed $seed..."
        
        # Submit with oarsub
        oarsub -p "host in ('byta6000i0','byt4090i0','byt4090i1')" \
               -l host=1/gpuset=1/cpu=1,walltime=168:00:00 \
               "bash $SCRIPT_PATH $TASK -n $N_EPISODES -s $seed -r"
        
        if [[ $? -eq 0 ]]; then
            success "Successfully submitted seed $seed"
        else
            error "Failed to submit seed $seed"
        fi
        
        # Small delay to avoid overwhelming the scheduler
        sleep 1
    done
    
    success "Batch submission completed!"
    info "Use 'oarstat -u' to check job status"
}

# Run main function
main "$@"