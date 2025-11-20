#!/bin/bash

# Simple batch submission script

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# Task list
TASKS=(
    'NeedlePick-v1'
    'NeedlePick-v2' 
    'GauzeRetrieve-v1'
    'GauzeRetrieve-v2'
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_passive.sh"

info "Submitting training jobs for ${#TASKS[@]} tasks..."

for task in "${TASKS[@]}"; do
    info "Submitting job for: $task"
    
    oarsub -p "host in ('byta6000i0','byt4090i0','byt4090i1')" \
           -l host=1/gpuset=1/cpu=1,walltime=168:00:00 \
           -n "${task}" \
           "bash $TRAIN_SCRIPT $task"
    
    success "Job submitted for: $task"
    echo
    sleep 1  # Small delay between submissions
done

success "All jobs submitted successfully!"
info "Use 'oarstat -u' to check job status"