#!/bin/bash
# Wrapper script to run multi_source_itm_node.py with conda environment

# Activate sg_apexnav conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sg_apexnav

# Run the Python script
exec python3 "$(dirname "$0")/multi_source_itm_node.py" "$@"
