#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Check if the --local flag is provided
REMOTE_FLAG=""
if [[ "$1" == "--launch_local" ]]; then
  REMOTE_FLAG="--launch_local"
fi

# Run the Python script with the appropriate flag
python3 src/visualization/visualize_rgbd_lab_base.py $REMOTE_FLAG