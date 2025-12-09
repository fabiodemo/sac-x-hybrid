#!/bin/bash

agent="${1:-ddpg}"
stage="${2:-1}"
lidar="${3:-10}"
load_models="${4}"

echo "Agent: $agent - Stage: $stage - Lidar=$lidar - load=${load_models:-without loading models}"
sleep 5


echo "...Training..."
if [ -n "$load_models" ]; then
    python train.py --agent $agent --stage $stage --lidar $lidar --load $load_models
else
    python train.py --agent $agent --stage $stage --lidar $lidar
fi
sleep 5

echo "...Saving models to best_models..."
python save_to_best.py --agent  $agent --stage $stage --lidar $lidar
sleep 5

echo "...Testing..."
python test.py --agent $agent --stage $stage --lidar $lidar
sleep 5

# echo "...Plotting learning curve..."
# python learning_curve.py --agent $agent --stage $stage --lidar $lidar