#!/bin/bash

# Set the folder path
folder="/srv/rl2-lab/flash8/mbronars3/workspace/diffusion_policy_robo/diffusion_policy/config/ICRA/2block_base_param_trans"

# Set the base command
base_command="python train.py --config-dir /srv/rl2-lab/flash8/mbronars3/workspace/diffusion_policy_robo/diffusion_policy/config/ICRA/2block_base_param_trans hydra.run.dir='/srv/rl2-lab/flash8/mbronars3/ICRA/results/hyperparam_scan/2block_base_params/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'"

# Function to handle Ctrl+C
function handle_ctrl_c {
    echo "Received Ctrl+C. Skipping the current file and moving to the next one..."
}

# Trap Ctrl+C and call the handle_ctrl_c function
trap handle_ctrl_c INT

# Loop through files in the folder and execute the base command with --config-name
for file in "$folder"/*; do
    if [ -f "$file" ]; then
        echo "Running $file"
        $base_command --config-name "$(basename "$file")"
    else
        echo "Skipping $file (not executable)"
    fi
done