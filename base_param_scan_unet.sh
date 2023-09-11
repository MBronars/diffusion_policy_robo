#!/bin/bash

# Set the folder path
folder="/srv/rl2-lab/flash8/mbronars3/workspace/diffusion_policy/config/ICRA/2block_base_param_unet"

# Set the base command
base_command="python train.py --config-dir /srv/rl2-lab/flash8/mbronars3/workspace/diffusion_policy/config/ICRA/2block_base_param_unet hydra.run.dir='/srv/rl2-lab/flash8/mbronars3/ICRA/results/hyperparam_scan/2block_base_params/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'"

# Loop through files in the folder and execute the base command with --config-name
for file in "$folder"/*; do
    if [ -f "$file" ]; then
        echo "Running $file"
        $base_command --config-name "$(basename "$file")"
    else
        echo "Skipping $file (not executable)"
    fi
done