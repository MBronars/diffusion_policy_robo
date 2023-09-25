#!/bin/bash

# Define parameter ranges
alpha_values=("-.2" "-.1" "0" ".1" ".2")
beta_values=("2.5")
gamma_values=(".99" ".95" ".85" ".75" )

output_file="/srv/rl2-lab/flash8/mbronars3/legibility/full_training_scan/unet_delta_default_sweep/unet_output_2.txt"
num_samples=40  # Number of random samples

# Clear or create the output file
> "$output_file"

# Function to generate a random parameter combination
generate_random_combination() {
  local alpha="${alpha_values[$RANDOM % ${#alpha_values[@]}]}"
  local beta="${beta_values[$RANDOM % ${#beta_values[@]}]}"
  local gamma="${gamma_values[$RANDOM % ${#gamma_values[@]}]}"
  echo "$alpha $beta $gamma"
}

# Initialize an array to track used combinations
used_combinations=()

# Loop to generate and evaluate random parameter combinations
for ((i=0; i<num_samples; i++)); do
  # Generate a random combination and check if it's already used
  random_combination=$(generate_random_combination)
  while [[ " ${used_combinations[*]} " == *" $random_combination "* ]]; do
    random_combination=$(generate_random_combination)
  done

  # Add the combination to the list of used combinations
  used_combinations+=("$random_combination")

  alpha=$(echo "$random_combination" | awk '{print $1}')
  beta=$(echo "$random_combination" | awk '{print $2}')
  gamma=$(echo "$random_combination" | awk '{print $3}')

  # Change the yaml file to use the current random parameter combination
  python /srv/rl2-lab/flash8/mbronars3/ICRA/scripts/update_yaml_2.py $alpha $beta $gamma

  # Run your program with the current random parameter combination and capture the output
  python robomimicEval.py --config-dir /srv/rl2-lab/flash8/mbronars3/workspace/diffusion_policy_robo/diffusion_policy/config/final_sweep --config-name stack_lowdim_unet_2.yaml  hydra.run.dir='/srv/rl2-lab/flash8/mbronars3/legibility/runs/${now:%H.%M.%S}_${name}_${task_name}'
  
  eval_save_path="/srv/rl2-lab/flash8/mbronars3/legibility/full_training_scan/unet_delta_default_sweep/alpha${alpha}_beta${beta}_gamma${gamma}.hdf5"

  # Pipe the program output to the evaluation program
  evaluation_result=$(python /srv/rl2-lab/flash8/mbronars3/ICRA/scripts/evaluate_legibility.py $eval_save_path 8)

  # Store the evaluation result along with the parameters in the output file
  echo "Parameter 1: $alpha" >> "$output_file"
  echo "Parameter 2: $beta" >> "$output_file"
  echo "Parameter 3: $gamma" >> "$output_file"

  echo "(Legibility, Success Rate): $evaluation_result" >> "$output_file"
  echo "------" >> "$output_file"  # Separation between results
done

echo "Random parameter sweep complete. Results stored in $output_file"
