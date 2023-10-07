#!/bin/bash

# Define parameter ranges
alpha_values=("-1" "-.5" "-.25" "0" ".25" )
beta_values=("-1" "-.5" "-.25" "0" ".25" )
gamma_values=("0.01" "0.1" "1")

output_file="/home/MBronars/workspace/ICRA/results/hyperparam_scan/output.txt"
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
  python /home/MBronars/workspace/ICRA/scripts/yaml_update.py $alpha $beta $gamma

  # Run your program with the current random parameter combination and capture the output
  python robomimicEval.py --config-dir /home/MBronars/workspace/diffusion_policy_robo/diffusion_policy/config/task --config-name stack_lowdim_sim.yaml  hydra.run.dir='/home/MBronars/Documents/ICML_paper/diffuser/data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
  
  eval_save_path="/home/MBronars/Documents/ICML_paper/datasets/alpha${alpha}_beta${beta}_gamma${gamma}.hdf5"

  # Pipe the program output to the evaluation program
  evaluation_result=$(python /home/MBronars/workspace/ICRA/scripts/evaluate_legibility.py $eval_save_path)

  # Store the evaluation result along with the parameters in the output file
  echo "Parameter 1: $alpha" >> "$output_file"
  echo "Parameter 2: $beta" >> "$output_file"
  echo "Parameter 3: $gamma" >> "$output_file"

  echo "(Legibility, Success Rate): $evaluation_result" >> "$output_file"
  echo "------" >> "$output_file"  # Separation between results
done

echo "Random parameter sweep complete. Results stored in $output_file"
