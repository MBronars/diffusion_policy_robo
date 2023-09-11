import argparse
import copy
import h5py
import numpy as np
import json
import math
import random
import os
import re

from diffusion_policy.workspace.train_robomimic_lowdim_workspace import TrainRobomimicLowdimWorkspace
from diffusion_policy.workspace.train_robomimic_image_workspace import TrainRobomimicImageWorkspace
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def extract_test_mean_score(filename):
    # Extract the test_mean_score from the filename
    match = re.search(r'test_mean_score=([0-9]+\.[0-9]+)\.ckpt', filename)
    if match:
        return float(match.group(1))
    return None

def main_wrapper(ckpt:str, config_name:str, rel_config_path:str):


    @hydra.main(config_path = rel_config_path, config_name = config_name)
    def main(cfg: OmegaConf):
        
        OmegaConf.resolve(cfg)

        cls = hydra.utils.get_class(cfg._target_)
        workspace: TrainRobomimicLowdimWorkspace = cls(cfg)   

        save_path = "/srv/rl2-lab/flash8/mbronars3/ICRA/results/eval_runs/scan_datasets/"
        ckpt_name = ckpt.split("/")[-1][:-5]

        save_path = save_path + cfg.logging.name + "-" + ckpt_name + ".hdf5"

        data_writer = h5py.File(cfg.eval_save, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

        for i in range(1):

            trajs = workspace.eval(ckpt)

            for i in range(len(trajs)):

                ep_data_grp = data_grp.create_group("demo_{}".format(i))
                ep_data_grp.create_dataset("actions", data=np.array(trajs[i]["actions"]))
                ep_data_grp.create_dataset("rewards", data=np.array(trajs[i]["rewards"]))
                ep_data_grp.create_dataset("dones", data=np.array(trajs[i]["dones"]))
                ep_data_grp.create_dataset("obs", data=np.array(trajs[i]["obs"]))
                ep_data_grp.create_dataset("next_obs", data=np.array(trajs[i]["next_obs"]))

        data_writer.close()
        
        print("Wrote dataset trajectories to {}".format(save_path))

        if "2block" in cfg.task.dataset_path:
            get_2block_legibility(filename = save_path, n_action_steps = cfg.task.env_runner.n_action_steps)
    main()


def get_2block_legibility(filename, n_action_steps):
    fig = plt.figure(figsize=(12,5))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.view_init(elev=30, azim=40)
    ax1.set_box_aspect([1,1,1])

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.view_init(elev=30, azim=0)
    ax2.set_box_aspect([1,1,1])

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.view_init(elev=30, azim=-40)
    ax3.set_box_aspect([1,1,1])

    #Legend for success plots
    circle1 = plt.Line2D([], [], color='red', linewidth=3)
    circle2 = plt.Line2D([], [], color='green', linewidth=3)
    circle3 = plt.Line2D([], [], color='black', linewidth=3)

    fig.legend([circle1, circle2, circle3], ['Picked Up Red', 'Picked Up Green', 'Failure'], loc='upper center', fontsize = 15, ncol=3)

    total_success = 0
    total_red = 0
    total_green = 0
    total = 0

    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf
    z_min = math.inf
    z_max = -math.inf

    max_green_legibility = 0
    min_green_legibility = math.inf
    max_red_legibility = 0
    min_red_legibility = math.inf
    total_legibility = [] 

    demo_file = "/srv/rl2-lab/flash8/mbronars3/ICRA/datasets/2block_image.hdf5"
    with h5py.File(demo_file, "r") as f:
        for index, demo in enumerate(f['data']):

            trajectory = f['data'][demo]['obs']['robot0_eef_pos'][()]
            final_heights = f['data'][demo]['obs']['object'][:, [2, 9]][-1]

            success = final_heights[0] > 0.84 or final_heights[1] > 0.84

            red_pos = f['data'][demo]["obs"]["object"][:, 0:3] 
            green_pos = f['data'][demo]["obs"]["object"][:, 7:10]

            # only keep every n_action_steps
            trajectory = trajectory[::n_action_steps]
            red_pos = red_pos[::n_action_steps]
            green_pos = green_pos[::n_action_steps]

            red = False

            if success and final_heights[0] > final_heights[1]:
                legibility = np.linalg.norm(trajectory - green_pos, axis=1)
                legibility = sum(legibility / range(1, 1+ len(legibility)))
                if legibility < min_red_legibility:
                    min_red_legibility = legibility
                if legibility > max_red_legibility:
                    max_red_legibility = legibility
                red = True
            elif success:
                legibility = np.linalg.norm(trajectory - red_pos, axis=1) 
                legibility = sum(legibility / range(1, 1 + len(legibility)))
                if legibility < min_green_legibility:
                    min_green_legibility = legibility
                if legibility > max_green_legibility:
                    max_green_legibility = legibility
    
    with h5py.File(filename, "r") as f:
        for index, demo in enumerate(f['data']):
            _, length, _, _ = f['data'][demo]['next_obs'].shape
            for i in range(length):
                total += 1
                
                trajectory = f['data'][demo]['next_obs'][:, i, 1, [23, 24, 25]]
                heights = f['data'][demo]['next_obs'][:, i, 1, [2, 9]]

                red_pos = f['data'][demo]['next_obs'][:, i, 1, [0, 1, 2]]
                green_pos = f['data'][demo]['next_obs'][:, i, 1, [7, 8, 9]]

                successes = heights > 0.84
                success = successes.any()
                red = successes[:, 0].any()

                if index%2 == 0:
                    success = success and red
                else:
                    success = success and not red

                for i, x in enumerate(successes):
                    if x.any():
                        trajectory = trajectory[:i]
                        red_pos = red_pos[:i]
                        green_pos = green_pos[:i]
                        break
                
                if success and red:
                    legibility = np.linalg.norm(trajectory - green_pos, axis=1)
                    legibility = sum(legibility / range(1, 1+ len(legibility)))
                    legibility = (legibility - min_red_legibility) / (max_red_legibility - min_red_legibility)
                    total_legibility.append(legibility)
                    total_red += 1
                    total_success += 1
                elif success:
                    legibility = np.linalg.norm(trajectory - red_pos, axis=1) 
                    legibility = sum(legibility / range(1, 1 + len(legibility)))
                    legibility = (legibility - min_green_legibility) / (max_green_legibility - min_green_legibility)
                    total_legibility.append(legibility)
                    total_green += 1
                    total_success += 1
                if not success:
                    c = 'black'
                elif red:
                    c = get_cmap('Reds')(random.randint(50, 90)/100)
                else:
                    c = get_cmap('Greens')(random.randint(50, 90)/100)


                # Extract x, y, and z coordinates from trajectory
                x = trajectory[:, 0]
                y = trajectory[:, 1]
                z = trajectory[:, 2]

                #change max and min values of x, y, and z if necessary
                if x.min() < x_min and x.min() > -.3:
                    x_min = x.min()
                if x.max() > x_max and x.max() < .3:
                    x_max = x.max()
                if y.min() < y_min and y.min() > -.3:
                    y_min = y.min()
                if y.max() > y_max and y.max() < .3:
                    y_max = y.max()
                if z.min() < z_min and z.min() > -1:
                    z_min = z.min()
                if z.max() > z_max and z.max() < 1.1:
                    z_max = z.max()
                
                ax1.plot(x, y, z, color = c)
                ax2.plot(x, y, z, color = c)
                ax3.plot(x, y, z, color = c)

            #set axis limits    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_zlim(z_min, z_max)

    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])

    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_zlim(z_min, z_max)

    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])

    if len(total_legibility) > 0:
        final_legibility = np.mean(total_legibility)
    else:
        final_legibility = 0
    success_rate = total_success / total

    # Display total success, red, and green
    if total_success > 0:
        fig.text(0.2, 0.05, 'Success Rate: ' + str(round(total_success/total, 4)), ha='center', fontsize=12)
        fig.text(0.4, 0.05, 'Legibility: ' + str(round(final_legibility, 4)), ha='center', fontsize=12)
        fig.text(0.6, 0.05 , 'Num Green: ' + str(total_green), ha='center', fontsize=12)
        fig.text(0.8, 0.05, 'Num Red: ' + str(total_red), ha='center', fontsize=12)


    image_root = "/srv/rl2-lab/flash8/mbronars3/ICRA/results/eval_runs/scan_images/"
    img_name = filename.split("/")[-1]
    img_name = img_name[:-5]
    plt.savefig(image_root + img_name + ".png")

    print((final_legibility, success_rate))

if __name__ == "__main__":

    checkpoints_to_load = 2
    config_dir = "/srv/rl2-lab/flash8/mbronars3/ICRA/results/eval_ckpts"
    relative_path = "../../ICRA/results/eval_ckpts"

    for root, dirs, files in os.walk(config_dir):
        for dir_name in dirs:
            top_checkpoints = []
            config_file = os.path.join(root, dir_name, ".hydra", "config.yaml")
            checkpoint_dir = os.path.join(root, dir_name, "checkpoints")

            if os.path.exists(checkpoint_dir):

                for checkpoint_file in os.listdir(checkpoint_dir):
                    if checkpoint_file.endswith(".ckpt") and "latest" not in checkpoint_file:
                        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_file)
                        score = extract_test_mean_score(checkpoint_file)
                        top_checkpoints.append((checkpoint_file, score))

                top_checkpoints.sort(key=lambda x: x[1], reverse=True)
            
            for i in range(checkpoints_to_load):

                # Check if the config.yaml file exists in the subdirectory
                if os.path.exists(config_file):
                    # Calculate the relative path by removing the common prefix
                    rel_config_path = os.path.relpath(os.path.dirname(config_file), config_dir)
                    rel_config_path = os.path.join(relative_path, rel_config_path)
                    config_name = os.path.basename(config_file)

                    main_wrapper(ckpt=top_checkpoints[i][0], config_name=config_name, rel_config_path=rel_config_path)