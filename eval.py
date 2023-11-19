"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import h5py
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from evaluate_legibility import get_legibility

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-a', '--alpha', default=0.85)
@click.option('-g', '--gamma', default=0.75)
@click.option('-w', '--guidance_weight', default=10)
def main(checkpoint, output_dir, device, alpha, gamma, guidance_weight):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # 0, .1, .5, .9, 1
    cfg.task.env_runner.alpha = alpha

    # 0.25, 0.5, 0.75, 0.99, 1
    cfg.task.env_runner.gamma = gamma

    cfg.task.env_runner.max_steps= 100

    cfg.task.env_runner.n_test = 25


    # -1, 0, 1, 2, 5, 10
    cfg.task.env_runner.guidance_weight = guidance_weight

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    # workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    # policy = workspace.model
    # if cfg.training.use_ema:
    #     policy = workspace.ema_model

    # policy = workspace.policy
    policy = workspace.model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    dataset: BaseLowdimDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        dataset=dataset)

    runner_log, trajs = env_runner.run(policy, save_rollout = True)

    data_writer = h5py.File(os.path.join(output_dir, 'trajs.hdf5'), "w")
    data_grp = data_writer.create_group("data")
    demo_counter = 0
    total_samples = 0
    for i in range(len(trajs)):
        for idx in range(len(trajs[i])):
            ep_data_grp = data_grp.create_group("demo_{}".format(demo_counter))
            ep_data_grp.create_dataset("actions", data=np.vstack(trajs[i][idx]["actions"]))
            ep_data_grp.create_dataset("rewards", data=np.array(trajs[i][idx]["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(trajs[i][idx]["dones"]))
            ep_data_grp.create_dataset("obs", data=np.vstack(trajs[i][idx]["obs"]))
            ep_data_grp.create_dataset("next_obs", data=np.vstack(trajs[i][idx]["next_obs"]))
            ep_data_grp.create_dataset("states", data=np.array(trajs[i][idx]["states"]))
            ep_data_grp.create_dataset("red_goal", data=np.array(trajs[i][idx]["red_goal"]))
            num_samples = len(data_grp["demo_{}".format(demo_counter)]["actions"])
            ep_data_grp.attrs["num_samples"] = num_samples
            ep_data_grp.attrs["model_file"] = trajs[i][idx]["initial_state_dict"]["model"]
            total_samples += num_samples
            demo_counter += 1
            
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = trajs[0][0]["env_args"]
    data_writer.close()

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    get_legibility(os.path.join(output_dir, 'trajs.hdf5'), 8)

if __name__ == '__main__':
    main()
