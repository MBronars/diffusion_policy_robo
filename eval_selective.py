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
import yaml
import ast
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from evaluate_legibility import get_legibility

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-a', '--alpha', default=.9)
@click.option('-g', '--gamma', default=.9)
@click.option('-w', '--guidance_weight', default= 2.5)
@click.option('-s', '--seed', default=420)
@click.option('-t', '--task', default=["kbtls", "kbtsh"])
def main(checkpoint, output_dir, device, alpha, gamma, guidance_weight, seed, task):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # example 1: ["mbtlsh", "kbtlsh"]
    # example 2: ["mkblsh", "mktlsh"]
    # example 3: ["mkbtsh", "mkbtlh"]

    # example 1: ["mbtlsh", "kbtlsh"]
    # example 2: ["mkblsh", "mktlsh"]
    # example 3: ["mkbtsh", "mkbtlh"]


    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # from IPython import embed; embed()

    file_path = '/srv/rl2-lab/flash8/mbronars3/RAL/selective_interaction/sim/diffusion_policy/config.yaml'

    # file_path = '/srv/rl2-lab/flash8/mbronars3/RAL/selective_interaction/sim/IBC/config.yaml'
    # Load the YAML file into a Python dictionary
    # with open(file_path, 'r') as file:
    #     cfg2 = yaml.safe_load(file)

    # from IPython import embed; embed()
    cfg = OmegaConf.load(file_path)

    # # 0, .1, .5, .9, 1
    cfg.task.env_runner.alpha = alpha

    # # 0.25, 0.5, 0.75, 0.99, 1
    cfg.task.env_runner.gamma = gamma

    cfg.task.env_runner.n_test = 25
    cfg.task.env_runner.n_envs = 25

    # cfg.task.env_runner.test_start_seed = 4200


    # -1, 0, 1, 2, 5, 10
    cfg.task.env_runner.guidance_weight = guidance_weight
    cfg.task.env_runner.goal_names = ast.literal_eval(task)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    # workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    # policy = workspace.policy
    # policy = workspace.model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)

    runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
