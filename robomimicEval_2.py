import argparse
import copy
import h5py
import numpy as np
import json

from diffusion_policy.workspace.train_robomimic_lowdim_workspace import TrainRobomimicLowdimWorkspace
from diffusion_policy.workspace.train_robomimic_image_workspace import TrainRobomimicImageWorkspace

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)

def main(cfg: OmegaConf):

    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: TrainRobomimicLowdimWorkspace = cls(cfg)   
    
    data_writer = h5py.File(cfg.eval_save, "w")
    data_grp = data_writer.create_group("data")
    total_samples = 0

    for i in range(1):

        #trajs = workspace.eval("/srv/rl2-lab/flash8/mbronars3/ICRA/results/ablations/20.03.36_train_diffusion_transformer_lowdim_long_stack_lowdim_long/checkpoints/epoch=0700-test_mean_score=1.000.ckpt")
        trajs = workspace.eval("/srv/rl2-lab/flash8/mbronars3/ICRA/results/eval_ckpts/20.16.16_train_diffusion_transformer_lowdim_long_stack_lowdim_long/checkpoints/epoch=0200-test_mean_score=1.000.ckpt")

        for i in range(len(trajs)):

            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(trajs[i]["actions"]))
            ep_data_grp.create_dataset("rewards", data=np.array(trajs[i]["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(trajs[i]["dones"]))
            ep_data_grp.create_dataset("obs", data=np.array(trajs[i]["obs"]))
            ep_data_grp.create_dataset("next_obs", data=np.array(trajs[i]["next_obs"]))

    
    print("Wrote dataset trajectories to {}".format(cfg.eval_save))





if __name__ == "__main__":
    main()
