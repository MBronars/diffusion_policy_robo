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

    data_writer = h5py.File("/home/MBronars/Documents/ICML_paper/datasets/diff_low_dim_long_g-10.hdf5", "w")
    data_grp = data_writer.create_group("data")
    total_samples = 0

    for i in range(1):

        # trajs = workspace.eval(ckpt_path="/home/MBronars/Documents/ICML_paper/diffuser/data/outputs/2023.07.24/16.43.41_train_diffusion_transformer_lowdim_long_stack_lowdim_long/checkpoints/epoch=0150-test_mean_score=1.000.ckpt")
        trajs = workspace.eval(ckpt_path="/home/MBronars/Documents/ICML_paper/diffuser/data/outputs/2023.08.18/04.01.05_train_diffusion_transformer_lowdim_long_stack_lowdim_long/checkpoints/epoch=0100-test_mean_score=1.000.ckpt")
        # assert isinstance(traj, dict)
        # assert isinstance(traj["obs"], dict)
        for i in range(len(trajs)):

            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(trajs[i]["actions"]))
            ep_data_grp.create_dataset("rewards", data=np.array(trajs[i]["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(trajs[i]["dones"]))
            ep_data_grp.create_dataset("obs", data=np.array(trajs[i]["obs"]))
            ep_data_grp.create_dataset("next_obs", data=np.array(trajs[i]["next_obs"]))
        # for k in traj["obs"]:
        #     ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
        #     ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

        # episode metadata
        # ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
        # total_samples += traj["actions"].shape[0]
    
    # data_grp.attrs["total"] = total_samples
    # data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    data_writer.close()
    print("Wrote dataset trajectories to {}".format("/home/MBronars/Documents/ICML_paper/datasets/diff_low_dim_e400.hdf5"))



if __name__ == "__main__":
    main()
    # parser = argparse.ArgumentParser()

    # # Path to trained model
    # parser.add_argument(
    #     "--ckpt",
    #     type=str,
    #     required=True,
    #     help="path to saved ckpt file",
    # )

    # # number of rollouts
    # parser.add_argument(
    #     "--n_rollouts",
    #     type=int,
    #     default=27,
    #     help="number of rollouts",
    # )

    # # config file
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     required=True,
    #     help="path to config file",
    # )

    # # If provided, an hdf5 file will be written with the rollout data
    # parser.add_argument(
    #     "--dataset_path",
    #     type=str,
    #     default=None,
    #     help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    # )

    # # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    # parser.add_argument(
    #     "--dataset_obs",
    #     action='store_true',
    #     help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
    #         observations are excluded and only simulator states are saved)",
    # )

    # # for seeding before starting rollouts
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=None,
    #     help="(optional) set seed for rollouts",
    # )

    # args = parser.parse_args()
    # main(args)