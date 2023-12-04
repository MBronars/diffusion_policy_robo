from typing import Dict, List
import torch
import numpy as np
import h5py
import time
from tqdm import tqdm
import copy
import random
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)

class RobomimicReplayLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_keys: List[str]=[
                'object', 
                'robot0_eef_pos', 
                'robot0_eef_quat', 
                'robot0_gripper_qpos'],
            abs_action=False,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
        ):
        obs_keys = list(obs_keys)
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file['data']
            # red = np.load("/srv/rl2-lab/flash8/mbronars3/RAL/datasets/red.npy")
            # green = np.load("/srv/rl2-lab/flash8/mbronars3/RAL/datasets/green.npy")
            red = None
            green = None
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[f'demo_{i}']
                episode = _data_to_obs(
                    raw_obs=demo['obs'],
                    raw_actions=demo['actions'][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                    horizon=horizon,
                    red = red,
                    green = green)
                # goal_data = episode['goal']
                # if goal_data[-1, 2] > goal_data[-1, 9]:
                #     if red is None:
                #         red = goal_data[-1]
                # else:
                #     if green is None:
                #         green = goal_data[-1]
                
                replay_buffer.add_episode(episode)

        # np.save("/srv/rl2-lab/flash8/mbronars3/RAL/datasets/red.npy", red)
        # np.save("/srv/rl2-lab/flash8/mbronars3/RAL/datasets/green.npy", green)
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.training_seed = seed
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.abs_action = abs_action
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

        val_mask = ~self.train_mask
        val_idx = np.where(val_mask)[0]
        rng = np.random.default_rng(seed=seed)
        test_idxs = rng.choice(val_idx, size=int(len(val_idx)/3), replace=False)
        test_mask = np.zeros_like(self.train_mask)
        test_mask[test_idxs] = True
        val_mask[test_idxs] = False
        self.val_mask = val_mask
        self.test_mask = test_mask
        

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        # val_set.train_mask = ~self.train_mask
        return val_set

    #     self.get_validation_dataset()
    
    # def get_validation_dataset(self):
    #     val_set = copy.copy(self)
    #     test_set = copy.copy(self)
    #     val_mask = ~self.train_mask
    #     val_data = np.where(val_mask)[0]
    #     rng = np.random.default_rng(seed=seed)
    #     val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    #     test_data = np.random.choice(val_data, size=int(len(val_data)/3), replace=False)
    #     test_mask = np.zeros_like(val_mask)
    #     test_mask = test_mask.astype(bool)
    #     test_mask[test_data] = True
    #     val_mask[test_data] = False
    #     val_set.sampler = SequenceSampler(
    #         replay_buffer=self.replay_buffer, 
    #         sequence_length=self.horizon,
    #         pad_before=self.pad_before, 
    #         pad_after=self.pad_after,
    #         episode_mask=val_mask
    #         )
    #     test_set.sampler = SequenceSampler(
    #         replay_buffer=self.replay_buffer,
    #         sequence_length=self.horizon,
    #         pad_before=self.pad_before,
    #         pad_after=self.pad_after,
    #         episode_mask=test_mask
    #         )
    #     val_set.train_mask = val_mask
    #     test_set.train_mask = test_mask
    #     self.val_mask = val_mask
    #     self.test_mask = test_mask
    #     return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer
        
        # aggregate obs stats
        obs_stat = array_to_stats(self.replay_buffer['obs'])

        normalizer['obs'] = normalizer_from_stat(obs_stat)
        normalizer['goal'] = normalizer_from_stat(obs_stat)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])
    
    def get_goal_list(self, val_set, rng) -> torch.Tensor:
        red = None
        green = None
        # while red is None or green is None:
        #     val_len = val_set.sampler.__len__()
        #     val_idx = rng.choice(range(val_len), size=1, replace=False)[0]
        #     data = val_set.sampler.sample_sequence(val_idx)
        #     torch_data = dict_apply(data, torch.from_numpy)
        #     goal_data = torch_data['goal']
        #     if goal_data[-1, 2] > goal_data[-1, 9]:
        #         if red is None:
        #             red = goal_data
        #     else:
        #         if green is None:
        #             green = goal_data
        
        # red = np.load("/srv/rl2-lab/flash8/mbronars3/RAL/datasets/red.npy")
        # green = np.load("/srv/rl2-lab/flash8/mbronars3/RAL/datasets/green.npy")

        


        # null_data = torch.zeros_like(red)

        red = np.load("/srv/rl2-lab/flash8/mbronars3/RAL/datasets/new_red.npy")
        green = np.load("/srv/rl2-lab/flash8/mbronars3/RAL/datasets/new_green.npy")
        # red = np.load("/srv/rl2-lab/flash8/mbronars3/RAL/datasets/real_red.npy")
        # green = np.load("/srv/rl2-lab/flash8/mbronars3/RAL/datasets/real_green.npy")
        red = torch.from_numpy(red)
        green = torch.from_numpy(green)
        # broadcast green from shape ([32]) to ([16, 32])
        green = green.unsqueeze(0).repeat(16, 1)
        red = red.unsqueeze(0).repeat(16, 1)
        null_data = torch.zeros_like(red)

        if rng.random() < 0.5:
            return ([red, green, null_data], True)
        return ([green, red, null_data], False)
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
    
def _data_to_obs(raw_obs, raw_actions, obs_keys, abs_action, rotation_transformer, horizon, red, green):
    obs = np.concatenate([
        raw_obs[key] if key != 'object' else raw_obs[key][:, :20] for key in obs_keys
        ], axis=-1).astype(np.float32)
    # syntax for in line if else
    # [on_true] if [expression] else [on_false]

    # we define goal state as the final observation from the dataset
    goal = np.zeros_like(obs)
    last_obs = obs[-1]

    # if last_obs[2] > last_obs[9]:
    #     if red is not None:
    #         last_obs = red
    # else:
    #     if green is not None:
    #         last_obs = green
    

    goal[:] = last_obs

    

    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
    
    data = {
        'obs': obs,
        'goal': goal,
        'action': raw_actions
    }
    return data
