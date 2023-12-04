from typing import Dict
import torch
import numpy as np
import copy
import pathlib
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
import einops
def transpose_batch_timestep(*args):
    return (einops.rearrange(arg, "b t ... -> t b ...") for arg in args)


OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3

class KitchenLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()

        data_directory = pathlib.Path(dataset_dir)
        observations = np.load(data_directory / "observations_seq.npy")
        actions = np.load(data_directory / "actions_seq.npy")
        masks = np.load(data_directory / "existence_mask.npy")

        observations, actions, masks = transpose_batch_timestep(observations, actions, masks)

        average_objects = {"microwave" : [], "bottom burner" : [], "top burner" : [], "light switch" : [], "slide cabinet" : [], "hinge cabinet" : [], "kettle" : []}

        target_objects = ["microwave", "kettle", "bottom burner", "light switch", "slide cabinet"]

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i in range(len(masks)):
            eps_len = int(masks[i].sum())
            obs = observations[i,:eps_len].astype(np.float32)
            obs = obs[:,:30]

            current_objects = {"microwave" : None, "bottom burner" : None, "top burner" : None, "light switch" : None, "slide cabinet" : None, "hinge cabinet" : None, "kettle" : None}
            start_idx = 0
            last_nonzero = 0
            last_goal = np.zeros(30)

            count = 0
            for idx, ob in enumerate(obs):
                goal = np.zeros(30)
                holder = np.zeros(30)
                for goals, thresholds in OBS_ELEMENT_GOALS.items():
                    current = ob[OBS_ELEMENT_INDICES[goals]]
                    target = thresholds
                    distance = np.linalg.norm(current - target)
                    include = distance < BONUS_THRESH
                    if include and count == 0:
                        goal[OBS_ELEMENT_INDICES[goals]] = current
                        holder[OBS_ELEMENT_INDICES[goals]] = current
                        if current_objects[goals] is None:
                            current_objects[goals] = idx
                        count += 1

                num_nonzero = np.count_nonzero(goal)
                if num_nonzero > last_nonzero:
                    last_nonzero = num_nonzero
                    curr_obs = obs[start_idx:idx+1]

                    goal = goal[None,:].repeat(len(curr_obs), axis=0)
                    curr_data = actions[i, start_idx:idx+1].astype(np.float32)
                    episode = {
                        'obs': curr_obs,
                        'goal': goal,
                        'action': curr_data
                    }
                    self.replay_buffer.add_episode(episode)
                    start_idx = idx+1
            
            # get the keys that are not associated with null values
            keys = [key for key, value in current_objects.items() if value is not None]

            # check if all of the keys are in the target objects
            if all(elem in target_objects for elem in keys):
                for key in target_objects:
                    if key in keys:
                        average_objects[key].append((280 - current_objects[key]) / 280.0)
                    else:
                        average_objects[key].append(0)



            goal = obs[-1,:]
            raw_goal = obs[-1, :]

            goal = np.zeros(30)

            # mask out the elements that are not within the bonus threshold for each task
            for goals, thresholds in OBS_ELEMENT_GOALS.items():
                current = raw_goal[OBS_ELEMENT_INDICES[goals]]
                target = thresholds
                distance = np.linalg.norm(current - target)
                include = distance < BONUS_THRESH
                if include:
                    goal[OBS_ELEMENT_INDICES[goals]] = current
            
            goal = goal[None,:].repeat(eps_len, axis=0)
            action = actions[i,:eps_len].astype(np.float32)
            data = {                              
                'obs': obs,
                'goal': goal,
                'action': action
            }
            self.replay_buffer.add_episode(data)
        
        from IPython import embed; embed()
        0/0
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'obs': self.replay_buffer['obs'],
            'goal': self.replay_buffer['goal'],
            'action': self.replay_buffer['action']
        }
        if 'range_eps' not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs['range_eps'] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data