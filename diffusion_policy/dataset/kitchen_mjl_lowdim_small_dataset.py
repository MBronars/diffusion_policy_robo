from typing import Dict
import torch
import numpy as np
import copy
import pathlib
from tqdm import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env.kitchen.kitchen_util import parse_mjl_logs


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

class KitchenMjlLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=True,
            robot_noise_ratio=0.0,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        robot_pos_noise_amp = np.array([0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   ,
            0.1   , 0.005 , 0.005 , 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
            0.0005, 0.005 , 0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ,
            0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ], dtype=np.float32)
        rng = np.random.default_rng(seed=seed)

        data_directory = pathlib.Path(dataset_dir)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i, mjl_path in enumerate(tqdm(list(data_directory.glob('*/*.mjl')))):
            try:
                data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                qpos = data['qpos'].astype(np.float32)
                obs = np.concatenate([
                    qpos[:,:9],
                    qpos[:,-21:]
                    #np.zeros((len(qpos),30),dtype=np.float32)
                ], axis=-1)

                start_idx = 0
                last_nonzero = 0
                last_goal = np.zeros(30)
                for i, ob in enumerate(obs):
                    goal = np.zeros(30)
                    holder = np.zeros(30)
                    for goals, thresholds in OBS_ELEMENT_GOALS.items():
                        current = ob[OBS_ELEMENT_INDICES[goals]]
                        target = thresholds
                        distance = np.linalg.norm(current - target)
                        include = distance < BONUS_THRESH
                        if include:
                            goal[OBS_ELEMENT_INDICES[goals]] = current
                            holder[OBS_ELEMENT_INDICES[goals]] = current
                    
                    # get number of non-zero elements in goal
                    num_nonzero = np.count_nonzero(goal)
                    if num_nonzero > last_nonzero:
                        last_nonzero = num_nonzero
                        curr_obs = obs[start_idx:i+1]

                        # maybe remove these three lines if things break
                        # temp = last_goal.copy()
                        # # get indicies of non-zero elements
                        # last_goal = goal.copy()
                        # # set non-zero elements to zero
                        # goal[np.where(temp != 0)[0]] = 0

                        goal = goal[None,:].repeat(len(curr_obs), axis=0)


                        if robot_noise_ratio > 0:
                            # add observation noise to match real robot
                            noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                                low=-1., high=1., size=(curr_obs.shape[0], 30))
                            curr_obs[:,:30] += noise

                        

                        curr_data = data['ctrl'][start_idx:i+1].astype(np.float32)
                        episode = {
                            'obs': curr_obs,
                            'goal': goal,
                            'action': curr_data
                        }
                        self.replay_buffer.add_episode(episode)
                        start_idx = i+1



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

                goal = goal[None,:].repeat(len(obs), axis=0)

                if robot_noise_ratio > 0:
                    # add observation noise to match real robot
                    noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                        low=-1., high=1., size=(obs.shape[0], 30))
                    obs[:,:30] += noise
                episode = {
                    'obs': obs,
                    'goal': goal,
                    'action': data['ctrl'].astype(np.float32)
                }
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                print(i, e)

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