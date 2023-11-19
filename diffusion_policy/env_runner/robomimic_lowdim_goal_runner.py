import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from copy import deepcopy
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env

# Notes on changes that are made for goal conditioning
# We require the LowdimRunner to be passed in a dataset
# The dataset should handle all of the


class RobomimicLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            obs_keys,
            dataset,
            guidance_weight,
            alpha,
            gamma,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        """
        Assuming:
        n_obs_steps=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|

        for goal conditioning, we require:
        dataset = RobomimicReplayLowdimDataset
        guidance_list = list of guidance weights for each goal
        decay = decay rate for guidance weights (first goal approaches 1, rest approach 0)
        """

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicLowdimWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = init_state


                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)
                

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        env = AsyncVectorEnv(env_fns)
        # env = SyncVectorEnv(env_fns)

        assert isinstance(dataset, RobomimicReplayLowdimDataset)
        
        goal_list = []
        goal_names = []
        rn_gen = np.random.default_rng(seed=test_start_seed)
        val_dataset = dataset.get_validation_dataset()
        
        
        for i in range(len(env_init_fn_dills)):
            # seed = start_seed + i
            goals, goal_name = dataset.get_goal_list(val_set = val_dataset, rng=rn_gen)
            goal_list.append(goals)
            goal_names.append(goal_name)

        

        # goal_list = [dataset.get_goal_list() for _ in range(len(env_init_fn_dills))]
        transposed_goals = [list(x) for x in zip(*goal_list)]
        final_goal_list = [torch.stack(x, dim=0) for x in transposed_goals]
        final_goal_list = [goal[:,-n_obs_steps:] for goal in final_goal_list]
        
        self.goal_list = final_goal_list
        self.goal_names = goal_names
        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        # self.guidance_list = guidance_list
        self.guidance_weight = guidance_weight
        self.alpha = alpha
        self.gamma = gamma

    def decay_weights(self, guidance_list):
        """Decay the weights of the guidance list, first weight approaches 1, rest approach 0"""
        # decay all but the first weight
        for i in range(1, len(guidance_list)):
            guidance_list[i] = guidance_list[i] * self.decay
        # set the first weight so that the sum of all weights is 1
        guidance_list[0] = 1 - sum(guidance_list[1:])
        return guidance_list

    def check_goal(self, obs, goal):
        """Check if the goal has been reached
           Default performance metric (commented out) is if the end effector is within 5cm of the goal
           We use task specific metrics, which should be changed depending on the task
        """
        reduced_goal = goal[:, -1, [2, 9]].detach().cpu().numpy()
        reduced_obs = obs[:, -1, [2, 9]]
        target = np.where(reduced_goal[..., 0] > reduced_goal[..., 1], 1, 2)
        condition = (reduced_obs[..., 0] > 0.84) | (reduced_obs[..., 1] > 0.84)
        current = np.where(condition, np.where(reduced_obs[..., 0] > reduced_obs[..., 1], 1, 2), 0)
        successes = (current == target).astype(float)
        return successes

    def run(self, policy: BaseLowdimPolicy, save_rollout = False):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # rollout data
        trajs = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

           
            # reset guidance list
            # guidance_list = self.guidance_list.copy()
            guidance_weight = self.guidance_weight

            # set successes to false
            total_successes = [0] * this_n_active_envs

            # convert goal list
            goal_list = [goal[this_global_slice] for goal in self.goal_list]
            goal_list = [{'goal': goal} for goal in goal_list]
            goal_list = [dict_apply(goal, lambda x: x.to(device=device)) for goal in goal_list]
            
            current_goals = self.goal_names[this_global_slice]

            # init trajectory
            # ,
            traj = [dict(actions=[], rewards=[], dones=[], states=[], obs=[], next_obs=[], initial_state_dict= env.call("get_state")[i], env_args = env.call("get_env_args")[i], red_goal = current_goals[i]) for i in range(this_n_active_envs)]

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Lowdim {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:

                small_list = [-self.alpha * guidance_weight, - (1 - self.alpha) * guidance_weight]
                guidance_list = [1 - sum(small_list), small_list[0], small_list[1]]
                
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[:,:self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                # policy is dependant on a goal list and guidance weights
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict, goal_list = goal_list, guidance_weights = guidance_list)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                next_obs, reward, dones, info = env.step(env_action)
                state_dict = env.call("get_state")
                done = np.all(dones)
                past_action = action
                

                # check if goal is reach - target is the first goal in the list
                successes = self.check_goal(obs, goal_list[0]['goal'])
                total_successes = [1.0 if total_successes[i] or successes[i] else 0.0 for i in range(len(total_successes))]

                if save_rollout:
                    # collect transition
                    for i in range(this_n_active_envs):
                        traj[i]["actions"].append(env_action[i])
                        traj[i]["rewards"].append(total_successes[i])
                        traj[i]["dones"].append(dones[i])
                        traj[i]["obs"].append(obs[i])
                        traj[i]["next_obs"].append(next_obs[i])
                        traj[i]["states"].append(state_dict[i]['states'])

                obs = deepcopy(next_obs)

                # update successes

                done = done or np.all(total_successes)

                # decay guidance weights
                guidance_weight = guidance_weight * self.gamma

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]

            # for goal conditioning, we are just doing the rewards based on self computed successes
            #all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            all_rewards[this_global_slice] = total_successes


            # collect trajectory
            if save_rollout:
                trajs.append(traj)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        if save_rollout:
            return log_data, trajs

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
