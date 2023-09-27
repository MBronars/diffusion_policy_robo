import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import logging
import wandb.sdk.data_types.video as wv
import gym
import gym.spaces
import multiprocessing as mp
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

module_logger = logging.getLogger(__name__)

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

base_goal = np.zeros(30)

kettle_top_bottom_slide = base_goal.copy()
kettle_top_bottom_slide[23] = -0.23
kettle_top_bottom_slide[24] = 0.75
kettle_top_bottom_slide[25] = 1.62
kettle_top_bottom_slide[26] = 0.99
kettle_top_bottom_slide[27] = 0.0
kettle_top_bottom_slide[28] = 0.0
kettle_top_bottom_slide[29] = -0.06
kettle_top_bottom_slide[15] = -0.92
kettle_top_bottom_slide[16] = -0.01
kettle_top_bottom_slide[19] = 0.37

kettle_top_bottom_slide[11] = -0.88
kettle_top_bottom_slide[12] = -0.01

kettle_top_switch_slide = base_goal.copy()
kettle_top_switch_slide[23] = -0.23
kettle_top_switch_slide[24] = 0.75
kettle_top_switch_slide[25] = 1.62
kettle_top_switch_slide[26] = 0.99
kettle_top_switch_slide[27] = 0.0
kettle_top_switch_slide[28] = 0.0
kettle_top_switch_slide[29] = -0.06
kettle_top_switch_slide[15] = -0.92
kettle_top_switch_slide[16] = -0.01
kettle_top_switch_slide[19] = 0.37

kettle_top_switch_slide[17] = -0.69
kettle_top_switch_slide[18] = -0.05



top_bottom_switch_hinge = base_goal.copy()
# top_bottom_switch_hinge[11] = -0.88
# top_bottom_switch_hinge[12] = -0.01
# top_bottom_switch_hinge[15] = -0.92
# top_bottom_switch_hinge[16] = -0.01
# top_bottom_switch_hinge[17] = -0.69
# top_bottom_switch_hinge[18] = -0.05
top_bottom_switch_hinge[20] = 0.0
top_bottom_switch_hinge[21] = 1.45

top_bottom_switch_slide = base_goal.copy()
# top_bottom_switch_slide[11] = -0.88
# top_bottom_switch_slide[12] = -0.01
# top_bottom_switch_slide[15] = -0.92
# top_bottom_switch_slide[16] = -0.01
# top_bottom_switch_slide[17] = -0.69
# top_bottom_switch_slide[18] = -0.05
top_bottom_switch_slide[19] = 0.37

microwave_kettle_top_slide = base_goal.copy()
microwave_kettle_top_slide[22] = -0.75
microwave_kettle_top_slide[23] = -0.23
microwave_kettle_top_slide[24] = 0.75
microwave_kettle_top_slide[25] = 1.62
microwave_kettle_top_slide[26] = 0.99
microwave_kettle_top_slide[27] = 0.0
microwave_kettle_top_slide[28] = 0.0
microwave_kettle_top_slide[29] = -0.06
microwave_kettle_top_slide[15] = -0.92
microwave_kettle_top_slide[16] = -0.01
microwave_kettle_top_slide[19] = 0.37


microwave_kettle_top_bottom = base_goal.copy()
microwave_kettle_top_bottom[22] = -0.75
microwave_kettle_top_bottom[23] = -0.23
microwave_kettle_top_bottom[24] = 0.75
microwave_kettle_top_bottom[25] = 1.62
microwave_kettle_top_bottom[26] = 0.99
microwave_kettle_top_bottom[27] = 0.0
microwave_kettle_top_bottom[28] = 0.0
microwave_kettle_top_bottom[29] = -0.06
microwave_kettle_top_bottom[11] = -0.88
microwave_kettle_top_bottom[12] = -0.01
microwave_kettle_top_bottom[15] = -0.92
microwave_kettle_top_bottom[16] = -0.01








class KitchenLowdimRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            dataset_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=280,
            n_obs_steps=2,
            n_action_steps=8,
            render_hw=(240,360),
            fps=12.5,
            crf=22,
            past_action=False,
            tqdm_interval_sec=5.0,
            abs_action=False,
            robot_noise_ratio=0.1,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        task_fps = 12.5
        steps_per_render = int(max(task_fps // fps, 1))

        def env_fn():
            from diffusion_policy.env.kitchen.v0 import KitchenAllV0
            from diffusion_policy.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
            env = KitchenAllV0(use_abs_action=abs_action)
            env.robot_noise_ratio = robot_noise_ratio
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    KitchenLowdimWrapper(
                        env=env,
                        init_qpos=None,
                        init_qvel=None,
                        render_hw=tuple(render_hw)
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
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        all_init_qpos = np.load(pathlib.Path(dataset_dir) / "all_init_qpos.npy")
        all_init_qvel = np.load(pathlib.Path(dataset_dir) / "all_init_qvel.npy")
        module_logger.info(f'Loaded {len(all_init_qpos)} known initial conditions.')

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis
            init_qpos = None
            init_qvel = None
            if i < len(all_init_qpos):
                init_qpos = all_init_qpos[i]
                init_qvel = all_init_qvel[i]

            def init_fn(env, init_qpos=init_qpos, init_qvel=init_qvel, enable_render=enable_render):
                from diffusion_policy.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
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

                # set initial condition
                assert isinstance(env.env.env, KitchenLowdimWrapper)
                env.env.env.init_qpos = init_qpos
                env.env.env.init_qvel = init_qvel
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                from diffusion_policy.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
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

                # set initial condition
                assert isinstance(env.env.env, KitchenLowdimWrapper)
                env.env.env.init_qpos = None
                env.env.env.init_qvel = None

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        def dummy_env_fn():
            # Avoid importing or using env in the main process
            # to prevent OpenGL context issue with fork.
            # Create a fake env whose sole purpos is to provide 
            # obs/action spaces and metadata.
            env = gym.Env()
            env.observation_space = gym.spaces.Box(
                -8, 8, shape=(60,), dtype=np.float32)
            env.action_space = gym.spaces.Box(
                -8, 8, shape=(9,), dtype=np.float32)
            env.metadata = {
                'render.modes': ['human', 'rgb_array', 'depth_array'],
                'video.frames_per_second': 12
            }
            env = MultiStepWrapper(
                env=env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
            return env
        
        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec


    def run(self, policy: BaseLowdimPolicy):
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
        last_info = [None] * n_inits

        


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

            goals = [kettle_top_switch_slide.copy(), kettle_top_bottom_slide.copy()]
            goal_dicts = [{'goal': goal} for goal in goals]
            ngoals = [policy.normalizer.normalize(goal_dict) for goal_dict in goal_dicts]
            goals = [ng['goal'] for ng in ngoals]
            goal_chunks = [goal.repeat(n_envs, self.n_obs_steps, 1) for goal in goals]
            goal_index = chunk_idx % len(goal_chunks)

            goal = goal_chunks[goal_index][this_local_slice]
            
            other_goals = [other_goal[this_local_slice] for i, other_goal in enumerate(goal_chunks) if i != goal_index]

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval BlockPushLowdimRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False

            alpha = .25
            beta = 2
            gamma = .5

            counter = 0

            while not done:
                # print(beta)
                # create obs dict

                obs = obs[:, :, :30]

                np_obs_dict = {
                    'obs': obs.astype(np.float32)
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
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict, goal = goal, other_goals = other_goals, alpha=alpha, beta=beta)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
                beta = beta * gamma
                alpha = alpha * gamma
                gamma = gamma * gamma

            pbar.close()
            completed_lists = env.call_each('get_completed_tasks')

            tuple_of_tuples = tuple(tuple(lst) for lst in completed_lists)

            # Use Counter to count the occurrences of each tuple
            tuple_counts = Counter(tuple_of_tuples)

            # Specify the file path where you want to save the counts
            file_path = self.output_dir + '/counts.txt'

            # Open the file for writing
            with open(file_path, 'w') as file:
                # Write the counts to the file
                for tpl, count in tuple_counts.items():
                    file.write(f"Tuple: {tpl}, Count: {count}\n")

            print(f"Counts saved to {file_path}")

            


            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            last_info[this_global_slice] = [dict((k,v[-1]) for k, v in x.items()) for x in info][this_local_slice]

        # reward is number of tasks completed, max 7
        # use info to record the order of task completion?
        # also report the probably to completing n tasks (different aggregation of reward).

        # log
        log_data = dict()
        prefix_total_reward_map = collections.defaultdict(list)
        prefix_n_completed_map = collections.defaultdict(list)
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
            this_rewards = all_rewards[i]
            total_reward = np.sum(this_rewards) / 7
            prefix_total_reward_map[prefix].append(total_reward)

            n_completed_tasks = len(last_info[i]['completed_tasks'])
            prefix_n_completed_map[prefix].append(n_completed_tasks)

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in prefix_total_reward_map.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
        for prefix, value in prefix_n_completed_map.items():
            n_completed = np.array(value)
            for i in range(7):
                n = i + 1
                p_n = np.mean(n_completed >= n)
                name = prefix + f'p_{n}'
                log_data[name] = p_n

        return log_data
