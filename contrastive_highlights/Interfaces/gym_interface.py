import importlib
import sys
from os.path import join, isdir, isfile

import numpy as np
import torch as th
import glob

import yaml
from stable_baselines3.common.utils import set_random_seed

from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams

from contrastive_highlights.Interfaces.abstract_interface import AbstractInterface
from utils.exp_manager import ExperimentManager


class GymInterface(AbstractInterface):
    def __init__(self, config, output_dir, ):
        super().__init__(config, output_dir, )

    def initiate(self):
        config, output_dir = self.config, self.output_dir
        # Going through custom gym packages to let them register in the global registory
        for env_module in config.gym_packages:
            importlib.import_module(env_module)

        env_id = config.env
        algo = config.algo
        folder = config.folder

        if config.exp_id == 0:
            config.exp_id = get_latest_run_id(join(folder, algo), env_id)
            print(f"Loading latest experiment, id={config.exp_id}")

        # Sanity checks
        if config.exp_id > 0:
            log_path = join(folder, algo, f"{env_id}_{config.exp_id}")
        else:
            log_path = join(folder, algo)

        assert isdir(log_path), f"The {log_path} folder was not found"

        found = False
        for ext in ["zip"]:
            model_path = join(log_path, f"{env_id}.{ext}")
            found = isfile(model_path)
            if found:
                break

        if config.load_best:
            model_path = join(log_path, "best_model.zip")
            found = isfile(model_path)

        if config.load_checkpoint is not None:
            model_path = join(log_path, f"rl_model_{config.load_checkpoint}_steps.zip")
            found = isfile(model_path)

        if config.load_last_checkpoint:
            checkpoints = glob.glob(join(log_path, "rl_model_*_steps.zip"))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoint found for {algo} on {env_id}, path: {log_path}")

            def step_count(checkpoint_path: str) -> int:
                # path follow the pattern "rl_model_*_steps.zip", we count from the back to ignore any other _ in the path
                return int(checkpoint_path.split("_")[-2])

            checkpoints = sorted(checkpoints, key=step_count)
            model_path = checkpoints[-1]
            found = True

        if not found:
            raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

        print(f"Loading {model_path}")

        # Off-policy algorithm only support one env for now
        off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

        if algo in off_policy_algos:
            config.n_envs = 1

        set_random_seed(config.seed)

        if config.num_threads > 0:
            if config.verbose > 1:
                print(f"Setting torch.num_threads to {config.num_threads}")
            th.set_num_threads(config.num_threads)

        config.is_atari = ExperimentManager.is_atari(env_id)

        stats_path = join(log_path, env_id)
        hyperparams, stats_path = get_saved_hyperparams(stats_path,
                                                        norm_reward=config.norm_reward,
                                                        test_mode=True)

        # load env_kwargs if existing
        env_kwargs = {}
        args_path = join(log_path, env_id, "args.yml")
        if isfile(args_path):
            with open(args_path, "r") as f:
                loaded_args = yaml.load(f,
                                        Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
                if loaded_args["env_kwargs"] is not None:
                    env_kwargs = loaded_args["env_kwargs"]
        # overwrite with command line arguments
        if config.env_kwargs is not None:
            env_kwargs.update(config.env_kwargs)

        env = create_test_env(
            env_id,
            n_envs=config.n_envs,
            stats_path=stats_path,
            seed=config.seed,
            log_dir=output_dir,
            should_render=not config.no_render,
            hyperparams=hyperparams,
            env_kwargs=env_kwargs,
        )

        kwargs = dict(seed=config.seed)
        if algo in off_policy_algos:
            # Dummy buffer size as we don't need memory to enjoy the trained agent
            kwargs.update(dict(buffer_size=1))

        # Check if we are running python 3.8+
        # we need to patch saved model under python 3.6/3.7 to load them
        newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

        custom_objects = {}
        if newer_python_version:
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }

        model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
        return env, model

    def get_state_action_values(self, agent, state):
        policy = agent.policy
        observation, vectorized_env = policy.obs_to_tensor(state)
        latent_pi, _, latent_sde = policy._get_latent(observation)
        distribution = policy._get_action_dist_from_latent(latent_pi, latent_sde)
        return np.array(distribution.distribution.probs.tolist()[0])

    def get_state_from_obs(self, agent, obs, params=None):
        return obs

    def get_next_action(self, agent, obs, state):
        a, _ = agent.predict(obs, state=state, deterministic=True)
        return a

    def get_features(self, env):
        return
