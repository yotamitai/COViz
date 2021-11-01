import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from contrastive_highlights.Interfaces.frogger_interface import FroggerInterface
from contrastive_highlights.Interfaces.gym_interface import GymInterface




def get_agent(args):
    """Implement here for specific agent and environment loading scheme"""
    if args.agent_type == "gym":
        interface = GymInterface(args.config, args.output_dir)
    elif args.agent_type == "frogger":
        interface = FroggerInterface(args.agent_config, args.output_dir, args.n_traces)
    env, agent = interface.initiate()
    agent.interface = interface
    env.seed(0)
    return env, agent

#
# def get_frogger_agent(config_file, output_dir, num_episodes, seed=0):
#     config = EnvironmentConfiguration.load_json(config_file)
#     config.num_episodes = num_episodes
#     agent_rng = np.random.RandomState(seed)
#     helper = create_helper(config)
#     config.save_json(join(output_dir, 'config.json'))
#     helper.save_state_features(join(output_dir, 'state_features.csv'))
#     env_id = '{}-{}-v0'.format(config.gym_env_id, 0)
#     helper.register_gym_environment(env_id, False, config.fps, config.show_score_bar)
#     env = gym.make(env_id)
#     video_callable = (lambda e: True)
#     exploration_strategy = GreedyExploration(config.min_temp, agent_rng)
#     agent = QValueBasedAgent(config.num_states, config.num_actions,
#                              action_names=config.get_action_names(),
#                              exploration_strategy=exploration_strategy)
#     agent_dir = config.load_dir
#     agent.load(agent_dir)
#     behavior_tracker = BehaviorTracker(config.num_episodes)
#     return env, agent, {"helper": helper,
#                         "behavior_tracker": behavior_tracker,
#                         "video_callable": video_callable}
#
#
# def get_gym_agent(args, log_dir):
#     # Going through custom gym packages to let them register in the global registory
#     for env_module in args.gym_packages:
#         importlib.import_module(env_module)
#
#     env_id = args.env
#     algo = args.algo
#     folder = args.folder
#
#     if args.exp_id == 0:
#         args.exp_id = get_latest_run_id(join(folder, algo), env_id)
#         print(f"Loading latest experiment, id={args.exp_id}")
#
#     # Sanity checks
#     if args.exp_id > 0:
#         log_path = join(folder, algo, f"{env_id}_{args.exp_id}")
#     else:
#         log_path = join(folder, algo)
#
#     assert isdir(log_path), f"The {log_path} folder was not found"
#
#     found = False
#     for ext in ["zip"]:
#         model_path = join(log_path, f"{env_id}.{ext}")
#         found = isfile(model_path)
#         if found:
#             break
#
#     if args.load_best:
#         model_path = join(log_path, "best_model.zip")
#         found = isfile(model_path)
#
#     if args.load_checkpoint is not None:
#         model_path = join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
#         found = isfile(model_path)
#
#     if args.load_last_checkpoint:
#         checkpoints = glob.glob(join(log_path, "rl_model_*_steps.zip"))
#         if len(checkpoints) == 0:
#             raise ValueError(f"No checkpoint found for {algo} on {env_id}, path: {log_path}")
#
#         def step_count(checkpoint_path: str) -> int:
#             # path follow the pattern "rl_model_*_steps.zip", we count from the back to ignore any other _ in the path
#             return int(checkpoint_path.split("_")[-2])
#
#         checkpoints = sorted(checkpoints, key=step_count)
#         model_path = checkpoints[-1]
#         found = True
#
#     if not found:
#         raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")
#
#     print(f"Loading {model_path}")
#
#     # Off-policy algorithm only support one env for now
#     off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
#
#     if algo in off_policy_algos:
#         args.n_envs = 1
#
#     set_random_seed(args.seed)
#
#     if args.num_threads > 0:
#         if args.verbose > 1:
#             print(f"Setting torch.num_threads to {args.num_threads}")
#         th.set_num_threads(args.num_threads)
#
#     args.is_atari = ExperimentManager.is_atari(env_id)
#
#     stats_path = join(log_path, env_id)
#     hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward,
#                                                     test_mode=True)
#
#     # load env_kwargs if existing
#     env_kwargs = {}
#     args_path = join(log_path, env_id, "args.yml")
#     if isfile(args_path):
#         with open(args_path, "r") as f:
#             loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
#             if loaded_args["env_kwargs"] is not None:
#                 env_kwargs = loaded_args["env_kwargs"]
#     # overwrite with command line arguments
#     if args.env_kwargs is not None:
#         env_kwargs.update(args.env_kwargs)
#
#     env = create_test_env(
#         env_id,
#         n_envs=args.n_envs,
#         stats_path=stats_path,
#         seed=args.seed,
#         log_dir=log_dir,
#         should_render=not args.no_render,
#         hyperparams=hyperparams,
#         env_kwargs=env_kwargs,
#     )
#
#     kwargs = dict(seed=args.seed)
#     if algo in off_policy_algos:
#         # Dummy buffer size as we don't need memory to enjoy the trained agent
#         kwargs.update(dict(buffer_size=1))
#
#     # Check if we are running python 3.8+
#     # we need to patch saved model under python 3.6/3.7 to load them
#     newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
#
#     custom_objects = {}
#     if newer_python_version:
#         custom_objects = {
#             "learning_rate": 0.0,
#             "lr_schedule": lambda _: 0.0,
#             "clip_range": lambda _: 0.0,
#         }
#
#     model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
#     return env, model
#
#
# def get_state_action_values(agent, state, agent_type):
#     if agent_type == "gym":
#         policy = agent.policy
#         observation, vectorized_env = policy.obs_to_tensor(state)
#         latent_pi, _, latent_sde = policy._get_latent(observation)
#         distribution = policy._get_action_dist_from_latent(latent_pi, latent_sde)
#         return np.array(distribution.distribution.probs.tolist()[0])
#     elif agent_type == 'frogger':
#         return agent.q[state] #TODO change to probability instead of q value?
#
#
# def get_state_from_obs(agent, obs, agent_type, params=None):
#     if agent_type == "gym":
#         return obs
#     elif agent_type == 'frogger':
#         return agent.agent_args["helper"].get_state_from_observation(obs, params[0], params[1])
#
# def get_next_action(agent, obs, state, agent_type):
#     if agent_type == "gym":
#         a, _ = agent.predict(obs, state=state, deterministic=True)
#     elif agent_type == 'frogger':
#         a = agent.act(state)
#     return a