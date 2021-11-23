from os.path import join

import gym
import numpy as np

from contrastive_highlights.Interfaces.abstract_interface import AbstractInterface
from contrastive_highlights.common.frogger_explorations import GreedyExploration
from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios import EnvironmentConfiguration, create_helper


class FroggerInterface(AbstractInterface):
    def __init__(self, config, output_dir, num_episodes, seed=0):
        super().__init__(config, output_dir)
        self.num_episodes = num_episodes
        self.seed = seed


    def initiate(self):
        config_file, output_dir = self.config, self.output_dir
        config = EnvironmentConfiguration.load_json(config_file)
        config.num_episodes = self.num_episodes
        agent_rng = np.random.RandomState(self.seed)
        helper = create_helper(config)
        config.save_json(join(output_dir, 'config.json'))
        helper.save_state_features(join(output_dir, 'state_features.csv'))
        env_id = '{}-{}-v0'.format(config.gym_env_id, 0)
        helper.register_gym_environment(env_id, False, config.fps, config.show_score_bar)
        env = gym.make(env_id)
        video_callable = (lambda e: True)
        exploration_strategy = GreedyExploration(config.min_temp, agent_rng)
        agent = QValueBasedAgent(config.num_states, config.num_actions,
                                 action_names=config.get_action_names(),
                                 exploration_strategy=exploration_strategy)
        # helper.agent = agent
        agent_dir = config.load_dir
        agent.load(agent_dir)
        behavior_tracker = BehaviorTracker(config.num_episodes)
        agent_args = {"helper": helper, "behavior_tracker": behavior_tracker,
                      "video_callable": video_callable}
        agent.agent_args = agent_args
        return env, agent

    def get_state_action_values(self, agent, state):
        return agent.q[state] #TODO change to probability instead of q value?

    def get_state_from_obs(self, agent, obs, params=None):
        return agent.agent_args["helper"].get_state_from_observation(obs, params[0], params[1])

    def get_next_action(self, agent, obs, state):
        return agent.act(state)

    def get_features(self, env):
        position = [round(x) for x in env.env.game_state.game.frog.position]
        return {"position": position}

    # def update_interface(self, agent,trace_idx, step, old_obs, new_obs, r, done, a, prev_state, old_state, new_state):
    #     r = agent.agent_args['helper'].get_reward(prev_state, a, r, new_state, done)
    #     agent.update(old_state, a, r, new_state)
    #     agent.agent_args['helper'].update_stats(trace_idx, step, old_obs, new_obs, prev_state, a, r, new_state)
    #     return r