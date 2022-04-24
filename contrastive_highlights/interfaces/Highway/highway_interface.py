import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw
from copy import deepcopy, copy
from matplotlib import pyplot as plt
import argparse

from os.path import abspath, join
from pathlib import Path

import gym

from contrastive_highlights.common import Trace
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.factory import agent_factory

from contrastive_highlights.interfaces.abstract_interface import AbstractInterface
from rl_agents.trainer.evaluation import Evaluation
import contrastive_highlights.interfaces.Highway
import multi_head.highway_env_local.envs.highway_env_local
from multi_head.DQNAgent_local_files.pytorch_local import DQNAgent


# ACTION_DICT = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}

class MyEvaluation(Evaluation):
    def __init__(self, env, agent, output_dir='../agents', num_episodes=1000, display_env=False):
        self.OUTPUT_FOLDER = output_dir
        super(MyEvaluation, self).__init__(env, agent, num_episodes=num_episodes,
                                           display_env=display_env)


agent_position = [164, 66]  # static in this domain


class HighwayInterface(AbstractInterface):
    def __init__(self, config, output_dir, load_path):
        super().__init__(config, output_dir)
        self.multi_head = None
        self.load_path = load_path

    def initiate(self, seed=0, evaluation_reset=False):
        config, output_dir = self.config, self.output_dir
        env_config, agent_config = config['env'], config['agent']
        env = gym.make(env_config["id"])
        env.seed(seed)
        env_config.update({"simulation_frequency": 15, "policy_frequency": 5, })
        env.configure(env_config)
        env.define_spaces()
        agent = agent_factory(env, agent_config)
        agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
        if evaluation_reset:
            evaluation_reset.training = False
            evaluation_reset.close()
        return env, agent

    def evaluation(self, env, agent):
        evaluation = MyEvaluation(env, agent, display_env=False, output_dir=self.output_dir)
        agent_path = Path(join(self.load_path, 'checkpoint-final.tar'))
        evaluation.load_agent_model(agent_path)
        return evaluation

    def get_state_action_values(self, agent, state):
        action_values = agent.get_state_action_values(state)
        return action_values if action_values.ndim == 1 else action_values[0][0]

    def get_state_RD_action_values(self, agent, state):
        return agent.get_state_action_values(state)[1:]

    def get_state_from_obs(self, agent, obs, params=None):
        return obs

    def get_next_action(self, agent, obs, state):
        return agent.act(state)

    def get_features(self, env):
        return {"position": copy(env.road.vehicles[0].destination)}

    def contrastive_trace(self, trace_idx, k_steps, params=None):
        return HighwayTrace(trace_idx, k_steps)

    def pre_contrastive(self, env):
        return deepcopy(env)

    def post_contrastive(self, agent1, agent2, pre_params=None):
        env = pre_params
        agent2.previous_state = agent1.previous_state
        return env


class HighwayTrace(Trace):
    def __init__(self, trace_idx, k_steps):
        super().__init__(trace_idx, k_steps)
        self.contrastive = []

    def update(self, state_object, obs, a, r, done, infos, params=None):
        self.obs.append(obs)
        self.rewards.append(r)
        self.dones.append(done)
        self.infos.append(infos)
        self.previous_actions.append(a)
        self.reward_sum += r
        self.states.append(state_object)
        self.length += 1

    def mark_frames(self, hl_idx, indexes, color=255, thickness=2):
        """
        mark the contrastive frames
        highway static agent position = [164, 66]
        """
        frames, rel_idx = [], 0
        """the common trajectory"""
        for state in range(indexes[0], hl_idx):
            frames.append(self.states[state].image)

        """the contrastive state"""
        rel_idx = len(frames)
        marked_frame = copy(self.states[hl_idx].image)
        top_left = (164, 66)
        bottom_right = (194, 81)
        marked_frame = np.ascontiguousarray(marked_frame, dtype=np.uint8)
        cv2.rectangle(marked_frame, top_left, bottom_right, color, thickness)
        # add text
        font = ImageFont.truetype('Roboto-Regular.ttf', 20)
        image = Image.fromarray(marked_frame, 'RGB')
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "Contrastive State", (0), font=font)
        new_marked_frame = np.asarray(image)
        frames.append(new_marked_frame)

        """the contrastive trajectory"""
        ref_pos = [164, 66]
        rel_pos = self.get_relative_position(indexes, hl_idx)
        for i in range(indexes[-1] - hl_idx):
            marked_frame = copy(self.states[hl_idx + i + 1].image)
            add_x, add_y = int(rel_pos[i][1] * 5), int(rel_pos[i][0] * 10)
            top_left = (ref_pos[0] + add_y, ref_pos[1] + add_x)
            bottom_right = (ref_pos[0] + 30 + add_y, ref_pos[1] + 15 + add_x)
            marked_frame = np.ascontiguousarray(marked_frame, dtype=np.uint8)
            cv2.rectangle(marked_frame, top_left, bottom_right, color, thickness)
            # cv2.rectangle(marked_frame, (ref_pos[0] + add_y + 4, ref_pos[1] + add_x + 4),
            #               (ref_pos[0] + 30 + add_y - 4, ref_pos[1] + 15 + add_x - 4),
            #               (43, 165, 0, 255), thickness)
            ## for debugging
            # plt.imshow(marked_frame)
            # plt.show()
            frames.append(marked_frame)
        return frames, rel_idx

    def mark_contrastive_state(self, hl, color=255, thickness=2):
        img2 = copy(self.states[hl[1]].image)
        # static_position = [164, 66]
        top_left = (164, 66)
        bottom_right = (194, 81)
        cv2.rectangle(img2, top_left, bottom_right, color, thickness)
        return img2

    def get_relative_position(self, trajectory, hl_idx):
        a1_obs = np.array(
            [x.features["position"] for x in self.states[hl_idx + 1:trajectory[-1] + 1]])
        a2_obs = [x.features["position"] for x in self.contrastive[hl_idx].states if
                  x.id[1] > hl_idx and x.id[1] <= self.contrastive[hl_idx].traj_end_state]
        assert len(a1_obs) == len(a2_obs), "Error with trajectory lengths"
        rel_cords = np.around(a2_obs - a1_obs, 3)
        return rel_cords


def highway_config(args):
    """highway"""
    args.config_filename = "metadata"
    """Highlight parameters"""
    args.config_changes = {"env": {}, "agent": {}}
    args.data_name = ''
    args.name = "rightLaneChangeLane"
    # args.name = "Plain_old"
    args.load_path = abspath(f'../agents/{args.interface}/{args.name}')
    args.n_traces = 2
    args.k_steps = 7
    args.overlay = args.k_steps // 2
    return args
