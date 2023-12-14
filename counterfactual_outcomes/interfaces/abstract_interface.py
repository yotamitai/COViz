import imageio
from matplotlib import pyplot as plt


class AbstractInterface(object):
    def __init__(self, config, output_dir):
        self.output_dir = output_dir
        self.config = config

    def initiate(self):
        return

    def get_state_action_values(self, agent, state):
        return

    def get_state_from_obs(self, agent, obs, params=None):
        return

    def get_next_action(self, agent, obs, state):
        return

    def get_features(self, env):
        return

    def contrastive_trace(self, trace_idx, k_steps, params=None):
        return

    def pre_contrastive(self, env):
        return

    def post_contrastive(self, agent1, agent2, pre_params=None):
        return


