import gym
import numpy as np

from contrastive_highlights.common.utils import State
from get_agent import get_agent


class ContrastiveHighlight(object):
    def __init__(self, original, contrastive, trajectory_length, contrastive_state):
        self.trajectory_original = original
        self.trajectory_contrastive = contrastive
        self.trajectory_length = trajectory_length
        self.contrastive_state = contrastive_state
        self.frames_original, self.frames_contrastive = self.fit_frames_to_trajectory_length()

    def fit_frames_to_trajectory_length(self):
        """fit original and contrastive trajectories to length of trajectory_length
        """
        """before states"""
        missing_start = (self.trajectory_length // 2) - (self.contrastive_state[1] + 1)
        cs_idx = next((i for i, item in enumerate(self.trajectory_original)
                       if item.name == self.contrastive_state), -1)

        before_frames = [self.trajectory_original[0].image for _ in range(missing_start)] + \
                        [self.trajectory_original[i].image for i in range(cs_idx)]

        o_traj = before_frames + [x.image for x in self.trajectory_original[cs_idx:]]
        c_traj = before_frames + [x.image for x in self.trajectory_contrastive]

        o_missing_end = self.trajectory_length - len(o_traj)
        c_missing_end = self.trajectory_length - len(c_traj)

        o_after_frames = [self.trajectory_original[-1].image for _ in range(o_missing_end)]
        c_after_frames = [self.trajectory_contrastive[-1].image for _ in range(c_missing_end)]

        return o_traj + o_after_frames,  c_traj + c_after_frames


def enact_contrastive(action_values, method):
    if method == "second":
        # return index of second largest value
        return list(action_values).index(sorted(action_values)[-2])


def get_contrastive_highlights(traces, states, summary_trajectories, args):
    contrastive_highlights = []
    for t_idx, hl_idx in summary_trajectories:
        env, agent = get_agent(args)
        env.args = args
        """reset env to the desired trace configuration"""
        highlight_obs = states[(t_idx, hl_idx)].observation
        contrastive_trajectory = get_contrastive_trace(env, agent, traces[t_idx], t_idx,
                                                       hl_idx, highlight_obs, args)
        if args.verbose: print(
            f"Contrastive {15 * '-' + '>'} Highlight {(t_idx, hl_idx)} Generated")
        s_rng = max(hl_idx - (args.trajectory_length // 2) + 1, 0)
        e_rng = min(hl_idx + (args.trajectory_length // 2), len(traces[t_idx].states) - 1)
        original_trajectory = [states[(t_idx, x)] for x in range(s_rng, e_rng + 1)]
        contrastive_highlights.append(
            ContrastiveHighlight(original_trajectory, contrastive_trajectory,
                                 args.trajectory_length, (t_idx, hl_idx)))

        env.close()
        if args.agent_type == "frogger":
            del gym.envs.registration.registry.env_specs[env.spec.id]

    return contrastive_highlights


def get_contrastive_trace(env, agent, trace, trace_idx, highlight_state_idx, highlight_obs, args):
    states_list = []
    [env.reset() for _ in range(trace_idx)]
    obs = env.reset()
    r, done, state, i = 0, False, None, 0
    """reach highlight state"""
    for i in range(highlight_state_idx):  # because first action is None
        assert np.array_equiv(obs, trace.obs[i]), "Unmatched trace"
        a = trace.actions[i]
        obs, r, done, infos = env.step(a)
        #TODO state, obs = new_state, new_obs

    """contrastive state"""
    state = agent.interface.get_state_from_obs(agent, obs, [r, done])
    state_action_values = agent.interface.get_state_action_values(agent, state)
    a = enact_contrastive(state_action_values, args.contrastive_method)
    state_id, state_img = (trace_idx, trace.length), env.render(mode='rgb_array')
    features = agent.interface.get_features(env)
    states_list.append(State(state_id, obs, state, state_action_values, features, state_img))
    """start contrastive trajectory"""
    for idx in range(highlight_state_idx+1,
                     highlight_state_idx + (args.trajectory_length // 2) + 1):
        new_obs, r, done, infos = env.step(a)
        new_state = agent.interface.get_state_from_obs(agent, obs, [r, done])
        """Generate State"""
        state, obs = new_state, new_obs
        a = agent.interface.get_next_action(agent, obs, state)
        state_action_values = agent.interface.get_state_action_values(agent, state)
        state_id, state_img = (trace_idx, trace.length), env.render(mode='rgb_array')
        features = agent.interface.get_features(env)
        states_list.append(State(state_id, obs, state, state_action_values, features, state_img))
        if done: break

    return states_list



