import imageio
from matplotlib import pyplot as plt

from ARCHIVE.get_agent import get_agent
from ARCHIVE.logging_info import log


class Trace(object):
    def __init__(self, trace_idx, trajectory_length):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.reward_sum = 0
        self.length = 0
        self.states = {}
        self.trajectory_length = trajectory_length
        self.a2_trajectories = []
        self.a1_trajectory_indexes = []
        self.idx = trace_idx
        self.a2_rewards = []
        self.disagreement_indexes = []
        self.disagreement_trajectories = []
        self.action_values = []

    def update(self, state_object, obs, a, q_vals, r, done, infos):
        self.obs.append(obs)
        self.rewards.append(r)
        self.dones.append(done)
        self.infos.append(infos)
        self.actions.append(a)
        self.reward_sum += r
        self.states[(self.idx,self.length)] = state_object
        self.length += 1
        self.action_values.append(q_vals)

    def get_trajectory(self, state):
        lower, upper = self.get_relevant_idx_range(state.id[0], len(self.states), self.trajectory_length)
        return self.states[lower:upper]

    def get_frames(self, s1_indexes, s2_indexes, s2_traj, mark_position=None):
        a1_frames = [self.states[x].image for x in s1_indexes]
        a2_frames = [self.a2_trajectories[s2_traj][x - min(s2_indexes)].image for x in s2_indexes]
        assert len(a1_frames) == self.trajectory_length, 'Error in highlight frame length'
        assert len(a2_frames) == self.trajectory_length, 'Error in highlight frame length'
        da_index = self.trajectory_length // 2 - 1
        if mark_position:
            for i in range(da_index - 1, da_index + 3):
                a1_frames[i] = mark_agent(a1_frames[i], position=mark_position)
                a2_frames[i] = a1_frames[i]
        return a1_frames, a2_frames

    def get_relevant_idx_range(self, indx, lst_len, range_len, overlay=0):
        if indx - range_len < 0:
            lb = 0
            ub = range_len - 1 - overlay
        elif indx + range_len > lst_len:
            ub = lst_len - 1
            lb = lst_len - 1 - range_len + overlay
        else:
            lb = indx - int(range_len / 2) + overlay
            ub = indx + int(range_len / 2) - overlay
        return lb, ub


class State(object):
    def __init__(self, idx, episode, obs, action_values, img, **kwargs):
        self.observation = obs
        self.image = img
        self.action_values = action_values
        self.id = (episode, idx)
        self.kwargs = kwargs

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def save_image(self, path, name):
        imageio.imwrite(path + '/' + name + '.png', self.image)


def get_execution_traces(args):
    """Obtain traces"""
    env, agent = get_agent(args.config, args.env_id)
    execution_traces = []
    for i in range(args.num_traces):
        log(f"\tRunning trace {i}...", args.verbose)
        get_single_trace(env, agent, i, execution_traces, args)
        log(f"\t\t... done", args.verbose)
    return execution_traces


def get_single_trace(env, agent, trace_idx, traces, args):
    """Implement a single trace while using the Trace and State classes"""
    trace = Trace(trace_idx, args.horizon)
    curr_obs = env.reset()
    agent.previous_state = curr_obs
    done, t, r = False, 0, 0
    action_values = agent.get_state_action_values(curr_obs)
    frame = env.render(mode='rgb_array')
    state = State(t, trace_idx, curr_obs, action_values, frame)
    action = agent.act(curr_obs)
    trace.update(state, curr_obs, action, action_values, r, done, {})
    while not done:
        curr_obs, r, done, infos = env.step(action)
        t += 1
        action_values = agent.get_state_action_values(curr_obs)
        frame = env.render(mode='rgb_array')
        state = State(t, trace_idx, curr_obs, action_values, frame)
        action = agent.act(curr_obs)
        trace.update(state, curr_obs, action, action_values, r, done, {})

    traces.append(trace)
