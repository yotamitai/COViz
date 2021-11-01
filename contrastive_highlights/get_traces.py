from os.path import join

import numpy as np

from common.utils import Trace, State
from contrastive_highlights.contrastive import enact_contrastive


def get_traces(environment, agent, args):
    """Obtain traces and state dictionary"""
    execution_traces, states_dictionary = [], {}
    for i in range(args.n_traces):
        trace = get_single_trace(environment, agent, i, states_dictionary)
        execution_traces.append(trace)
        if args.verbose: print(f"\tTrace {i} {15 * '-' + '>'} Obtained")
    if args.verbose: print(f"Highlights {15 * '-' + '>'} Traces & States Generated")
    return execution_traces, states_dictionary


def get_single_trace(env, agent, trace_idx, states_dict):
    """Execute a single trace while using the Trace and State classes"""
    trace = Trace()
    obs = env.reset()
    done, state = False, None
    """update initial state"""
    state_id, state_img = (trace_idx, 0), env.render(mode='rgb_array')
    state = agent.interface.get_state_from_obs(agent, obs, [0, None])
    trace.update(obs, 0, done, 0, None, state_id)
    state_action_values = agent.interface.get_state_action_values(agent, state)
    states_dict[state_id] = State(state_id, obs, state, state_action_values, None, state_img)
    while not done:
        a = agent.interface.get_next_action(agent, obs, state)
        obs, r, done, infos = env.step(a)
        """Generate State"""
        state = agent.interface.get_state_from_obs(agent, obs, [r, done])
        state_img = env.render(mode='rgb_array')
        state_q_values = agent.interface.get_state_action_values(agent, state)
        features = None
        state_id = (trace_idx, trace.length)
        states_dict[state_id] = State(state_id, obs, state, state_q_values, features, state_img)
        """Add step and state to trace"""
        trace.update(obs, r, done, infos, a, state_id)
    return trace

