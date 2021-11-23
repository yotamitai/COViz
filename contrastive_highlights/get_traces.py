from os.path import join

import numpy as np

from common.utils import Trace, State
from contrastive_highlights.contrastive import enact_contrastive


def get_traces(environment, agent, args):
    """Obtain traces and state dictionary"""
    execution_traces, states_dictionary = [], {}
    if args.verbose: print(f"Obtaining Execution Traces:")
    for i in range(args.n_traces):
        trace = get_single_trace(environment, agent, i, states_dictionary)
        execution_traces.append(trace)
        if args.verbose: print(f"\tTrace {i} {15 * '-' + '>'} Obtained")

    # if args.verbose: print(f"Highlights {15 * '-' + '>'} Traces & States Generated")
    return execution_traces, states_dictionary


def get_single_trace(env, agent, trace_idx, states_dict):
    """Execute a single trace while using the Trace and State classes"""
    trace = Trace()
    done, r, infos = False, 0, {}
    """update initial state"""
    obs = env.reset()
    state = agent.interface.get_state_from_obs(agent, obs, [r, done])
    state_action_values = agent.interface.get_state_action_values(agent, state)
    a = agent.interface.get_next_action(agent, obs, state) if not done else None
    state_id, state_img = (trace_idx, trace.length), env.render(mode='rgb_array')
    features = agent.interface.get_features(env)
    trace.update(obs, r, done, infos, a, state_id)
    states_dict[state_id] = State(state_id, obs, state, state_action_values, features, state_img)
    while not done:
        new_obs, r, done, infos = env.step(a)
        new_state = agent.interface.get_state_from_obs(agent, new_obs, [r, done])
        """Generate State"""
        state, obs = new_state, new_obs
        a = agent.interface.get_next_action(agent, obs, state) if not done else None
        state_action_values = agent.interface.get_state_action_values(agent, state)
        state_id, state_img = (trace_idx, trace.length), env.render(mode='rgb_array')
        features = agent.interface.get_features(env)
        trace.update(obs, r, done, infos, a, state_id)
        states_dict[state_id] = State(state_id, obs, state, state_action_values, features, state_img)
    return trace

