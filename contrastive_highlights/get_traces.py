from os.path import join


from utils import Trace, State


def get_traces(environment, agent, args):
    """Obtain traces and state dictionary"""
    execution_traces, states_dictionary = [], {}
    for i in range(args.n_traces):
        get_single_trace(environment, agent, i, execution_traces, states_dictionary, args)
        if args.verbose: print(f"\tTrace {i} {15*'-'+'>'} Obtained")
    if args.verbose: print(f"Highlights {15*'-'+'>'} Traces & States Generated")
    return execution_traces, states_dictionary


def get_single_trace(env, agent, trace_idx, agent_traces, states_dict, args):
    """Implement a single trace while using the Trace and State classes"""
    trace = Trace()
    # ********* Implement here *****************
    curr_obs = env.reset()
    done = False
    while not done:
        a = agent.act(curr_obs)
        obs, r, done, infos = env.step(a)


        """Generate State"""
        state_img = env.render(mode='rgb_array')
        state_q_values = agent.get_state_action_values(obs)
        features = NotImplemented #TODO implement here
        state_id = (trace_idx, trace.length)
        states_dict[state_id] = State(state_id, obs, state_q_values, features, state_img)
        """Add step and state to trace"""
        trace.update(obs, r, done, infos, a, state_id)



    agent_traces.append(trace)
