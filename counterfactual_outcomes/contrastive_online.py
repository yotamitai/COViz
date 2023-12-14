from counterfactual_outcomes.common import log_msg
from counterfactual_outcomes.common import State


class ContrastiveTrajectory(object):
    def __init__(self, state_id, k_steps, trace):
        self.importance = 0
        self.k_steps = k_steps
        self.id = state_id
        self.start_idx = (state_id[1] - k_steps) if (state_id[1] - k_steps) >= 0 else 0
        self.rewards = []
        self.states = trace.states[self.start_idx:]
        self.actions = []

    def update(self, state_obj, r, action):
        self.states.append(state_obj)
        self.rewards.append(r)
        self.actions.append(action)

    def get_contrastive_trajectory(self, env, agent, state_id, contra_action, contra_counter):
        action = contra_action
        for step in range(state_id[1] + 1, state_id[1] + self.k_steps + 1):
            obs, r, done, info = env.step(action)
            contra_counter -= 1  # reduce contra counter
            s = agent.interface.get_state_from_obs(agent, obs)
            s_a_values = agent.interface.get_state_action_values(agent, s)
            frame = env.render(mode='rgb_array')
            features = agent.interface.get_features(env)
            contra_state_id = (state_id[0], step)
            state_obj = State(contra_state_id, obs, s, s_a_values, frame, features)
            self.update(state_obj, r, action)
            if done: break
            if contra_counter > 0: continue
            action = agent.interface.get_next_action(agent, obs, s)


def get_contrastive_trajectory(state_id, trace, env, agent, contra_action, k_steps,
                               contra_counter):
    traj = ContrastiveTrajectory(state_id, k_steps, trace)
    traj.get_contrastive_trajectory(env, agent, state_id, contra_action, contra_counter)
    return traj


def online_comparison(env1, agent1, env2, agent2, args, evaluation1=None, evaluation2=None):
    """
    get all contrastive trajectories a given agent
    """
    """Run"""
    traces = []
    for n in range(args.n_traces):
        log_msg(f'Executing Trace number: {n}', args.verbose)
        trace = agent1.interface.contrastive_trace(n, args.k_steps)
        """initial state"""
        obs, _ = env1.reset(), env2.reset()
        assert obs.tolist() == _.tolist(), f'Nonidentical environment'
        step, r, done, infos, agent1_a = 0, 0, False, {}, None
        agent1.previous_state = agent2.previous_state = obs  # required

        # for _ in range(30):  # TODO remove
        while not done:
            log_msg(f'\ttime-step number: {step}', args.verbose)
            state = agent1.interface.get_state_from_obs(agent1, obs, [r, done])
            s_a_values = agent1.interface.get_state_action_values(agent1, state)
            state_id, frame = (n, step), env1.render(mode='rgb_array')
            features = agent1.interface.get_features(env1)
            state_obj = State(state_id, obs, state, s_a_values, frame, features)
            trace.update(state_obj, obs, agent1_a, r, done, infos, params=[s_a_values])
            """actions"""
            agent1_a = agent1.interface.get_next_action(agent1, obs, state) if not done else None
            agent2_a = sorted(list(enumerate(s_a_values)), key=lambda x: x[1])[-2][0]
            """contrastive trajectory"""
            pre_vars = agent2.interface.pre_contrastive(env1)
            trace.contrastive.append(
                get_contrastive_trajectory(state_id, trace, env2, agent2, agent2_a, args.k_steps,
                                           args.contra_action_counter))
            """return agent 2 environment to the current state"""
            env2 = agent2.interface.post_contrastive(agent1, agent2, pre_vars)
            """Transition both agent's based on agent 1 action"""
            step += 1
            obs, r, done, info = env1.step(agent1_a)
            if done: break
            _ = env2.step(agent1_a)  # dont need returned values
            assert obs.tolist() == _[0].tolist(), f'Nonidentical environment transition'

        """end of episode"""
        traces.append(trace)
    return traces
