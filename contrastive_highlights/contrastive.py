import gym
import numpy as np

from contrastive_highlights.common.utils import State
from get_agent import get_agent


class ContrastiveHighlight(object):
    def __init__(self, original, contrastive):
        self.original_trajectory = original
        self.contrastive_trajectory = contrastive

def enact_contrastive(method):
    print()


def get_contrastive_highlights(traces, states, summary_trajectories, args):
    """get contrastive highlights from """

    """get the traces needed from the summary states"""

    """sort important states by relative trace"""

    """run the selected traces"""
    # env, agent = get_agent(args)
    # for t in traces:
    #     obs = env.reset()
    #     assert np.array_equiv(obs, t.obs[0])

    contrastive_highlights = []
    for trace_idx, highlight_idx in summary_trajectories:
        env, agent = get_agent(args)
        env.args = args
        """reset env to the desired trace configuration"""
        highlight_obs = states[(trace_idx, highlight_idx)].observation
        contrastive_trajectory = get_contrastive_trace(env, agent, traces[trace_idx], trace_idx,
                                                       highlight_idx, highlight_obs, args)
        original_trajectory = [states[(trace_idx, (highlight_idx - args.trajectory_length // 2 + 1) + x)] for x
                               in range(args.trajectory_length)]
        contrastive_highlights.append(ContrastiveHighlight(original_trajectory, contrastive_trajectory))

        env.close()
        del gym.envs.registration.registry.env_specs[args.config.env]

        if args.verbose: print(f"HIGHLIGHTS {15 * '-' + '>'} Traces Generated")

    """in each trace stop at the relevant important state"""

    """split off for a set number of steps"""
    """     use inverse action to constrain agent"""

    """revert back to important state and continue (unless this is the last one)"""

    return contrastive_highlights


def get_contrastive_trace(env, agent, trace, trace_idx, highlight_state_idx, highlight_obs, args):
    states_dict = {}
    [env.reset() for _ in range(trace_idx)]
    obs = env.reset()
    done, state, idx = False, None, 1
    while not done:
        if idx < highlight_state_idx:
            a = trace.actions[idx]
            obs, r, done, infos = env.step(a)
            assert np.array_equiv(obs, trace.obs[idx]), "Unmatched trace"
            if idx == highlight_state_idx: break  # Contrastive
            idx += 1
        elif idx <= highlight_state_idx + (args.trajectory_length // 2):
            """get contrastive sequence"""
            idx = enact_contrastive() #TODO
            """resume trace"""
            a, state = agent.predict(obs, state=state, deterministic=True)
            obs, r, done, infos = env.step(a)
            """Generate State"""
            state_img = env.render(mode='rgb_array')
            state_q_values = agent.interface.get_state_action_values(agent, obs)
            features = None
            state_id = (trace_idx, idx)
            states_dict[state_id] = State(state_id, obs, state_q_values, features, state_img)
        else:
            break

    return states_dict