import argparse
from os.path import join

import pandas as pd

from contrastive_highlights.highlights_state_selection import compute_states_importance, \
    summary_states_by_single_state
from contrastive_highlights.logging_info import get_logging, log
from contrastive_highlights.traces import get_execution_traces
from contrastive_highlights.trajectories import states_to_trajectories
from contrastive_highlights.utils import load_traces, save_traces


def get_states(traces):
    states = {}
    for t in traces:
        states.update(t.states)
    data = {
        'state': list(states.keys()),
        'q_values': [x.action_values for x in states.values()]
    }
    return pd.DataFrame(data)


def main(args):
    name, file_name = get_logging(args)
    # Obtain execution traces
    if args.traces_path:
        traces = load_traces(args.traces_path)
        log(f'Loaded traces', args.verbose)
    else:
        traces = get_execution_traces(args)
        log(f"Execution traces generated", args.verbose)

    # save traces
    output_dir = join(args.results_dir, file_name)
    save_traces(traces, output_dir)
    log(f'Saved traces to: {output_dir}', args.verbose)
    # Obtain states
    df = get_states(traces)
    # calculate importance of states
    # TODO importance by importance function
    # implement user selection importance
    state_importance_df = compute_states_importance(df, compare_to=args.state_importance)
    # state_importance_dict = dict(zip(state_importance_df["state"], state_importance_df["importance"]))

    # select summary states
    summary_states = summary_states_by_single_state(state_importance_df, traces,
                                                    args.n_highlights, args.horizon,
                                                    args.overlay_limit)
    # TODO implement summary_states_by_trajectory()

    # TODO select method for choosing counterfactual actions
    cf_method = "importance"
    # get trajectories
    states_to_trajectories(cf_method, summary_states, traces)

    # get summary states
    """get highlights"""
    if args.trajectory_importance == "single_state":
        """highlights importance by single state importance"""
        summary_states = highlights(df, traces, args.num_trajectories, args.trajectory_length,
                                    args.minimum_gap, args.overlay_limit)

        all_trajectories = states_to_trajectories(summary_states, state_importance_dict)
        summary_trajectories = all_trajectories
    else:
        """highlights importance by trajectory"""
        all_trajectories, summary_trajectories = \
            trajectories_by_importance(traces, state_importance_dict, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent Comparisons')
    parser.add_argument('-env', '--env_id', help='environment name', default="highway_local-v0")
    parser.add_argument('-a', '--name', help='agent name', type=str, default="Agent-1")
    parser.add_argument('-n', '--num_traces', help='number of episodes to run', type=int,
                        default=3)
    parser.add_argument('-fps', '--fps', help='summary video fps', type=int, default=1)
    parser.add_argument('-l', '--horizon', help='number of frames to show per highlight',
                        type=int, default=10)
    parser.add_argument('-sb', '--show_score_bar', help='score bar', type=bool, default=False)
    parser.add_argument('-rand', '--randomized', help='randomize order of summary trajectories',
                        type=bool, default=True)
    parser.add_argument('-k', '--n_highlights', help='# of disagreements in the summary',
                        type=int, default=5)
    parser.add_argument('-overlaplim', '--overlay_limit',
                        help='# of allowed overlapping states between trajectories', type=int,
                        default=3)
    parser.add_argument('-impMeth', '--importance_type',
                        help='importance by state or trajectory', default='trajectory')
    parser.add_argument('-impTraj', '--trajectory_importance',
                        help='method calculating trajectory importance', default='last_state')
    parser.add_argument('-impState', '--state_importance',
                        help='method calculating state importance', default='second')
    parser.add_argument('-v', '--verbose', help='print information to the console', default=False)
    parser.add_argument('-ass', '--agent_assessment', help='apply agent ratio by agent score',
                        default=False)
    parser.add_argument('-se', '--seed', help='environment seed', default=0)
    parser.add_argument('-res', '--results_dir', help='results directory', default='results')
    parser.add_argument('-tr', '--traces_path', help='path to traces file if exists',
                        default=None)
    args = parser.parse_args()

    """experiment parameters"""
    args.config = {
        "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
        "path": '../agents/DQN_1000ep/checkpoint-final.tar',
    }
    args.name = args.config["path"].split('/')[2]
    args.num_traces = 1
    args.traces_path = '/home/yotama/OneDrive/Local_Git/Contrastive_Highlights/contrastive_highlights/results/2021-07-14_12:08:00_DQN_1000ep'
    main(args)
