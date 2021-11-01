import argparse
from datetime import datetime
import random

import gym
import pandas as pd
from os.path import join, basename, abspath

from get_agent import get_agent
from get_traces import get_traces
from highlights.utils import create_video, make_clean_dirs, pickle_save
from highlights_state_selection import compute_states_importance, highlights, highlights_div
from get_trajectories import states_to_trajectories, trajectories_by_importance, \
    get_trajectory_images
from ffmpeg import merge_and_fade


def get_highlights(args):
    args.output_dir = join(abspath('results'), '_'.join(
        [args.id, datetime.now().strftime("%H:%M:%S_%d-%m-%Y")]))
    make_clean_dirs(args.output_dir)

    env, agent = get_agent(args)
    traces, states = get_traces(env, agent, abspath('results'), args)

    """highlights algorithm"""
    data = {
        'state': list(states.keys()),
        'q_values': [x.observed_actions for x in states.values()]
    }
    q_values_df = pd.DataFrame(data)

    """importance by state"""
    highlights_df = compute_states_importance(q_values_df, compare_to=args.state_importance)
    state_importance_dict = dict(zip(highlights_df["state"], highlights_df["importance"]))

    """get highlights"""
    if args.trajectory_importance == "single_state":
        """highlights importance by single state importance"""
        summary_states = highlights(highlights_df, traces, args.num_trajectories,
                                    args.trajectory_length, args.minimum_gap, args.overlay_limit)
        # summary_states = highlights_div(highlights_df, traces, args.num_trajectories,
        #                             args.trajectory_length,
        #                             args.minimum_gap)
        all_trajectories = states_to_trajectories(summary_states, state_importance_dict)
        summary_trajectories = all_trajectories

    else:
        """highlights importance by trajectory"""
        all_trajectories, summary_trajectories = \
            trajectories_by_importance(traces, state_importance_dict, args)

    # random highlights
    # summary_trajectories = random.choices(all_trajectories, k=5)

    # random order
    if args.randomized: random.shuffle(summary_trajectories)

    """Save data used for this run"""
    pickle_save(traces, join(args.output_dir, 'Traces.pkl'))
    pickle_save(states, join(args.output_dir, 'States.pkl'))
    pickle_save(all_trajectories, join(args.output_dir, 'Trajectories.pkl'))

    """Save Highlight videos"""
    frames_dir = join(args.output_dir, 'Highlight_Frames')
    videos_dir = join(args.output_dir, "Highlight_Videos")
    height, width, layers = list(states.values())[0].image.shape
    img_size = (width, height)
    get_trajectory_images(summary_trajectories, states, frames_dir)
    create_video(frames_dir, videos_dir, args.num_trajectories, img_size, args.fps)
    if args.verbose: print(f"HIGHLIGHTS {15 * '-' + '>'} Videos Generated")

    """Merge Highlights to a single video with fade in/ fade out effects"""
    fade_out_frame = args.trajectory_length - args.fade_duration
    merge_and_fade(videos_dir, args.num_trajectories, fade_out_frame, args.fade_duration,
                   args.id)

    """Save data used for this run"""
    pickle_save(traces, join(args.output_dir, 'Traces.pkl'))
    pickle_save(states, join(args.output_dir, 'States.pkl'))
    pickle_save(all_trajectories, join(args.output_dir, 'Trajectories.pkl'))
    if args.verbose: print(f"Highlights {15 * '-' + '>'} Run Configurations Saved")

    env.close()
    # del gym.envs.registration.registry.env_specs[env.spec.id]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HIGHLIGHTS')
    parser.add_argument('-a', '--name', help='agent name', type=str, default="Agent-0")
    parser.add_argument('-num_ep', '--num_episodes', help='number of episodes to run', type=int,
                        default=1)
    parser.add_argument('-fps', '--fps', help='summary video fps', type=int, default=1)
    parser.add_argument('-n', '--n_traces', help='number of traces to obtain', type=int,
                        default=10)
    parser.add_argument('-k', '--num_trajectories',
                        help='number of highlights trajectories to obtain', type=int, default=5)
    parser.add_argument('-l', '--trajectory_length',
                        help='length of highlights trajectories ', type=int, default=10)
    parser.add_argument('-v', '--verbose', help='print information to the console',
                        action='store_true')
    parser.add_argument('-overlapLim', '--overlay_limit', help='# overlaping', type=int,
                        default=3)
    parser.add_argument('-minGap', '--minimum_gap', help='minimum gap between trajectories',
                        type=int, default=0)
    parser.add_argument('-rand', '--randomized', help='randomize order of summary trajectories',
                        type=bool, default=True)
    parser.add_argument('-impMeth', '--importance_type',
                        help='importance by state or trajectory', default='single_state')
    parser.add_argument('-impState', '--state_importance',
                        help='method calculating state importance', default='second')
    parser.add_argument('-loadTrace', '--load_last_traces',
                        help='load previously generated traces', type=bool, default=False)
    parser.add_argument('-loadTraj', '--load_last_trajectories',
                        help='load previously generated trajectories', type=bool, default=False)
    args = parser.parse_args()

    """agent parameters"""
    args.agent_config = {
        # "__class__": "<class 'rl_agents.trained.simple.open_loop.OpenLoopAgent'>",
        "__class__": "<class 'rl_agents.trained.deep_q_network.pytorch.DQNAgent'>",
        "gamma": 0.7,
    }
    args.num_episodes = 1  # max 2000 (defined in configuration.py)
    args.fps = 2
    args.verbose = True
    args.record = 'all'
    args.show_score_bar = False
    args.clear_results = True

    """Highlight parameters"""
    args.n_traces = 10
    args.trajectory_importance = "single_state"
    args.state_importance = "second"
    args.num_trajectories = 5
    args.trajectory_length = 10
    args.fade_duration = 2
    args.minimum_gap = 0
    args.overlay_limit = 3
    args.allowed_similar_states = 3
    args.highlights_selection_method = 'importance_scores'  # 'scores_and_similarity', 'similarity'
    args.load_traces = False
    args.load_trajectories = False
    args.randomized = True

    # RUN
    args.name = args.agent_config["__class__"].split('.')[-1][:-2]
    get_highlights(args)
