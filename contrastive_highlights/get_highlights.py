import argparse
import json
import random

import gym
import pandas as pd
from os.path import join

from contrastive_highlights.ffmpeg import merge_and_fade
from get_agent import get_agent
from get_traces import get_traces
from common.utils import pickle_save, pickle_load, create_video, serialize_states
from highlights_state_selection import compute_states_importance, highlights, highlights_div
from get_trajectories import states_to_trajectories, trajectories_by_importance, \
    get_trajectory_images

def save_videos(summary_states, states, args):
    """Save Highlight videos"""
    frames_dir = join(args.output_dir, 'Highlight_Frames')
    videos_dir = join(args.output_dir, "Highlight_Videos")
    height, width, layers = states[(0, 0)].image.shape
    img_size = (width, height)

    get_trajectory_images(summary_states, states, frames_dir, args.randomized)
    create_video(frames_dir, videos_dir, args.num_trajectories, img_size, args.fps)
    if args.verbose: print(f"HIGHLIGHTS {15 * '-' + '>'} Videos Generated")

    """Merge Highlights to a single video with fade in/ fade out effects"""
    fade_out_frame = args.trajectory_length - args.fade_duration
    merge_and_fade(videos_dir, args.num_trajectories, fade_out_frame, args.fade_duration,
                   args.config.name)


def get_highlights(args):
    if args.load_dir:
        """Load traces and state dictionary"""
        traces = pickle_load(join(args.load_dir, 'Traces.pkl'))
        states = pickle_load(join(args.load_dir, 'States.pkl'))
        if args.verbose: print(f"Highlights {15 * '-' + '>'} Traces & States Loaded")
    else:
        env, agent = get_agent(args)
        env.args = args
        traces, states = get_traces(env, agent, args)
        env.close()
        del gym.envs.registration.registry.env_specs[args.config.env]

    """Save data used for this run in output dir"""
    pickle_save(traces, join(args.output_dir, 'Traces.pkl'))
    pickle_save(states, join(args.output_dir, 'States.pkl'))

    """highlights algorithm"""
    data = {
        'state': list(states.keys()),
        'q_values': [x.observed_actions for x in states.values()]
    }
    q_values_df = pd.DataFrame(data)

    """importance by state"""
    q_values_df = compute_states_importance(q_values_df, compare_to=args.state_importance)
    highlights_df = q_values_df
    state_importance_dict = dict(zip(highlights_df["state"], highlights_df["importance"]))

    """get highlights"""

    """highlights importance by single state importance"""
    summary_states = highlights(highlights_df, traces, args.num_trajectories,
                                args.trajectory_length, args.minimum_gap, args.overlay_limit)

    with open(join(args.output_dir, 'summary_states.json'), 'w') as f:
        json.dump(serialize_states(list(summary_states.keys())), f)
    # TODO highlight-div
    # summary_states = highlights_div(highlights_df, traces, args.num_trajectories,
    #                             args.trajectory_length,
    #                             args.minimum_gap)

    # TODO is saving trajectories necessary?
    # all_trajectories = states_to_trajectories(summary_states, state_importance_dict)
    # summary_trajectories = all_trajectories

    # random highlights
    # summary_trajectories = random.choices(all_trajectories, k=5)

    save_videos(summary_states, states, args)
    return
