import argparse
import json
import random

import gym
import pandas as pd
from os.path import join
import numpy as np

from contrastive_highlights.ffmpeg import ffmpeg_highlights
from get_agent import get_agent
from get_traces import get_traces
from common.utils import pickle_save, pickle_load, serialize_states, \
    create_highlights_videos
from highlights_state_selection import compute_states_importance, highlights, highlights_div
from get_trajectories import states_to_trajectories, trajectories_by_importance, \
    get_trajectory_images


def save_highlights(img_shape, hl_name, args):
    """Save Highlight videos"""
    height, width, layers = img_shape
    img_size = (width, height)

    hl_len = create_highlights_videos(args.frames_dir, args.videos_dir, args.num_trajectories,
                                      img_size, args.fps, pause=args.pause)
    if args.verbose: print(f"HIGHLIGHTS {15 * '-' + '>'} Videos Generated")

    """Merge Highlights to a single video with fade in/ fade out effects"""
    fade_out_frame = hl_len - args.fade_duration
    ffmpeg_highlights(hl_name, args.videos_dir, args.num_trajectories, fade_out_frame,
                      args.fade_duration)


def get_traces_and_highlights(args):
    if args.load:
        """Load traces and state dictionary"""
        traces = pickle_load(join(args.load, 'Traces.pkl'))
        states = pickle_load(join(args.load, 'States.pkl'))
        if args.verbose: print(f"Traces Loaded")
    else:
        env, agent = get_agent(args)
        env.args = args
        traces, states = get_traces(env, agent, args)
        env.close()
        if args.agent_type == "frogger":
            del gym.envs.registration.registry.env_specs[env.spec.id]
        if args.verbose: print(f"Traces Generated")

    """Save data used for this run in output dir"""
    pickle_save(traces, join(args.output_dir, 'Traces.pkl'))
    pickle_save(states, join(args.output_dir, 'States.pkl'))

    """importance by state"""
    a, b, c = states[(0, 0)].image.shape
    data = {'state': list(states.keys()),
            'q_values': [x.observed_actions for x in states.values()],
            'features': [x.image.reshape(a * b * c) for x in states.values()]}

    if args.highlights_div:
        i = len(traces[0].states) // 2
        threshold = args.div_coefficient * (
            sum(states[(0, i)].image.reshape(a * b * c) - states[(0, i + 1)].image.reshape(
                a * b * c)))

    q_values_df = pd.DataFrame(data)
    q_values_df = compute_states_importance(q_values_df, compare_to=args.state_importance)
    highlights_df = q_values_df
    state_importance_dict = dict(zip(highlights_df["state"], highlights_df["importance"]))

    """highlights by single state importance"""
    trace_lengths = {k: len(v.states) for k, v in enumerate(traces)}
    if args.highlights_div:
        summary_states = highlights_div(highlights_df, trace_lengths, args.num_trajectories,
                                        args.trajectory_length, args.minimum_gap,
                                        threshold=threshold)
    else:
        summary_states = highlights(highlights_df, trace_lengths, args.num_trajectories,
                                    args.trajectory_length, args.minimum_gap)

    with open(join(args.output_dir, 'summary_states.json'), 'w') as f:
        json.dump(serialize_states(list(summary_states.keys())), f)

    # TODO is saving trajectories necessary?
    # all_trajectories = states_to_trajectories(summary_states, state_importance_dict)
    # summary_trajectories = all_trajectories

    # random highlights
    # summary_trajectories = random.choices(all_trajectories, k=5)

    # get_trajectory_images(summary_states, states, args.frames_dir, args.randomized)
    # img_shape = states[(0,0)].image.shape
    # save_highlights(img_shape, "Original Highlights", args)
    return traces, states, summary_states
