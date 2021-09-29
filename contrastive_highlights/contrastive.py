import argparse
import json
import random

import gym
import pandas as pd
from os.path import join

from contrastive_highlights.ffmpeg import merge_and_fade
from get_agent import get_agent
from get_traces import get_traces
from common.utils import pickle_save, pickle_load, create_video
from highlights_state_selection import compute_states_importance, highlights, highlights_div
from get_trajectories import states_to_trajectories, trajectories_by_importance, \
    get_trajectory_images

from common.utils import unserialize_states


def get_contrastive_highlights(args):
    """get contrastive highlights from """

    """Load traces and state dictionary"""
    traces = pickle_load(join(args.output_dir, 'Traces.pkl'))
    states = pickle_load(join(args.output_dir, 'States.pkl'))
    with open(join(args.output_dir, 'summary_states.json'), 'r') as f:
        summary_states = unserialize_states(json.load(f))

    """get the traces needed from the summary states"""

    """sort important states by relative trace"""

    """run the selected traces"""

    """in each trace stop at the relevant important state"""

    """split off for a set number of steps"""
    """     use inverse action to constrain agent"""

    """revert back to important state and continue (unless this is the last one)"""


    return
