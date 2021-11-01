import json
import random
from datetime import datetime
from os import makedirs, getpid
from os.path import join, abspath
from pathlib import Path
from types import SimpleNamespace

from contrastive_highlights.common.utils import make_same_length, save_disagreements
from contrastive_highlights.contrastive import get_contrastive_highlights
from contrastive_highlights.ffmpeg import merge_and_fade
from contrastive_highlights.get_highlights import get_traces_and_highlights
from contrastive_highlights.mark_agents import get_and_mark_frames


def output_and_metadata(args):
    log_name = 'run_{}_{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), getpid())
    args.output_dir = join(abspath('results'), log_name)
    makedirs(args.output_dir)
    with Path(join(args.output_dir, 'metadata.json')).open('w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


def save_videos():
    pass


def main(args):
    output_and_metadata(args)
    """get environment and agent configs"""
    with open(args.agent_config, 'r') as f:
        args.config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    traces, state_dict, summary_states = get_traces_and_highlights(args)
    contrastive = get_contrastive_highlights(traces, state_dict, summary_states, args)

    """make all trajectories the same length"""
    contrastive = make_same_length(contrastive, args.horizon, traces)

    """randomize order"""
    if args.randomized: random.shuffle(contrastive)

    """obtain and mark disagreement frames"""
    a1_contrastive_frames, a2_contrastive_frames = \
        get_and_mark_frames(contrastive, traces, agent_position=[164, 66], color=args.color)

    """save disagreement frames"""
    video_dir = save_disagreements(a1_contrastive_frames, a2_contrastive_frames,
                                   args.output, args.fps)

    """generate video"""
    fade_duration = 2
    fade_out_frame = args.horizon - fade_duration + 11  # +11 from pause in save_disagreements
    # side_by_side_video(video_dir, args.n_disagreements, fade_out_frame, name)
    merge_and_fade(video_dir, args.n_disagreements, fade_out_frame, name)







