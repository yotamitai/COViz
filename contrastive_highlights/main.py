import json
import random
from datetime import datetime
from os import makedirs, getpid
from os.path import join, abspath
from pathlib import Path
from types import SimpleNamespace

from contrastive_highlights.common.utils import save_contrastive_videos, create_highlights_videos, \
    save_image, make_clean_dirs, save_frames
from contrastive_highlights.contrastive import get_contrastive_highlights
from contrastive_highlights.ffmpeg import ffmpeg_highlights
from contrastive_highlights.get_highlights import get_traces_and_highlights, save_highlights
from contrastive_highlights.mark_agents import get_marked_frames


def output_and_metadata(args):
    log_name = 'run_{}_{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), getpid())
    args.output_dir = join(abspath('results'), log_name)
    makedirs(args.output_dir)
    with Path(join(args.output_dir, 'metadata.json')).open('w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


def main(args):
    output_and_metadata(args)
    """get environment and agent configs"""
    with open(args.agent_config, 'r') as f:
        args.config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    args.videos_dir = join(args.output_dir, "Highlight_Videos")
    args.frames_dir = join(args.output_dir, 'Highlight_Frames')

    traces, state_dict, summary_states = get_traces_and_highlights(args)
    highlights = get_contrastive_highlights(traces, state_dict, summary_states, args)

    """randomize order"""
    if args.randomized: random.shuffle(highlights)

    """obtain and mark disagreement frames"""
    # static_position = [164, 66]
    frames = get_marked_frames(highlights, args.config.highlights, colors=args.colors)

    """save highlight frames"""
    save_frames(frames, args.frames_dir)

    """generate highlights video"""
    img_shape = frames[0][0].shape
    save_highlights(img_shape, "Contrastive_Highlights", args)

    # save_contrastive_videos(frames, args.output_dir, args.fps)

