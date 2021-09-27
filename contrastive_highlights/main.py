import json
from datetime import datetime
from os import makedirs, getpid
from os.path import join, abspath
from pathlib import Path

from contrastive_highlights.ffmpeg import merge_and_fade
from contrastive_highlights.get_highlights import get_highlights
from contrastive_highlights.get_trajectories import get_trajectory_images
from contrastive_highlights.utils import create_video


def save_videos(states, summary_trajectories, args):
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
                   args.name)


def output_and_metadata(args):
    log_name = 'run_{}_{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), getpid())
    args.output_dir = join(abspath('results'), log_name)
    makedirs(args.output_dir)
    with Path(join(args.output_dir, 'metadata.json')).open('w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


def main(args):
    output_and_metadata(args)
    states, summary_trajectories = get_highlights(args)
    save_videos(states, summary_trajectories, args)






