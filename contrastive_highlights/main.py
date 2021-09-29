import json
from datetime import datetime
from os import makedirs, getpid
from os.path import join, abspath
from pathlib import Path
from types import SimpleNamespace

from contrastive_highlights.contrastive import get_contrastive_highlights
from contrastive_highlights.get_highlights import get_highlights



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

    if not args.load_highlights: get_highlights(args)
    get_contrastive_highlights(args)
    save_videos()







