# import utils.import_envs  # noqa: F401 pylint: disable=unused-import
#
import json
from os import listdir
from os.path import join


def get_agent(args):
    """Implement here for specific agent and environment loading scheme"""
    if args.interface == "Frogger":
        from contrastive_highlights.interfaces.Frogger.frogger_interface import FroggerInterface
        interface = FroggerInterface(args.config_path, args.config, args.load_path,
                                     args.output_dir, args.n_traces, args.fps)
    # elif args.interface == "gym":
    #     from highlights.interfaces.gym_interface import GymInterface
    #     interface = GymInterface(args.config, args.output_dir)
    else:
        from contrastive_highlights.interfaces.Highway.highway_interface import HighwayInterface
        interface = HighwayInterface(args.config, args.output_dir, args.load_path)
    env, agent = interface.initiate()
    agent.interface = interface
    env.seed(0)
    return env, agent

def get_config(load_path, filename, changes=None):
    config_filename = [x for x in listdir(load_path) if filename in x][0]
    f = open(join(load_path, config_filename))
    config = json.load(f)

    if changes:
        for section in changes:
            for k, v in changes[section].items():
                config[section][k] = v

    return config