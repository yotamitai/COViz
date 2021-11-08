import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from contrastive_highlights.Interfaces.frogger_interface import FroggerInterface
from contrastive_highlights.Interfaces.gym_interface import GymInterface




def get_agent(args):
    """Implement here for specific agent and environment loading scheme"""
    if args.agent_type == "gym":
        interface = GymInterface(args.config, args.output_dir)
    # elif args.agent_type == "frogger":
    else:
        interface = FroggerInterface(args.agent_config, args.output_dir, args.n_traces)
    env, agent = interface.initiate()
    agent.interface = interface
    env.seed(0)
    return env, agent
