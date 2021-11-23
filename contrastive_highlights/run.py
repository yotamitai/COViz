import argparse

from contrastive_highlights.main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HIGHLIGHTS')
    parser.add_argument('--load', help='path to existing execution traces')
    parser.add_argument('--agent-config', help='path to environment and agent params')
    parser.add_argument('--n_traces', help='number of traces to obtain', default=3)
    parser.add_argument('--state-importance', help='state importance method', default='second')
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--num-trajectories', default=5)
    parser.add_argument('--trajectory-length', help="summary trajectories' length", default=10)
    parser.add_argument('--overlay-limit', help='# overlaping', default=3)
    parser.add_argument('--minimum-gap', help='minimum gap between trajectories', default=0)
    parser.add_argument('--randomized', help='randomize trajectories order', default=False)
    parser.add_argument('--fps', help='summary video fps', type=int, default=3)
    parser.add_argument('--fade-duration', help='fade-in fade-out duration', type=int, default=2)
    parser.add_argument('--agent-type', help='model-agent interface', default="gym")
    parser.add_argument('--highlights_div', type=bool, default=False)
    parser.add_argument('--div_coefficient', type=int, default=2)


    # parser.add_argument('-n', '--n_traces', help='number of traces to obtain', type=int,
    #                     default=10)
    # parser.add_argument('-k', '--num_trajectories',
    #                     help='number of highlights trajectories to obtain', type=int, default=5)


    # parser.add_argument('-impMeth', '--importance_type',
    #                     help='importance by state or trajectory', default='single_state')

    # parser.add_argument('-loadTrace', '--load_last_traces',
    #                     help='load previously generated traces', type=bool, default=False)
    # parser.add_argument('-loadTraj', '--load_last_trajectories',
    #                     help='load previously generated trajectories', type=bool, default=False)
    args = parser.parse_args()
    # args.load = 'results/run_2021-10-14_12:16:23_25541'
    # args.agent_config = "configs/BreakoutNoFrameskip-v4.json"
    # args.agent_type = "gym"
    args.agent_config = "configs/Frogger_Expert.json"
    args.agent_type = "frogger"

    args.div_coefficient = 2
    args.highlights_div = False
    args.trajectory_length = 10
    args.n_traces = 3
    args.contrastive_method = "second"
    args.colors = [255,0]
    args.pause = 3
    args.fps = 2
    # RUN
    main(args)
