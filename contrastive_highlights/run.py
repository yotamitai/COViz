import argparse

from contrastive_highlights.main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HIGHLIGHTS')
    parser.add_argument('--load_dir', help='path to existing execution traces')
    # parser.add_argument('-num_ep', '--num_episodes', help='number of episodes to run', type=int,
    #                     default=1)
    # parser.add_argument('-fps', '--fps', help='summary video fps', type=int, default=1)
    # parser.add_argument('-n', '--n_traces', help='number of traces to obtain', type=int,
    #                     default=10)
    # parser.add_argument('-k', '--num_trajectories',
    #                     help='number of highlights trajectories to obtain', type=int, default=5)
    # parser.add_argument('-l', '--trajectory_length',
    #                     help='length of highlights trajectories ', type=int, default=10)
    # parser.add_argument('-v', '--verbose', help='print information to the console',
    #                     action='store_true')
    # parser.add_argument('-overlapLim', '--overlay_limit', help='# overlaping', type=int,
    #                     default=3)
    # parser.add_argument('-minGap', '--minimum_gap', help='minimum gap between trajectories',
    #                     type=int, default=0)
    # parser.add_argument('-rand', '--randomized', help='randomize order of summary trajectories',
    #                     type=bool, default=True)
    # parser.add_argument('-impMeth', '--importance_type',
    #                     help='importance by state or trajectory', default='single_state')
    # parser.add_argument('-impState', '--state_importance',
    #                     help='method calculating state importance', default='second')
    # parser.add_argument('-loadTrace', '--load_last_traces',
    #                     help='load previously generated traces', type=bool, default=False)
    # parser.add_argument('-loadTraj', '--load_last_trajectories',
    #                     help='load previously generated trajectories', type=bool, default=False)
    args = parser.parse_args()

    # RUN
    main(args)