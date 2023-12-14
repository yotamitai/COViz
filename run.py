from os.path import abspath

import argparse

from counterfactual_outcomes.main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HIGHLIGHTS')
    parser.add_argument('--traces-path', help='path to existing execution traces')
    parser.add_argument('--n_traces', help='number of traces to obtain', default=3)
    parser.add_argument('--state-importance', help='state importance method', default='second')
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--k-steps', help="trajectory steps to proceed", default=5)
    parser.add_argument('--overlay', help='# overlaping', default=3)
    parser.add_argument('--minimum-gap', help='minimum gap between trajectories', default=0)
    parser.add_argument('--randomized', help='randomize trajectories order', default=False)
    parser.add_argument('--fps', help='summary video fps', type=int, default=3)
    parser.add_argument('--fade-duration', help='fade-in fade-out duration', type=int, default=2)
    parser.add_argument('--agent-interface', help='model-agent interface', default="Highway")
    parser.add_argument('--importance-method', help='model-agent interface', default="lastState")
    parser.add_argument('--highlights_div', type=bool, default=False)
    parser.add_argument('--div_coefficient', type=int, default=2)
    parser.add_argument('--num_highlights', type=int, default=5)
    args = parser.parse_args()

    args.interface = "Highway"
    args.pause = 3
    args.fps = 3
    args.contra_action_counter = 1
    args.num_highlights = 15
    args.min_traj_len = 10
    args.overlay = args.min_traj_len//2
    args.no_mark = True

    args.importance_method = 'lastState'  # highlights_secondbest / highlights_worst / frequency

    """If existing (saved) traces"""
    if args.traces_path:
        main(args)
    else:
        args.multi_head = True
        # RUN
        if args.interface == "Highway":
            from counterfactual_outcomes.interfaces.Highway.highway_interface import highway_config
            args = highway_config(args)

        main(args)
