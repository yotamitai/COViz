from collections import defaultdict

import json
import numpy as np
import random
from datetime import datetime
from os import makedirs, getpid
from os.path import join, abspath
from pathlib import Path

from contrastive_highlights.common import save_traces, log_msg, load_traces, \
    get_highlight_traj_indxs, save_highlights, save_frames
from contrastive_highlights.contrastive_online import online_comparison
from contrastive_highlights.contrastive_online_RD import online_comparison_RD
from contrastive_highlights.get_agent import get_config, get_agent


def output_and_metadata(args):
    log_name = 'run_{}_{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"), getpid())
    args.output_dir = join(abspath('results'), log_name)
    makedirs(args.output_dir)
    with Path(join(args.output_dir, 'metadata.json')).open('w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


def contrastive_online(args):
    args.config = get_config(args.load_path, args.config_filename, changes=args.config_changes)

    if args.interface == "Highway":
        env1, agent1 = get_agent(args)
        evaluation1 = agent1.interface.evaluation(env1, agent1)
        env2, agent2 = get_agent(args)
        evaluation2 = agent2.interface.evaluation(env2, agent2)
        env1.args = args
        env2.args = args
        if args.multi_head:
            traces = online_comparison_RD(env1, agent1, env2, agent2, args, evaluation1=evaluation1,
                              evaluation2=evaluation2)
        else:
            traces = online_comparison(env1, agent1, env2, agent2, args, evaluation1=evaluation1,
                                   evaluation2=evaluation2)
        env1.close()
        env2.close()
        evaluation1.close()
        evaluation2.close()
    else:
        NotImplementedError
    return traces


def rank_trajectories(traces, method):
    for t in traces:
        for i in range(len(t.states)):
            contra_states = t.contrastive[i].states
            max_trace_state = len(t.states) - 1
            max_contrastive_state = contra_states[-1].id[1]
            end_state = min(i + t.k_steps, max_trace_state, max_contrastive_state)
            t.contrastive[i].traj_end_state = end_state
            if method == "lastState":
                state1 = t.states[end_state]
                state2 = [x for x in contra_states if x.id[1] == end_state][0]
                """the value of the state is defined by the best available action from it"""
                t.contrastive[i].importance = abs(
                    max(state1.observed_actions) - max(state2.observed_actions))
            elif method=="highlights":
                # defined by second-best importance
                action_values = t.states[i].observed_actions
                t.contrastive[i].importance = np.max(action_values) - \
                                              np.partition(action_values.flatten(), -2)[-2]






def get_top_k_diverse(traces, args):
    """
    sort contrastive trajectories by importance and return the top k important ones
    diversity measure - check intersection between trajectory indexes
    """
    all_contrastive_trajs = []
    for t in traces: all_contrastive_trajs += t.contrastive
    all_contrastive_trajs.sort(key=lambda x: x.importance)

    top_k, seen = [], defaultdict(lambda: [])
    while all_contrastive_trajs:
        current = all_contrastive_trajs[-1]
        if current.id[1] in seen[current.id[0]]:
            all_contrastive_trajs.pop(-1)
        else:
            top_k.append(current)
            idxs = list(range(current.id[1] + 1 - args.overlay, current.id[1] + args.overlay))
            new_set = set(list(seen[current.id[0]]) + idxs)
            seen[current.id[0]] = new_set
            if len(top_k) == args.num_highlights: break
            all_contrastive_trajs.pop(-1)

    return top_k


def main(args):
    output_and_metadata(args)
    """get environment and agent configs"""
    args.videos_dir = join(args.output_dir, "Highlight_Videos")
    args.frames_dir = join(args.output_dir, 'Highlight_Frames')

    """get traces"""
    traces = load_traces(args.traces_path) if args.traces_path else contrastive_online(args)
    log_msg(f'Obtained traces', args.verbose)

    """save traces"""
    save_traces(traces, args.output_dir)
    if not args.traces_path: save_traces(traces, abspath('results'))
    log_msg(f'Saved traces', args.verbose)

    """rank trajectories"""
    rank_trajectories(traces, args.importance_method) #TODO change importance by highlights

    """select top k diverse trajectories"""
    highlights = get_top_k_diverse(traces, args)
    if not highlights:
        log_msg(f'No disagreements found', args.verbose)
        return
    log_msg(f'Obtained {len(highlights)} contrastive highlights', args.verbose)

    """randomize order"""
    if args.randomized: random.shuffle(highlights)
    id_list = []
    for hl in highlights:
        t,s = hl.id
        id_list.append(tuple([hl.id, traces[t].RD_vals[s]]))
    save_traces(id_list, abspath('results'), name="Selected_Highlights.pkl")
    # save_traces([x.id for x in highlights], abspath('results'), name="Selected_Indexes.pkl")

    """obtain trajectory indexes"""
    traj_indxs = get_highlight_traj_indxs(highlights)

    """mark contrastive frames"""
    highlight_frames, contra_rel_idxs = {}, {}
    for hl, indxs in traj_indxs.items():
        trace = traces[hl[0]]
        highlight_frames[hl], contra_rel_idxs[hl] = trace.mark_frames(hl[1], indxs, no_mark=args.no_mark)

    """save highlight frames"""
    save_frames(highlight_frames, args.frames_dir) if args.no_mark else \
        save_frames(highlight_frames, args.frames_dir, contra_rel_idxs)

    """generate highlights video"""
    img_shape = highlight_frames[hl][0].shape
    save_highlights(img_shape, len(highlight_frames), args)
    log_msg(f'Highlights Saved', args.verbose)

    """ writes results to files"""
    log_msg(f'\nResults written to:\n\t\'{args.output_dir}\'', args.verbose)
