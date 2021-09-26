import numpy as np
from scipy.spatial import distance
from bisect import bisect
from bisect import insort_left


def compute_states_importance(states_q_values_df, compare_to='worst'):
    if compare_to == 'worst':
        states_q_values_df['importance'] = states_q_values_df['q_values'].apply(
            lambda x: np.max(x) - np.min(x))
    elif compare_to == 'second':
        states_q_values_df['importance'] = states_q_values_df['q_values'].apply(
            lambda x: np.max(x) - np.partition(x.flatten(), -2)[-2])
    return states_q_values_df


def highlights(state_importance_df, exec_traces, budget, context_length, minimum_gap=0,
               overlay_limit=0):
    """generate highlights summary"""

    sorted_df = state_importance_df.sort_values(['importance'], ascending=False)
    summary_states, summary_traces, state_trajectories = [], [], {}
    seen_indexes, seen_importance = {x: [] for x in range(len(exec_traces))}, []

    """for each state by importance"""
    for index, row in sorted_df.iterrows():
        state = row['state']

        """unique score for frogger"""
        if row["importance"] in seen_importance:
            continue
        else:
            seen_importance.append(row["importance"])

        trace_len = len(exec_traces[state[0]].states)
        lower, upper = get_relevant_range(state[1], trace_len, context_length, minimum_gap,
                                          overlay_limit)
        if lower not in seen_indexes[state[0]] and upper not in seen_indexes[state[0]]:
            seen_indexes[state[0]] += list(range(lower, upper + 1))
            summary_states.append(state)

        if len(summary_states) == budget:
            break

        #
        # trajectories = {}
        # for trace_idx, trace in enumerate(exec_traces):
        #     if state in trace.states:
        #         state_index = trace.states.index(state)
        #         trace_len = len(trace.states)
        #         lower, upper = get_relevant_range(state_index, trace_len, context_length,
        #                                           minimum_gap, overlay_limit)
        #         """check if these states are not neighbours of previously seen states"""
        #         for seen_state in summary_states:
        #             # if [1 for x in trace.states[lower:upper] if x == seen_state]:
        #             if seen_state[0] != trace_idx:
        #                 break
        #             else:
        #                 if seen_state[1] in trace.states[lower:upper]:
        #                     break
        #                 else:
        #                     trajectories[trace_idx] = state_index
        #         if not summary_states:
        #             trajectories[trace_idx] = state_index
        #
        # """if no siutable trajectories found - try next state"""
        # if not trajectories:
        #     continue
        # else:
        #     state_trajectories[state] = trajectories
        #
        # """once a trace is obtained, get the state index in it"""
        # summary_states.append(state)
        # summary_traces.append(list(trajectories.keys()))
        # if len(summary_states) == budget:
        #     break

    summary_state_trajectories = {}
    for t_i, s_i in summary_states:
        t = exec_traces[t_i].states
        lower, upper = get_relevant_range(s_i, len(t), context_length)
        summary_state_trajectories[(t_i, s_i)] = t[lower:upper]
    return summary_state_trajectories


def get_relevant_range(indx, lst_len, range_len, gap=0, overlay=0):
    if indx - range_len < 0:
        lb = 0
        ub = range_len - 1 + gap - overlay
    elif indx + range_len > lst_len:
        ub = lst_len - 1
        lb = lst_len - 1 - range_len - gap + overlay
    else:
        lb = indx - int(range_len / 2) - gap + overlay
        ub = indx + int(range_len / 2) + gap - overlay
    return lb, ub


def highlights_div(state_importance_df, exec_traces, budget, context_length, minimum_gap,
                   distance_metric=distance.euclidean, percentile_threshold=3,
                   subset_threshold=1000):
    ''' generate highlights-div  summary
    :param state_importance_df: dataframe with 2 columns: state and importance score of the state
    :param budget: allowed length of summary - note this includes only the important states, it
    doesn't count context around them
    :param context_length: how many states to show around the chosen important state (e.g., if
    context_lenght=10, we will show 10 states before and 10 states after the important state
    :param minimum_gap: how many states should we skip after showing the context for an important
    state. For example, if
    we chose state 200, and the context length is 10, we will show states 189-211. If
    minimum_gap=10, we will not consider states 212-222 and states 178-198
    because they are too close
    :param distance_metric: metric to use for comparing states (function)
    :param percentile_threshold: what minimal distance to allow between states in summary
    :param subset_threshold: number of random states to be used as basis for the div-threshold
    :return: a list with the indices of the important states, and a list with all
    summary states (includes the context)
    '''

    min_state = state_importance_df['state'].values.min()
    max_state = state_importance_df['state'].values.max()

    state_features = state_importance_df['features'].values
    state_features = np.random.choice(state_features, size=subset_threshold, replace=False)
    distances = []
    for i in range(len(state_features - 1)):
        for j in range(i + 1, len(state_features)):
            distance = distance_metric(state_features[i], state_features[j])
            distances.append(distance)
    distances = np.array(distances)
    threshold = np.percentile(distances, percentile_threshold)
    print('threshold:', threshold)

    sorted_df = state_importance_df.sort_values(['importance'], ascending=False)
    summary_states = []
    summary_states_with_context = []
    num_chosen_states = 0
    for index, row in sorted_df.iterrows():
        state_index = row['state']
        index_in_summary = bisect(summary_states, state_index)
        # print('state: ', state_index)
        # print('index in summary: ', index_in_summary)
        # print('summary: ', summary_states)
        state_before = None
        state_after = None
        if index_in_summary > 0:
            state_before = summary_states[index_in_summary - 1]
        if index_in_summary < len(summary_states):
            state_after = summary_states[index_in_summary]
        if state_after is not None:
            if state_index + context_length + minimum_gap > state_after:
                continue
        if state_before is not None:
            if state_index - context_length - minimum_gap < state_before:
                continue

        # if num_chosen_states < budget:
        #     insort_left(summary_states,state_index)
        #     num_chosen_states += 1

        # compare to most similar state
        most_similar_state, min_distance = find_similar_state_in_summary(state_importance_df,
                                                                         summary_states_with_context,
                                                                         row['features'],
                                                                         distance_metric)
        if most_similar_state is None:
            insort_left(summary_states, state_index)
            num_chosen_states += 1
            print('summary_states:', summary_states)

        else:
            # similar_state_importance = state_importance_df.loc[state_importance_df['state'] == most_similar_state].iloc[0].importance
            # if row['importance'] > similar_state_importance:
            if min_distance > threshold:
                insort_left(summary_states, state_index)
                num_chosen_states += 1
                print('summary_states:', summary_states)
                # print('took')
            # else:
            #     print(state_index)
            #     print('skipped')

        # recalculate the context states
        summary_states_with_context = []
        for state in summary_states:
            left_index = max(state - context_length, min_state)
            right_index = min(state + context_length, max_state) + 1
            summary_states_with_context.extend((range(left_index, right_index)))

        if len(summary_states) == budget:
            break

    return summary_states, summary_states_with_context


def find_similar_state_in_summary(state_importance_df, summary_states, new_state, distance_metric,
                                  distance_threshold=None):
    most_similar_state = None
    minimal_distance = 10000000
    for state in summary_states:
        state_features = state_importance_df.loc[state_importance_df['state'] == state].iloc[
            0].features
        distance = distance_metric(state_features, new_state)
        if distance < minimal_distance:
            minimal_distance = distance
            most_similar_state = state
    if distance_threshold is None:
        return most_similar_state, minimal_distance
    elif minimal_distance < distance_threshold:
        return most_similar_state, minimal_distance
    return None
