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


def highlights(state_importance_df, trace_lens, budget, context_length, minimum_gap):
    ''' generate highlights summary
    :param state_importance_df: dataframe with 2 columns: state and importance score of the state
    :param budget: allowed length of summary - note this includes only the important states, it doesn't count context
    around them
    :param context_length: how many states to show around the chosen important state (e.g., if context_lenght=10, we
    will show 10 states before and 10 states after the important state
    :param minimum_gap: how many states should we skip after showing the context for an important state. For example, if
    we chose state 200, and the context length is 10, we will show states 189-211. If minimum_gap=10, we will not
    consider states 212-222 and states 178-198 because they are too close
    :return: a list with the indices of the important states, and a list with all summary states (includes the context)
    '''
    sorted_df = state_importance_df.sort_values(['importance'], ascending=False)
    summary_states = []
    for index, row in sorted_df.iterrows():

        state_index = row['state']
        index_in_summary = bisect(summary_states, state_index)
        state_before = None
        state_after = None
        if index_in_summary > 0:
            state_before = summary_states[index_in_summary-1]
        if index_in_summary < len(summary_states):
            state_after = summary_states[index_in_summary]
        if state_after is not None:
            if state_index[0] == state_after[0]:
                if state_index[1]+context_length+minimum_gap > state_after[1]:
                    continue
        if state_before is not None:
            if state_index[0] == state_before[0]:
                if state_index[1]-context_length-minimum_gap < state_before[1]:
                    continue
        insort_left(summary_states,state_index)
        if len(summary_states) == budget:
            break

    summary_states_with_context = {}
    for state in summary_states:
        s, e = max(state[1]-context_length,0), min(state[1]+context_length, trace_lens[state[0]]-1)
        summary_states_with_context[state] = [(state[0], x) for x in (range(s,e))]
    return summary_states_with_context



def find_similar_state_in_summary(state_importance_df, threshold, summary_states, new_state):
    most_similar_state = None
    minimal_distance = threshold
    for state in summary_states:
        state_features = state_importance_df.loc[state_importance_df['state'] == state].iloc[
            0].features
        distance = sum(state_features - new_state)
        if distance < minimal_distance:
            minimal_distance = distance
            most_similar_state = state
    return most_similar_state, minimal_distance


def highlights_div(state_importance_df, trace_lens, budget, context_length, minimum_gap,
                   threshold=50000):
    ''' generate highlights-div  summary
    :param state_importance_df: dataframe with 2 columns: state and importance score of the state
    :param budget: allowed length of summary - note this includes only the important states, it doesn't count context
    around them
    :param context_length: how many states to show around the chosen important state (e.g., if context_lenght=10, we
    will show 10 states before and 10 states after the important state
    :param minimum_gap: how many states should we skip after showing the context for an important state. For example, if
    we chose state 200, and the context length is 10, we will show states 189-211. If minimum_gap=10, we will not
    consider states 212-222 and states 178-198 because they are too close
    :param distance_metric: metric to use for comparing states (function)
    :param percentile_threshold: what minimal distance to allow between states in summary
    :param subset_threshold: number of random states to be used as basis for the div-threshold
    :return: a list with the indices of the important states, and a list with all summary states (includes the context)
    '''
    sorted_df = state_importance_df.sort_values(['importance'], ascending=False)

    summary_states = []
    summary_states_with_context = []
    num_chosen_states = 0
    for index, row in sorted_df.iterrows():
        state_index = row['state']
        index_in_summary = bisect(summary_states, state_index)
        state_before = None
        state_after = None
        if index_in_summary > 0:
            state_before = summary_states[index_in_summary - 1]
        if index_in_summary < len(summary_states):
            state_after = summary_states[index_in_summary]
        if state_after is not None:
            if state_index[0] == state_after[0]:
                if state_index[1] + context_length + minimum_gap > state_after[1]:
                    continue
        if state_before is not None:
            if state_index[0] == state_before[0]:
                if state_index[1] - context_length - minimum_gap < state_before[1]:
                    continue

        most_similar_state, min_distance = \
            find_similar_state_in_summary(state_importance_df, threshold,
                                          summary_states_with_context, row['features'])
        if most_similar_state is None:
            insort_left(summary_states, state_index)
            num_chosen_states += 1
            print('summary_states:', summary_states)

        else:
            if min_distance > threshold:
                insort_left(summary_states, state_index)
                num_chosen_states += 1
                print('summary_states:', summary_states)

        summary_states_with_context = {}
        for state in summary_states:
            s, e = max(state[1] - context_length, 0), min(state[1] + context_length,
                                                          trace_lens[state[0]] - 1)
            summary_states_with_context[state] = [(state[0], x) for x in (range(s, e))]

        if len(summary_states) == budget:
            break

    return summary_states_with_context
