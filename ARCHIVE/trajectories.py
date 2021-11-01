import numpy as np


class CounterfactualTrajectory(object):


    def __init__(self, cf_index, a1_states, a2_states, horizon, episode, i, action_values):
        self.a1_states = a1_states
        self.a2_states = a2_states
        self.episode = episode
        self.trajectory_index = i
        self.horizon = horizon
        self.cf_index = cf_index
        self.importance = None
        self.state_importance_list = []
        self.action_values = action_values
        self.importance_funcs = {
            "max_min": trajectory_importance_max_min,
            "max_avg": trajectory_importance_max_min,
            "avg": trajectory_importance_max_min,
            "avg_delta": trajectory_importance_max_min,
        }

    # def calculate_trajectory_importance(self, trace, i, trajectory_importance, state_importance,
    #                                     a1_q_max, a2_q_max):
    #     """calculate trajectory score"""
    #     s_i, e_i = min(self.a1_states), max(self.a1_states) + 1
    #     a1_states, a2_states = trace.states[s_i: e_i], trace.a2_trajectories[i]
    #     self.trajectory_importance = trajectory_importance
    #     self.state_importance = state_importance
    #     if trajectory_importance == "last_state":
    #         return self.trajectory_importance_last_state(a1_states[-1], a2_states[-1],
    #                                                      a1_q_max, a2_q_max)
    #     else:
    #         return self.get_trajectory_importance(trajectory_importance, state_importance,
    #                                               a1_q_max, a2_q_max)
    #
    # def get_trajectory_importance(self, trajectory_importance, state_importance):
    #     """state values"""
    #     s1_a1_vals = np.array(self.a1_s_a_values)
    #     s1_a2_vals = np.array(self.a2_values_for_a1_states)
    #     s2_a1_vals = np.array(self.a1_values_for_a2_states)
    #     s2_a2_vals = np.array(self.a2_s_a_values)
    #     """calculate value of all individual states in both trajectories,
    #      as ranked by both trained"""
    #     traj1_importance_of_states = [
    #         self.state_disagreement_score(s1_a1_vals[i], s1_a2_vals[i], state_importance) for i
    #         in range(len(s1_a1_vals))]
    #     traj2_importance_of_states = [
    #         self.state_disagreement_score(s2_a1_vals[i], s2_a2_vals[i], state_importance) for i
    #         in range(len(s2_a2_vals))]
    #     """calculate score of trajectories"""
    #     traj1_score = self.importance_funcs[trajectory_importance](traj1_importance_of_states)
    #     traj2_score = self.importance_funcs[trajectory_importance](traj2_importance_of_states)
    #     """return the difference between them. bigger == greater disagreement"""
    #     return abs(traj1_score - traj2_score)
    #
    # def trajectory_importance_last_state(self, s1, s2, a1_q_max, a2_q_max):
    #     """state values"""
    #     if s1.state.tolist() == s2.state.tolist(): return 0
    #     s1_a1_vals = self.a1_s_a_values[-1] / a1_q_max
    #     s1_a2_vals = self.a2_values_for_a1_states[-1] / a2_q_max
    #     s2_a1_vals = self.a1_values_for_a2_states[-1] / a1_q_max
    #     s2_a2_vals = self.a2_s_a_values[-1] / a2_q_max
    #     """the value of the state is defined by the best available action from it"""
    #     s1_score = max(s1_a1_vals) * self.agent_ratio + max(s1_a2_vals)
    #     s2_score = max(s2_a1_vals) * self.agent_ratio + max(s2_a2_vals)
    #     return abs(s1_score - s2_score)
    #
    # def second_best_confidence(self, a1_vals, a2_vals):
    #     """compare best action to second-best action"""
    #     sorted_1 = sorted(a1_vals, reverse=True)
    #     sorted_2 = sorted(a2_vals, reverse=True)
    #     a1_diff = sorted_1[0] - sorted_1[1] * self.agent_ratio
    #     a2_diff = sorted_2[0] - sorted_2[1]
    #     return a1_diff + a2_diff
    #
    # def better_than_you_confidence(self, a1_vals, a2_vals):
    #     a1_diff = (max(a1_vals) - a1_vals[np.argmax(a2_vals)]) * self.agent_ratio
    #     a2_diff = max(a2_vals) - a2_vals[np.argmax(a1_vals)]
    #     return a1_diff + a2_diff
    #
    # def state_disagreement_score(self, s1_vals, s2_vals, importance):
    #     # softmax trick to prevent overflow and underflow
    #     new_s1_vals = s1_vals - s1_vals.max()
    #     new_s2_vals = s2_vals - s2_vals.max()
    #     a1_vals = softmax(new_s1_vals)
    #     a2_vals = softmax(new_s2_vals)
    #     if importance == 'sb':
    #         return self.second_best_confidence(a1_vals, a2_vals)
    #     elif importance == 'bety':
    #         return self.better_than_you_confidence(a1_vals, a2_vals)


def trajectories_by_importance(execution_traces, state_importance, args):
    if args.load_trajectories:
        all_trajectories = pickle_load(join(args.results_dir, 'Trajectories.pkl'))
        if args.verbose: print(f"HIGHLIGHTS {15 * '-' + '>'} Trajectories Loaded")
    else:
        all_trajectories = get_all_trajectories(execution_traces, args.trajectory_length, state_importance)
        pickle_save(all_trajectories, join(args.results_dir, 'Trajectories.pkl'))
        if args.verbose: print(f"HIGHLIGHTS {15 * '-' + '>'} Trajectories Generated")

    sorted_by_method = sorted([(x.importance[args.trajectory_importance], x) for x in all_trajectories],
                              key=lambda y: y[0], reverse=True)
    sorted_trajectories = [x[1] for x in sorted_by_method]
    trajectories_scores = [x[0] for x in sorted_by_method]
    summary_trajectories = trajectory_highlights(sorted_trajectories, trajectories_scores, args.allowed_similar_states,
                                                 args.num_trajectories, args.highlights_selection_method)
    return all_trajectories, summary_trajectories


def get_all_trajectories(traces, length, importance):
    trajectories = []
    for trace in traces:
        for i in range(len(trace.states)):
            if (i + length) <= len(trace.states):
                trajectories.append(Trajectory(trace.states[i:i + length], importance))
    return trajectories


def trajectory_highlights(trajectories, scores, similarity_limit, budget, method):
    summary = [trajectories[0]]
    seen_score = {scores[1]}
    #TODO rethink diversity measures for obtianing trajectories
    if method == 'only_score':
        _, indx = np.unique(scores, return_index=True)
        indx = indx[-1:-budget-1:-1]
        summary = [trajectories[i] for i in indx]
    else:
        for i in range(1, len(trajectories)):
            if method == 'score_and_similarity':
                np.unique(scores, return_inverse=True)
                if scores[i] in seen_score:
                    continue
                else: seen_score.add(scores[i])
            for t in summary:
                if len(set(t.states).intersection(trajectories[i].states)) > similarity_limit:
                    break
            else:
                summary.append(trajectories[i])
            if len(summary) == budget:
                break

        assert len(summary) == budget, "Not enough dis-similar trajectories found"
    return summary


def get_counterfactual_trajectory(trace, idx):
    pass


def states_to_trajectories(cf_method, states, traces):

    for state in states:
        s_id= state[1]
        trace = traces[state[0]]
        # get original trajectory
        original_states = trace.get_trajectory(s_id)
        # get counterfactual trajectory
        cf_states = get_counterfactual_trajectory(trace, state.id[1])



    trajectories = []
    for states in states_list.values():
        trajectories.append(Trajectory(states, importance_dict))
    return trajectories


def trajectory_importance_max_min(states_importance):
    """ computes the importance of the trajectory, according to max-min approach:
     delta(max state, min state) """
    return max(states_importance) - min(states_importance)


def trajectory_importance_max_avg(states_importance):
    """ computes the importance of the trajectory, according to max-avg approach:
     delta(max state, avg) """
    avg = sum(states_importance) / len(states_importance)
    return max(states_importance) - avg


def trajectory_importance_avg(states_importance):
    """ computes the importance of the trajectory, according to avg approach """
    avg = sum(states_importance) / len(states_importance)
    return avg


def trajectory_importance_avg_delta(states_importance):
    """ computes the importance of the trajectory, according to the average delta approach """
    sum_delta = 0
    for i in range(len(states_importance)):
        sum_delta += states_importance[i] - states_importance[i - 1]
    avg_delta = sum_delta / len(states_importance)
    return avg_delta

